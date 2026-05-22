"""Interpreter builds B3 / B4 / V2 (Phase 3.3).

Each build attaches an interpreter to the *shared* RL agent (PLAN §5.0
(C-thin) 부수효과) and is trained by its own loss — never by RL reward.
The common interface is :class:`Build` (P3-3-4 박제):

    build.update(h_agent, ids, lengths) -> {"loss": ..., diagnostics...}
    build.interpreter_parameters()      -> the params its optimizer steps

This file currently implements:

- :class:`Build`    — abstract base.
- :class:`B3Probe`  — direct probe (PLAN §6 B3; P3-3-1 박제: 1-hidden MLP,
  4 separate heads, probe CE on describer-oracle slots, ``h_agent.detach()``).

B4Adapter (Flamingo cross-attn, P3-3-2) and V2ACC (ACC recon, P3-3-5) are
added in the following sub-steps (3.3.2 / 3.3.3) — see SESSION_HANDOFF §9.13.

(C-thin) boundary 1 is enforced inside every ``update``: ``h_agent`` is
detached so the agent core stays on pure RL.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import AgentResampler, GatedCrossAttentionBlock, PerceiverBlock
from .language import (
    CHEESE_DIR_VALUES,
    HEADING_VALUES,
    REGION_COLS,
    REGION_ROWS,
    parse,
)
from .lm import MazeLM, MazeTokenizer


# Slot class counts (B3 head dims) — LANGUAGE_SPEC v0.3.
N_ROW: int = len(REGION_ROWS)          # 3
N_COL: int = len(REGION_COLS)          # 3
N_HEADING: int = len(HEADING_VALUES)   # 9
N_CHEESE: int = len(CHEESE_DIR_VALUES) # 8

# CrossEntropy ignore sentinel (matches torch default).
IGNORE_INDEX: int = -100


# ---- abstract base ------------------------------------------------------


class Build(nn.Module, ABC):
    """Common interface for the B3/B4/V2 interpreter builds (P3-3-4 박제)."""

    @abstractmethod
    def update(
        self,
        h_agent: torch.Tensor,
        ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict:
        """One interpreter learning step on a batch of pairs.

        Args:
            h_agent: (B, d_agent) agent embeddings (detached internally).
            ids:     (B, T) padded describer-sentence token ids.
            lengths: (B,) valid token count per row.

        Returns:
            dict with at least ``"loss"`` (scalar) plus diagnostics.
        """

    @abstractmethod
    def interpreter_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters this build's optimizer should step (never the agent
        core, never the LM core)."""


# ---- B3: direct probe ---------------------------------------------------


class B3Probe(Build):
    """Direct slot probe on ``h_agent`` (PLAN §6 B3; P3-3-1 박제).

    Structure: shared 1-hidden MLP trunk (d_agent → hidden → ReLU) feeding
    four independent linear heads (row, col, heading, cheese). Loss is the
    mean of four cross-entropies against the describer-oracle slot indices
    decoded from ``ids``. The agent is insulated by ``h_agent.detach()``.

    The probe answers "is the slot information *linearly-ish* present in
    h_agent?" — a reference for measurement #2/#3, not an interpreter that
    goes through the LM.

    Args:
        tokenizer: :class:`MazeTokenizer`, to decode ``ids`` → tokens →
                   :func:`split_maze.language.parse` → slot indices.
        d_agent:   IMPALA-CNN embedding width (256).
        hidden:    MLP hidden width (P3-3-1 박제 → 256).
    """

    def __init__(
        self,
        tokenizer: MazeTokenizer,
        *,
        d_agent: int = 256,
        hidden: int = 256,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.trunk = nn.Sequential(nn.Linear(d_agent, hidden), nn.ReLU())
        self.head_row = nn.Linear(hidden, N_ROW)
        self.head_col = nn.Linear(hidden, N_COL)
        self.head_heading = nn.Linear(hidden, N_HEADING)
        self.head_cheese = nn.Linear(hidden, N_CHEESE)

    # ---- forward ----

    def forward(self, h_agent: torch.Tensor) -> dict:
        """Return per-slot logits dict from ``h_agent`` (no detach here —
        callers that want (C-thin) pass a detached tensor; :meth:`update`
        does)."""
        z = self.trunk(h_agent)
        return {
            "row": self.head_row(z),
            "col": self.head_col(z),
            "heading": self.head_heading(z),
            "cheese": self.head_cheese(z),
        }

    # ---- targets ----

    def _targets_from_ids(
        self,
        ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode each row's ids → tokens → parse → slot index.

        Slots that parse fails to recover are marked ``IGNORE_INDEX`` so
        their CE contribution is dropped (rare for oracle-generated pairs).
        """
        B = ids.shape[0]
        row_t = torch.full((B,), IGNORE_INDEX, dtype=torch.long)
        col_t = torch.full((B,), IGNORE_INDEX, dtype=torch.long)
        head_t = torch.full((B,), IGNORE_INDEX, dtype=torch.long)
        chee_t = torch.full((B,), IGNORE_INDEX, dtype=torch.long)

        ids_cpu = ids.detach().cpu()
        len_cpu = lengths.detach().cpu()
        for i in range(B):
            n = int(len_cpu[i].item())
            toks = self.tokenizer.decode(ids_cpu[i, :n].tolist())
            ps = parse(toks)
            if ps.agent_region is not None:
                r, c = ps.agent_region
                if r in REGION_ROWS:
                    row_t[i] = REGION_ROWS.index(r)
                if c in REGION_COLS:
                    col_t[i] = REGION_COLS.index(c)
            if ps.heading in HEADING_VALUES:
                head_t[i] = HEADING_VALUES.index(ps.heading)
            if ps.cheese_dir in CHEESE_DIR_VALUES:
                chee_t[i] = CHEESE_DIR_VALUES.index(ps.cheese_dir)
        return row_t, col_t, head_t, chee_t

    # ---- loss helper ----

    @staticmethod
    def _ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy that is safe when *all* targets are IGNORE_INDEX
        (returns a 0 that still carries a graph edge to ``logits``)."""
        if (target != IGNORE_INDEX).any():
            return F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
        return logits.sum() * 0.0

    # ---- update ----

    def update(
        self,
        h_agent: torch.Tensor,
        ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict:
        # Boundary 1 — agent core is insulated.
        logits = self.forward(h_agent.detach())
        row_t, col_t, head_t, chee_t = self._targets_from_ids(ids, lengths)

        dev = logits["row"].device
        row_t = row_t.to(dev)
        col_t = col_t.to(dev)
        head_t = head_t.to(dev)
        chee_t = chee_t.to(dev)

        l_row = self._ce(logits["row"], row_t)
        l_col = self._ce(logits["col"], col_t)
        l_head = self._ce(logits["heading"], head_t)
        l_chee = self._ce(logits["cheese"], chee_t)
        loss = (l_row + l_col + l_head + l_chee) / 4.0

        return {
            "loss": loss,
            "loss_row": l_row,
            "loss_col": l_col,
            "loss_heading": l_head,
            "loss_cheese": l_chee,
        }

    def interpreter_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.parameters()


# ---- B4: SPLIT-9-pattern adapter ----------------------------------------


class B4Adapter(Build):
    """Flamingo-style adapter build (PLAN §6 B4 ★; P3-3-2 / P3-3-5 박제).

    The faithful re-creation of the SPLIT-9 failure pattern — the build
    whose head-to-head defeat (or not) by V2 is the project's decisive
    test (PLAN §5.1, §5.6 합리화율).

    Pipeline:
        h_agent --AgentResampler--> adapter_tokens (B, n_latents, d_model)
        ids --tok_embed(+pos)--> x
        for each LM block:  x = block(x);  x = gated_xattn(x, adapter_tokens)
        x --ln_f--> --lm_head--> logits
        loss = next-token CE on the describer sentence (next-token only).

    (C-thin) boundaries:
    - **Agent** insulated: ``update`` feeds ``h_agent.detach()`` → only the
      resampler + xattn learn from h_agent, never the agent core.
    - **LM core** protected: the Phase-2 LM is frozen
      (``requires_grad_(False)``) and kept in eval (``train`` override) — a
      deterministic feature extractor, the Flamingo convention. Only the
      resampler + gated xattn blocks are trainable.

    The describer sentence is fed as a *plain* ``[<BOS> ... <EOS>]`` (no
    ``<SUM>``) — B4 uses standard next-token LM, not handle-B encode.

    Args:
        lm:        a Phase-2 :class:`MazeLM` (weights loaded from lm.pt).
        d_agent:   IMPALA-CNN embedding width (256).
        n_latents: adapter tokens emitted by the resampler (P3-3-2 → 16).
        n_kv:      resampler pseudo-token KV slots (≥2).
        n_heads:   attention heads (resampler + xattn).
        resampler_blocks: PerceiverBlock depth inside the resampler.
    """

    def __init__(
        self,
        lm: MazeLM,
        *,
        d_agent: int = 256,
        n_latents: int = 16,
        n_kv: int = 8,
        n_heads: int = 4,
        resampler_blocks: int = 2,
    ):
        super().__init__()
        self.lm = lm
        d_model = lm.config.d_model

        # Freeze the LM core (P3-3-3 + (C-thin) LM 코어 보호).
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.eval()

        self.resampler = AgentResampler(
            d_agent=d_agent,
            d_model=d_model,
            n_latents=n_latents,
            n_kv=n_kv,
            n_heads=n_heads,
            n_blocks=resampler_blocks,
        )
        # One gated cross-attention block after each LM transformer block.
        self.xattn = nn.ModuleList(
            [GatedCrossAttentionBlock(d_model, n_heads) for _ in self.lm.blocks]
        )
        self.pad_id = lm.config.pad_id

    def train(self, mode: bool = True) -> "B4Adapter":
        """Set train mode for the adapter but keep the frozen base LM in
        eval (deterministic feature extractor — Flamingo convention)."""
        super().train(mode)
        self.lm.eval()
        return self

    # ---- forward ----

    def forward(self, ids: torch.Tensor, h_agent: torch.Tensor) -> torch.Tensor:
        """Standard next-token logits with adapter injection.

        Args:
            ids:     (B, T) plain ``[<BOS> ... <EOS>]`` token ids.
            h_agent: (B, d_agent).

        Returns:
            logits: (B, T, vocab) — logit at position t predicts token t+1.
        """
        T = ids.size(1)
        if T > self.lm.config.max_len:
            raise ValueError(
                f"sequence length {T} exceeds LMConfig.max_len "
                f"{self.lm.config.max_len}"
            )
        adapter_tokens = self.resampler(h_agent)             # (B, N, d_model)
        # Replicate MazeLM._transform but inject xattn between blocks, and
        # skip embed_dropout (frozen deterministic base).
        x = self.lm.tok_embed(ids) + self.lm.pos_embed[:, :T]
        for block, xblock in zip(self.lm.blocks, self.xattn):
            x = block(x)
            x = xblock(x, adapter_tokens)
        x = self.lm.ln_f(x)
        return self.lm.lm_head(x)

    # ---- update ----

    def update(
        self,
        h_agent: torch.Tensor,
        ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict:
        # Boundary 1 — agent insulated.
        logits = self.forward(ids, h_agent.detach())          # (B, T, vocab)
        # Next-token CE: logit at t predicts token t+1. pad ignored.
        pred = logits[:, :-1].reshape(-1, logits.size(-1))
        target = ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(pred, target, ignore_index=self.pad_id)
        return {"loss": loss}

    @torch.no_grad()
    def generate(self, h_agent: torch.Tensor, max_len: int = 16) -> torch.Tensor:
        """Greedy autoregressive generation with adapter injection (Phase 4).

        Starts from ``<BOS>`` and decodes token-by-token; the adapter tokens
        (from ``h_agent``) are injected at every LM block on every step. The
        agent→language path for the B4 build (mirrors :meth:`forward`).

        Returns (B, T_gen) ids beginning with <BOS>.
        """
        was_training = self.training
        self.eval()
        try:
            B = h_agent.shape[0]
            device = h_agent.device
            cfg = self.lm.config
            adapter = self.resampler(h_agent)            # (B, N, d_model)
            seq = torch.full((B, 1), cfg.bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            for _ in range(max_len - 1):
                T = seq.shape[1]
                x = self.lm.tok_embed(seq) + self.lm.pos_embed[:, :T]
                for block, xblock in zip(self.lm.blocks, self.xattn):
                    x = block(x)
                    x = xblock(x, adapter)
                x = self.lm.ln_f(x)
                logits = self.lm.lm_head(x[:, -1])       # (B, vocab)
                nxt = logits.argmax(dim=-1)
                nxt = torch.where(finished, torch.full_like(nxt, cfg.pad_id), nxt)
                seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
                finished = finished | (nxt == cfg.eos_id)
                if bool(finished.all()):
                    break
            return seq
        finally:
            if was_training:
                self.train()

    def interpreter_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.resampler.parameters()
        yield from self.xattn.parameters()


# ---- B4-thin: thin-interface next-token (CTRL-2x2) ----------------------


class B4Thin(Build):
    """Thin-interface next-token build — (thin × next-token) cell of CTRL-2x2.

    Holds V2's *single-vector* interface fixed but trains it with **next-token
    CE** (B4's loss) instead of reconstruction. The controlled twin that
    disentangles the Phase-4.2 confound (PLAN §10.1 CTRL-2x2):

      * vs **V2** (thin × reconstruction): identical interface AND identical
        generation path (``lm.generate(ĥ_lm)``); only the *loss* differs →
        isolates the *learning-signal* effect.
      * vs **B4** (rich × next-token): identical loss; only the *interface*
        differs (single bridged vector vs K-latent distributed xattn) →
        isolates the *interface* effect.

    Mirrors V2's interface exactly: ``ñ_agent = LN(h_agent)``;
    ``ĥ_lm = W · ñ_agent`` with ``W`` shape ``(d_lm, d_a)`` (interface_proj
    space, same role as ``ACC.W``). Training conditions the frozen LM decoder
    on ``ĥ_lm`` and applies next-token CE over the describer sentence — i.e.
    the handle-B decode path of :meth:`MazeLM.decode_logits`, the same path the
    LM's autoencoding objective used.

    Pipeline (update):
        ñ_agent = LN(h_agent.detach())            # (B, d_a)  (C-thin boundary 1)
        ĥ_lm    = W · ñ_agent                      # (B, d_lm) single bridged vec
        logits  = lm.decode_logits(ĥ_lm, ids[:, :-1])   # (B, T, vocab)
        loss    = CE(logits, ids)                  # next-token, pad-ignored

    Generation: ``lm.generate(W · LN(h_agent))`` — byte-for-byte the decode
    path V2 uses, so the eval harness scores both thin builds uniformly.

    (C-thin): agent detached; the entire LM is frozen + eval — only ``W`` (and
    ``ln_agent`` if affine) learn the bridge.

    Args:
        lm:               a Phase-2 :class:`MazeLM`.
        d_agent:          IMPALA-CNN embedding width (256).
        layernorm_affine: LN affine on the agent side. Default False to match
                          the ACC (POST-HOC-6) so the thin interface has the
                          *same shape* as V2's.
        init:             ``"orthogonal"`` (default, matches ACC) or ``"xavier"``.
    """

    def __init__(
        self,
        lm: MazeLM,
        *,
        d_agent: int = 256,
        layernorm_affine: bool = False,
        init: str = "orthogonal",
    ):
        super().__init__()
        self.lm = lm
        d_model = lm.config.d_model

        # Freeze the LM core ((C-thin) LM 코어 보호; matches V2/B4).
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.eval()

        self.ln_agent = nn.LayerNorm(d_agent, elementwise_affine=layernorm_affine)
        # Agent→lm bridge, same shape/role as ACC.W (generation direction).
        self.W = nn.Parameter(torch.empty(d_model, d_agent))
        if init == "orthogonal":
            nn.init.orthogonal_(self.W)
        elif init == "xavier":
            nn.init.xavier_uniform_(self.W)
        else:
            raise ValueError(f"unknown W init: {init!r}")
        self.pad_id = lm.config.pad_id

    def train(self, mode: bool = True) -> "B4Thin":
        """Keep the frozen base LM in eval (deterministic decode path)."""
        super().train(mode)
        self.lm.eval()
        return self

    def bridge(self, h_agent: torch.Tensor) -> torch.Tensor:
        """ĥ_lm = W · LN(h_agent). (B, d_a) → (B, d_lm)."""
        return F.linear(self.ln_agent(h_agent), self.W)

    def update(
        self,
        h_agent: torch.Tensor,
        ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict:
        # Boundary 1 — agent insulated.
        h_lm = self.bridge(h_agent.detach())                  # (B, d_lm)
        # Next-token CE through the handle-B decode path (mirrors
        # MazeLM.autoencode_loss but conditions on the bridged ĥ_lm instead of
        # encode(ids)). logits[:, i] predicts ids[:, i].
        prefix = ids[:, :-1]
        logits = self.lm.decode_logits(h_lm, prefix)          # (B, T, vocab)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            ids.reshape(-1),
            ignore_index=self.pad_id,
        )
        return {"loss": loss}

    @torch.no_grad()
    def generate(self, h_agent: torch.Tensor, max_len: int = 16) -> torch.Tensor:
        """Greedy decode from the bridged vector — identical path to V2."""
        return self.lm.generate(self.bridge(h_agent), max_len=max_len)

    def interpreter_parameters(self) -> Iterator[nn.Parameter]:
        yield self.W
        if self.ln_agent.elementwise_affine:
            yield from self.ln_agent.parameters()


# ---- V2: ACC (the main model) -------------------------------------------


class V2ACC(Build):
    """ACC build — the project's main model (PLAN §4; P3-3-5 박제).

    The reconstruction-only interpreter. Pipeline:

        h_lm = lm.encode(ids)              # re-computed every step (P3-2-3)
        loss = acc.recon_loss(h_agent, h_lm)   # bidirectional MSE (C-thin)

    (C-thin) boundaries:
    - **Agent** insulated: ``ACC.recon_loss`` detaches ``h_agent`` internally
      (boundary 1).
    - **LM core** protected: every LM parameter is frozen *except*
      ``interface_proj`` (P3-2-A 박제). The reconstruction gradient reaches
      only ``interface_proj`` (via the re-computed ``h_lm``) + the ACC
      params (W + LayerNorms). The LM is kept in eval (``train`` override)
      so the encode path is deterministic.

    Because ``h_lm`` is re-computed from ``ids`` on every ``update`` (rather
    than cached), the gradient always flows through the *current*
    ``interface_proj`` — see PLAN §10.3 P3-2-3.

    Args:
        lm:  a Phase-2 :class:`MazeLM`.
        acc: an :class:`split_maze.acc.ACC` with matching ``d_lm`` /
             ``d_agent``.
    """

    def __init__(self, lm: MazeLM, acc: nn.Module):
        super().__init__()
        self.lm = lm
        self.acc = acc

        # POST-HOC-6 (2026-05-21): freeze the *entire* LM, interface_proj
        # included. Training interface_proj (P3-2-A) was the collapse path —
        # the ACC recon loss drove it to a constant (diagnose_v2). Now the LM
        # is a fixed, informative feature extractor and only ACC W learns the
        # bridge (= SPLIT-MNIST V2 본형). See PLAN §10.1 POST-HOC-6.
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.eval()

    def train(self, mode: bool = True) -> "V2ACC":
        """Keep the LM in eval (deterministic encode); interface_proj still
        trains because eval only affects dropout, not grad."""
        super().train(mode)
        self.lm.eval()
        return self

    def update(
        self,
        h_agent: torch.Tensor,
        ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict:
        # P3-2-3: re-compute h_lm so grad reaches the *current* interface_proj.
        h_lm = self.lm.encode(ids)
        # ACC.recon_loss handles boundary 1 (h_agent.detach()) internally.
        return self.acc.recon_loss(h_agent, h_lm)

    def interpreter_parameters(self) -> Iterator[nn.Parameter]:
        # POST-HOC-6: only the ACC learns (W; LN is non-affine so has no
        # params). The LM — interface_proj included — is frozen.
        yield from self.acc.parameters()


# ---- V2-rich: rich-interface reconstruction (CTRL-2x2) ------------------


class V2Rich(Build):
    """Rich-interface reconstruction build — (rich × reconstruction) cell of
    CTRL-2x2 (PLAN §10.1 CTRL-2x2).

    The 'reconstruction with a distributed interface' analog of V2. Where V2
    reconstructs the LM's single ``<SUM>`` summary vector (thin target), V2Rich
    reconstructs the LM's *per-position hidden sequence* — the rich,
    multi-vector target — from a distributed K-latent agent bridge, and is
    trained by **MSE on the frozen LM's hidden states** (reconstruction), NOT
    next-token CE.

    Pairs against **B4** (rich × next-token): the interface is rich (K latents,
    multi-position) in both; only the *loss* differs (MSE-on-hiddens vs
    next-token-CE).

    ARCHITECTURAL CAVEAT (박제): V2Rich uses a *separate* Perceiver
    reconstructor (learned position queries cross-attending to the agent
    latents), not B4's in-LM gated-xattn injection. So the rich pair is
    'spiritually matched' (both distributed/K-latent) rather than
    byte-identical. The **thin pair (V2 vs B4Thin) is the identical-interface
    gold standard**; the rich pair adds the 2×2 interaction term.

    Why no sentence tokens reach the student: if the student saw the true
    tokens, the bridge could be ignored (the LM would reconstruct its own
    hiddens trivially). Feeding only learned position queries forces the agent
    latents to carry the whole sentence — the reconstruction is *necessary*,
    mirroring V2's "agent must predict the sentence's LM-representation".

    Pipeline (update):
        Z       = frozen_lm hidden states reading the true sentence  (target, detached)
                  = lm._transform(tok_embed(append_sum(ids)))         (B, T+1, d_model)
        latents = resampler(h_agent.detach())                          (B, K, d_model)
        Ẑ       = recon_blocks(pos_queries[:T+1], latents)             (B, T+1, d_model)
        loss    = MSE(Ẑ, Z) over real-token positions [0, length)

    Generation (parallel, non-autoregressive): the LM's next-token convention
    means ``lm_head(Z[:, t])`` predicts token ``t+1``; so
    ``argmax(lm_head(Ẑ[:, t]))`` over positions reconstructs the sentence. We
    prepend ``<BOS>`` and pad-fill after the first ``<EOS>``.

    (C-thin): agent detached; the entire LM is frozen + eval — only the
    resampler + position queries + reconstructor blocks learn.

    Args:
        lm:               a Phase-2 :class:`MazeLM`.
        d_agent:          IMPALA-CNN embedding width (256).
        n_latents:        agent latents emitted by the resampler (16, = B4).
        n_kv:             resampler pseudo-token KV slots (≥2).
        n_heads:          attention heads (resampler + reconstructor).
        resampler_blocks: PerceiverBlock depth inside the resampler.
        recon_blocks:     PerceiverBlock depth inside the reconstructor.
    """

    def __init__(
        self,
        lm: MazeLM,
        *,
        d_agent: int = 256,
        n_latents: int = 16,
        n_kv: int = 8,
        n_heads: int = 4,
        resampler_blocks: int = 2,
        recon_blocks: int = 2,
    ):
        super().__init__()
        self.lm = lm
        d_model = lm.config.d_model

        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.eval()

        self.resampler = AgentResampler(
            d_agent=d_agent,
            d_model=d_model,
            n_latents=n_latents,
            n_kv=n_kv,
            n_heads=n_heads,
            n_blocks=resampler_blocks,
        )
        # Learned position queries — one per LM position (covers the appended
        # <SUM>, since _transform caps T at config.max_len ⇒ T+1 ≤ max_len).
        self.pos_queries = nn.Parameter(
            torch.randn(lm.config.max_len, d_model) * 0.02
        )
        self.recon_blocks = nn.ModuleList(
            [PerceiverBlock(d_model, n_heads) for _ in range(recon_blocks)]
        )
        self.pad_id = lm.config.pad_id
        self.bos_id = lm.config.bos_id
        self.eos_id = lm.config.eos_id

    def train(self, mode: bool = True) -> "V2Rich":
        super().train(mode)
        self.lm.eval()
        return self

    # ---- frozen-LM teacher hidden states ----

    @torch.no_grad()
    def _teacher_hidden(self, ids: torch.Tensor) -> torch.Tensor:
        """Frozen LM per-position hidden states (ln_f output) reading the true
        sentence with ``<SUM>`` appended. (B, T+1, d_model). Detached target."""
        ids_full = self.lm._append_sum(ids)
        emb = self.lm.tok_embed(ids_full)
        return self.lm._transform(emb)

    # ---- distributed reconstructor (student) ----

    def _student(self, h_agent: torch.Tensor, length: int) -> torch.Tensor:
        """Reconstruct (B, length, d_model) hiddens from the agent latents."""
        latents = self.resampler(h_agent)                      # (B, K, d_model)
        B = h_agent.shape[0]
        q = self.pos_queries[:length].unsqueeze(0).expand(B, -1, -1).contiguous()
        for blk in self.recon_blocks:
            q = blk(q, latents)
        return q                                               # (B, length, d_model)

    def update(
        self,
        h_agent: torch.Tensor,
        ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict:
        # Frozen target (no grad to LM).
        Z = self._teacher_hidden(ids)                          # (B, T1, d_model)
        T1 = Z.size(1)
        # Boundary 1 — agent insulated.
        Z_hat = self._student(h_agent.detach(), T1)            # (B, T1, d_model)

        # MSE over real-token positions [0, length): those are the hiddens that
        # (via lm_head) regenerate the content tokens. <SUM>/pad excluded.
        pos = torch.arange(T1, device=Z.device).unsqueeze(0)   # (1, T1)
        mask = (pos < lengths.to(Z.device).unsqueeze(1)).float()  # (B, T1)
        per_pos = ((Z_hat - Z) ** 2).mean(dim=-1)              # (B, T1)
        denom = mask.sum().clamp(min=1.0)
        loss = (per_pos * mask).sum() / denom
        return {"loss": loss}

    @torch.no_grad()
    def generate(self, h_agent: torch.Tensor, max_len: int = 16) -> torch.Tensor:
        """Parallel (non-autoregressive) decode via lm_head over reconstructed
        hiddens. Returns (B, L+1) ids beginning with <BOS>, pad-filled after
        the first <EOS>."""
        was_training = self.training
        self.eval()
        try:
            L = max(1, min(max_len, self.lm.config.max_len))
            Z_hat = self._student(h_agent, L)                  # (B, L, d_model)
            logits = self.lm.lm_head(Z_hat)                    # (B, L, vocab)
            pred = logits.argmax(dim=-1)                       # (B, L) = ids[1..L]
            B = h_agent.shape[0]
            bos = torch.full((B, 1), self.bos_id, dtype=torch.long,
                             device=h_agent.device)
            seq = torch.cat([bos, pred], dim=1)                # (B, L+1)
            # Pad-fill strictly after each row's first <EOS>.
            is_eos = (seq == self.eos_id)
            after_first_eos = (is_eos.cumsum(dim=1) - is_eos.long()) > 0
            seq = torch.where(after_first_eos,
                              torch.full_like(seq, self.pad_id), seq)
            return seq
        finally:
            if was_training:
                self.train()

    def interpreter_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.resampler.parameters()
        yield self.pos_queries
        yield from self.recon_blocks.parameters()


__all__ = [
    "Build",
    "B3Probe",
    "B4Adapter",
    "B4Thin",
    "V2ACC",
    "V2Rich",
    "N_ROW",
    "N_COL",
    "N_HEADING",
    "N_CHEESE",
    "IGNORE_INDEX",
]
