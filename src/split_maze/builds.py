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

from .adapter import AgentResampler, GatedCrossAttentionBlock
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

    def interpreter_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.resampler.parameters()
        yield from self.xattn.parameters()


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

        # Freeze the LM core; leave only interface_proj trainable (P3-2-A).
        for name, p in self.lm.named_parameters():
            p.requires_grad_(name.startswith("interface_proj"))
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
        yield from self.acc.parameters()
        yield from self.lm.interface_parameters()


__all__ = [
    "Build",
    "B3Probe",
    "B4Adapter",
    "V2ACC",
    "N_ROW",
    "N_COL",
    "N_HEADING",
    "N_CHEESE",
    "IGNORE_INDEX",
]
