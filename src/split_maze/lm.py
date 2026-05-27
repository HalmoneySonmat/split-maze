"""Decoder-only transformer LM for the synthetic maze-language (Phase 2).

Implements PLAN §3.4 (handle B = autoencoding-consistency) with the
Phase-2-time settings frozen on 2026-05-19 (PLAN P2-1..P2-4):

- ``<SUM>`` token is appended at the *end* of every encode-time sequence.
  Its final-position hidden state is the summary vector h_lm (handle B).
- Decoder transformer: 3 layers, d_model=256, n_head=4, FFN=1024.
- Auto-encoding loss weight λ_ae = 1.0 (set by the caller).
- Neutral corpus N = 50_000 (training-loop concern, not this module).

The decode forward conditions on h_lm by injecting it as the *position-0
hidden* (replacing what would otherwise be a token embedding). This makes
encode/decode naturally symmetric: encode produces a single vector from the
right side of the causal mask, decode produces a sequence from that vector
on the left side.

Grad boundary (PLAN §4.3, "(C-thin)"): in Phase 3 the ACC reconstruction
grad is allowed to reach the **interface** parameters only — here that is
``interface_proj``. The rest of the LM (``tok_embed``, ``pos_embed``,
``blocks``, ``ln_f``, weight-tied ``lm_head``) is the **core**, trained in
Phase 2 by next-token + auto-encoding losses and *stop-grad'd* by ACC in
Phase 3. Convenience accessors ``core_parameters()`` /
``interface_parameters()`` expose that split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .language import BOS, EOS, PAD, SUM, vocab


# ---- Tokenizer ----------------------------------------------------------

class MazeTokenizer:
    """Static ``token <-> id`` mapping built from ``language.vocab()``.

    The vocabulary is the same one defined in :mod:`split_maze.language` (the
    sorted list of all tokens that can appear in a maze-language sentence,
    plus the four special tokens ``<BOS> <EOS> <PAD> <SUM>``). The mapping
    is deterministic across processes because ``vocab()`` returns a sorted
    list.
    """

    def __init__(self) -> None:
        self.tokens: list[str] = vocab()
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        self.pad_id: int = self.token_to_id[PAD]
        self.bos_id: int = self.token_to_id[BOS]
        self.eos_id: int = self.token_to_id[EOS]
        self.sum_id: int = self.token_to_id[SUM]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id[t] for t in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.tokens[int(i)] for i in ids]

    def collate(self, sequences: list[list[int]],
                device: Optional[torch.device] = None) -> torch.Tensor:
        """Pad a list of variable-length id lists into a (B, T) tensor.

        Padding uses ``pad_id``. The loss functions in :class:`MazeLM` honour
        this via ``ignore_index=pad_id``.
        """
        if not sequences:
            raise ValueError("collate requires at least one sequence")
        max_t = max(len(s) for s in sequences)
        out = torch.full((len(sequences), max_t), self.pad_id, dtype=torch.long)
        for i, s in enumerate(sequences):
            if len(s) > 0:
                out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        if device is not None:
            out = out.to(device)
        return out


# ---- Config -------------------------------------------------------------

@dataclass
class LMConfig:
    """Phase-2 LM hyperparameters (PLAN P2-2 frozen on 2026-05-19).

    The defaults match the frozen baseline; ``vocab_size`` and the special-
    token ids must be filled in from a :class:`MazeTokenizer`. Use
    :meth:`from_tokenizer` for the canonical construction.
    """

    vocab_size: int
    d_model: int = 256
    n_head: int = 4
    n_layer: int = 3
    d_ff: int = 1024
    max_len: int = 32
    dropout: float = 0.1
    pad_id: int = 0
    bos_id: int = 0
    eos_id: int = 0
    sum_id: int = 0

    @classmethod
    def from_tokenizer(cls, tok: MazeTokenizer, **overrides) -> "LMConfig":
        cfg = cls(
            vocab_size=tok.vocab_size,
            pad_id=tok.pad_id,
            bos_id=tok.bos_id,
            eos_id=tok.eos_id,
            sum_id=tok.sum_id,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


# ---- Building blocks ---------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask (lower-triangular).

    Uses ``F.scaled_dot_product_attention(is_causal=True)`` so the mask is
    applied without materializing a (T, T) tensor — important for clean
    behaviour at small T as well as cheap at larger T.
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_head={n_head}")
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.d_head)
        # → (3, B, n_head, T, d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )  # (B, n_head, T, d_head)
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → attn → +; LN → MLP → +."""

    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---- MazeLM ------------------------------------------------------------

class MazeLM(nn.Module):
    """Decoder transformer with ``<SUM>``-handle-B autoencoding.

    Conventions for input ``ids``:
    - User-facing methods (``forward``, ``encode``, ``next_token_loss``,
      ``autoencode_loss``) expect *plain* corpus sequences of the form
      ``[<BOS>, t1, ..., tn, <EOS>]`` (with optional ``<PAD>`` to the right
      of ``<EOS>``). The ``<SUM>`` token is **always appended internally**
      and must not be supplied by the caller.
    - ``decode_logits`` is the lower-level teacher-forced decode and accepts
      an already-prepared prefix.
    """

    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config

        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        # Learnable absolute position embedding (1, max_len, d_model).
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_len, config.d_model))
        self.embed_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_head,
                             config.d_ff, config.dropout)
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)

        # Interface projection — the explicit "LM interface" PLAN §4.3 names.
        # In Phase 2 it trains via the autoencoding loss along with the core;
        # in Phase 3 it is the only LM-side parameter that ACC's reconstruction
        # gradient is allowed to touch (the rest is stop-grad).
        self.interface_proj = nn.Linear(config.d_model, config.d_model)

        # LM head — bias-free linear from hidden → vocab. Weight-tied to the
        # token embedding for parameter efficiency (standard small-LM trick).
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Init everything, *then* tie the lm_head/tok_embed weights so we
        # don't double-init the shared tensor.
        self.apply(self._init_weights)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        self.lm_head.weight = self.tok_embed.weight  # weight tying

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---- low-level transformer pipeline -----------------------------

    def _transform(self, embed: torch.Tensor) -> torch.Tensor:
        """Apply position embedding + blocks + final LN to a (B, T, d_model)
        tensor of *raw* embeddings (token-derived OR injected h_lm).

        Raises if T > max_len.
        """
        T = embed.size(1)
        if T > self.config.max_len:
            raise ValueError(
                f"sequence length {T} exceeds LMConfig.max_len {self.config.max_len}"
            )
        x = embed + self.pos_embed[:, :T]
        x = self.embed_dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def _append_sum(self, ids: torch.Tensor) -> torch.Tensor:
        """Append a column of ``sum_id`` to ``ids`` (B, T) → (B, T+1)."""
        sum_col = torch.full(
            (ids.size(0), 1), self.config.sum_id,
            dtype=ids.dtype, device=ids.device,
        )
        return torch.cat([ids, sum_col], dim=1)

    # ---- forward / encode -------------------------------------------

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over ``[<BOS>, ..., <EOS>] + <SUM>``.

        Args:
            ids: (B, T) integer token ids without ``<SUM>``; the function
                 appends ``<SUM>`` internally.
        Returns:
            logits: (B, T+1, vocab) raw LM logits at every position.
            h_lm:   (B, d_model)    interface_proj(hidden at the appended
                                    ``<SUM>`` position) — the handle-B summary.
        """
        ids_full = self._append_sum(ids)
        emb = self.tok_embed(ids_full)
        x = self._transform(emb)
        logits = self.lm_head(x)
        h_lm = self.interface_proj(x[:, -1])
        return logits, h_lm

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """Return only the summary vector h_lm (B, d_model)."""
        _, h_lm = self.forward(ids)
        return h_lm

    def summarize_vector(self, h_in: torch.Tensor) -> torch.Tensor:
        """The LM's immediate read of a *bridged vector* injected at position 0.

        Phase-6 R2 feedback (PREREG §1, lm→agent). ``h_in`` is a vector in
        LM space — e.g. ``ĥ_lm = W·LN(h_agent)`` produced by the ACC — fed at
        position 0 exactly as :meth:`generate`/:meth:`decode_logits` inject
        ``h_lm``. We run the transformer over that length-1 sequence and take
        ``interface_proj`` of the final hidden — the same handle-B summary
        formula as :meth:`forward`, but conditioned on the bridged vector
        rather than token embeddings. This is the LM's *interpretation* of what
        it read (differentiable), as opposed to a bare bridge round-trip.

        Args:
            h_in: (B, d_model).
        Returns:
            (B, d_model) summary vector h_lm.
        """
        x = self._transform(h_in.unsqueeze(1))   # (B, 1, d_model)
        return self.interface_proj(x[:, -1])

    # ---- decode -----------------------------------------------------

    def decode_logits(self, h_lm: torch.Tensor,
                      prefix_ids: torch.Tensor) -> torch.Tensor:
        """Teacher-forced decode.

        The decoder sequence is ``[h_lm, embed(prefix_ids)]``: h_lm replaces
        what would be a token embedding at position 0, the prefix tokens occupy
        positions 1..T_p. The output ``logits[:, i]`` is the model's prediction
        for the token that should appear at position i+1 of the *original*
        sequence — i.e. position 0 logits predict ``<BOS>``, position 1
        predicts the first content token, etc.

        Args:
            h_lm: (B, d_model).
            prefix_ids: (B, T_p) teacher-forcing prefix tokens.
        Returns:
            logits: (B, 1 + T_p, vocab).
        """
        tok_emb = self.tok_embed(prefix_ids)
        h_lm_emb = h_lm.unsqueeze(1)  # (B, 1, d_model)
        emb = torch.cat([h_lm_emb, tok_emb], dim=1)
        x = self._transform(emb)
        return self.lm_head(x)

    # ---- losses -----------------------------------------------------

    def next_token_loss(self, ids: torch.Tensor) -> torch.Tensor:
        """Standard next-token cross-entropy loss.

        Args:
            ids: (B, T) corpus sequences ``[<BOS>, ..., <EOS>]`` (no ``<SUM>``).
                 ``<SUM>`` is appended internally; the position after ``<EOS>``
                 learns to predict ``<SUM>``.
        Returns:
            scalar loss. ``pad_id`` positions are ignored.
        """
        logits, _ = self.forward(ids)  # (B, T+1, vocab)
        ids_full = self._append_sum(ids)
        # Predict ids_full[:, t+1] from logits[:, t]. Skip the last logits
        # column (no following token to predict).
        pred = logits[:, :-1].reshape(-1, self.config.vocab_size)
        target = ids_full[:, 1:].reshape(-1)
        return F.cross_entropy(pred, target, ignore_index=self.config.pad_id)

    def autoencode_loss(self, ids: torch.Tensor) -> torch.Tensor:
        """``decode(encode(S)) ≈ S`` reconstruction loss — handle B (PLAN §3.4).

        Args:
            ids: (B, T) sequences ``[<BOS>, t1, ..., tn, <EOS>]`` (no ``<SUM>``).
        Returns:
            scalar cross-entropy loss over all (non-pad) token positions.
        """
        h_lm = self.encode(ids)
        # Decoder teacher-forcing prefix = sequence without its final token.
        # The full reconstruction target is `ids` itself: position 0 of the
        # decoder output predicts <BOS>, position T-1 predicts <EOS>.
        prefix = ids[:, :-1]
        logits = self.decode_logits(h_lm, prefix)  # (B, T, vocab)
        target = ids                              # (B, T)
        return F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            target.reshape(-1),
            ignore_index=self.config.pad_id,
        )

    def combined_loss(self, ids: torch.Tensor,
                      lambda_ae: float = 1.0) -> dict[str, torch.Tensor]:
        """One-call training loss: ``L = L_nexttoken + λ_ae · L_ae``.

        Defaults to λ_ae = 1.0 (PLAN P2-3, frozen 2026-05-19). Returns a dict
        of the components plus the scalar total, which is useful for logging.
        """
        l_next = self.next_token_loss(ids)
        l_ae = self.autoencode_loss(ids)
        total = l_next + lambda_ae * l_ae
        return {"loss": total, "next_token": l_next, "autoencode": l_ae}

    # ---- generation -------------------------------------------------

    @torch.no_grad()
    def generate(self, h_lm: torch.Tensor, max_len: int = 16,
                 eos_id: Optional[int] = None) -> torch.Tensor:
        """Greedy autoregressive decode starting from h_lm.

        The decoder is fed ``[h_lm]`` first; the argmax of position 0 logits
        is the first emitted token. Generation continues until every row has
        emitted ``<EOS>`` or ``max_len`` tokens have been produced; tokens
        after a row's first ``<EOS>`` are filled with ``pad_id``.

        Args:
            h_lm: (B, d_model).
            max_len: maximum number of tokens to emit (excluding the
                     position-0 conditioning slot). Capped at ``max_len - 1``
                     of :class:`LMConfig` to stay inside the position embedding.
        Returns:
            (B, T_gen) token ids with 1 ≤ T_gen ≤ max_len.
        """
        if eos_id is None:
            eos_id = self.config.eos_id
        pad_id = self.config.pad_id
        B = h_lm.size(0)
        device = h_lm.device

        # Position 0 is occupied by h_lm; subsequent positions hold generated
        # tokens. Total sequence length is bounded by config.max_len.
        max_len = max(1, min(max_len, self.config.max_len - 1))

        was_training = self.training
        self.eval()
        try:
            finished = torch.zeros(B, dtype=torch.bool, device=device)
            generated: list[torch.Tensor] = []
            for step in range(max_len):
                if generated:
                    prefix = torch.stack(generated, dim=1)
                    tok_emb = self.tok_embed(prefix)
                    emb = torch.cat([h_lm.unsqueeze(1), tok_emb], dim=1)
                else:
                    emb = h_lm.unsqueeze(1)
                x = self._transform(emb)
                next_logits = self.lm_head(x[:, -1])
                next_tok = next_logits.argmax(dim=-1)
                # Finished rows continue emitting <PAD>.
                next_tok = torch.where(
                    finished,
                    torch.full_like(next_tok, pad_id),
                    next_tok,
                )
                generated.append(next_tok)
                finished = finished | (next_tok == eos_id)
                if bool(finished.all()):
                    break
            return torch.stack(generated, dim=1)
        finally:
            if was_training:
                self.train()

    # ---- parameter-group helpers (Phase 3 stop-grad split) ----------

    def interface_parameters(self) -> Iterator[nn.Parameter]:
        """Parameters that ACC's reconstruction grad is allowed to touch
        (PLAN §4.3 "(C-thin)"). In Phase 3 only these are unfrozen on the
        LM side; the core is stop-grad'd."""
        yield from self.interface_proj.parameters()

    def core_parameters(self) -> Iterator[nn.Parameter]:
        """All other LM parameters — the core that ACC must NOT modify."""
        seen = set(id(p) for p in self.interface_parameters())
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))  # avoid yielding the same shared tensor twice
            yield p
