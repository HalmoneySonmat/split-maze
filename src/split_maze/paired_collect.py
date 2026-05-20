"""paired_collect — (h_agent, ids) pair extraction + FIFO replay buffer (Phase 3.2).

Implements the data pipeline for ACC/B3/B4/V2 co-training, with the Phase-3.2
decisions frozen on 2026-05-21 (PLAN §10.3 P3-2-1..P3-2-4):

- **P3-2-1**: This module is *separate* from ``train.py`` — the Phase-1
  PPO loop is left untouched, the Phase-3 entry point ``train_phase3.py``
  (next sub-step) wires the rollout buffer into this module's
  ``PairedCollector``.
- **P3-2-2**: FIFO replay buffer of capacity 256k pairs, with **uniform
  random sampling** of mini-batches (batch=128).
- **P3-2-3**: Only token *ids* are stored — ``h_lm`` is re-computed by the
  caller via ``MazeLM.encode(ids)`` on every ACC update, so the
  (C-thin) grad path always reaches the *current* ``interface_proj``.
- **P3-2-4**: Sampling cadence is the caller's responsibility (the
  current default is K=32 mini-batches per RL update — that's
  ``train_phase3.py``'s loop, not this module's).

This module has two classes:

- :class:`PairBuffer` — pre-allocated ring buffer, ``add`` / ``sample``.
- :class:`PairedCollector` — stateless wrapper around tokenizer +
  ``describer_oracle``; turns a rollout slice into pairs and pushes them
  into a :class:`PairBuffer`.

There is intentionally **no** dependency on procgen / gym3 / the agent —
those live in the caller (``train_phase3.py``). Tests can therefore use
``MockMazeEnv``-style synthetic data and assert pure-Python behaviour.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Sequence

import torch

from .language import MazeState, Slots, describer_oracle, render
from .lm import MazeTokenizer


# ---- Config -------------------------------------------------------------


@dataclass
class PairBufferConfig:
    """Phase-3.2 P3-2-2 / P3-4 / P3-3 박제값 default.

    Attributes:
        capacity:      Max pairs held simultaneously. PLAN §10.2 P3-4 → 256k.
        batch_size:    Sampling batch. PLAN §10.2 P3-3-A → 128.
        d_agent:       IMPALA-CNN final dense width. PROCGEN_ENV §7 → 256.
        max_token_len: Maximum rendered-sentence length. The maze grammar
                       (LANGUAGE_SPEC v0.3, 4 slots × 2 toks + BOS/EOS) is
                       ≤ 12 tokens; we pad to 16 for safety.
    """

    capacity: int = 256_000
    batch_size: int = 128
    d_agent: int = 256
    max_token_len: int = 16


# ---- PairBuffer ---------------------------------------------------------


class PairBuffer:
    """FIFO ring buffer of (h_agent, ids, length) tuples.

    Storage layout — three pre-allocated CPU tensors of length ``capacity``:
        h_agent_buf: (capacity, d_agent)        float
        ids_buf:     (capacity, max_token_len)  long, pad_id filled
        lengths_buf: (capacity,)                long

    On ``add``, batches are appended at ``_write_ptr`` with wrap-around.
    On ``sample(n)``, ``n`` uniform-random indices are drawn from
    ``[0, _size)``; the returned dict's tensors are *clones* of the buffer
    slices so the caller gets a stable snapshot even as the buffer rolls.

    Args:
        cfg:    :class:`PairBufferConfig`.
        pad_id: token id used to fill the unused tail of each ``ids`` row.
                Default matches ``MazeTokenizer.pad_id``.
    """

    def __init__(self, cfg: Optional[PairBufferConfig] = None, *, pad_id: int = 0):
        self.cfg = cfg or PairBufferConfig()
        self.pad_id = pad_id

        self.h_agent_buf = torch.empty(self.cfg.capacity, self.cfg.d_agent)
        self.ids_buf = torch.full(
            (self.cfg.capacity, self.cfg.max_token_len),
            self.pad_id,
            dtype=torch.long,
        )
        self.lengths_buf = torch.zeros(self.cfg.capacity, dtype=torch.long)

        self._write_ptr: int = 0
        self._size: int = 0

    # ---- introspection ----

    def __len__(self) -> int:
        return self._size

    def is_ready(self, n: Optional[int] = None) -> bool:
        """True when at least ``n`` (default ``batch_size``) pairs are stored."""
        return self._size >= (n if n is not None else self.cfg.batch_size)

    @property
    def capacity(self) -> int:
        return self.cfg.capacity

    # ---- write ----

    def add(
        self,
        h_agent: torch.Tensor,
        ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> int:
        """Append a batch of pairs to the ring buffer.

        Args:
            h_agent: (B, d_agent). Float tensor; will be cast to the buffer
                     dtype. Detached automatically (this module never
                     propagates agent grad).
            ids:     (B, max_token_len). Long tensor; rows padded with
                     ``pad_id`` to the right.
            lengths: (B,). Long tensor — number of valid tokens per row.

        Returns:
            Number of pairs effectively added (= ``B`` capped at capacity).
        """
        if h_agent.dim() != 2 or h_agent.shape[1] != self.cfg.d_agent:
            raise ValueError(
                f"h_agent must be (B, {self.cfg.d_agent}); got {tuple(h_agent.shape)}"
            )
        if ids.dim() != 2 or ids.shape[1] != self.cfg.max_token_len:
            raise ValueError(
                f"ids must be (B, {self.cfg.max_token_len}); got {tuple(ids.shape)}"
            )
        if lengths.dim() != 1 or lengths.shape[0] != h_agent.shape[0]:
            raise ValueError(
                f"lengths must be (B={h_agent.shape[0]},); got {tuple(lengths.shape)}"
            )
        B = h_agent.shape[0]
        if B == 0:
            return 0

        # Detach + move to buffer device on the fly (CPU by default).
        h_agent = h_agent.detach().to(self.h_agent_buf.dtype).cpu()
        ids = ids.detach().to(self.ids_buf.dtype).cpu()
        lengths = lengths.detach().to(self.lengths_buf.dtype).cpu()

        cap = self.cfg.capacity

        # If the incoming batch is bigger than capacity, keep only the last
        # `capacity` entries (most recent) — pathological case but defined.
        if B >= cap:
            self.h_agent_buf.copy_(h_agent[-cap:])
            self.ids_buf.copy_(ids[-cap:])
            self.lengths_buf.copy_(lengths[-cap:])
            self._write_ptr = 0
            self._size = cap
            return cap

        write = self._write_ptr
        end = write + B
        if end <= cap:
            self.h_agent_buf[write:end] = h_agent
            self.ids_buf[write:end] = ids
            self.lengths_buf[write:end] = lengths
        else:
            first = cap - write
            self.h_agent_buf[write:] = h_agent[:first]
            self.ids_buf[write:] = ids[:first]
            self.lengths_buf[write:] = lengths[:first]
            rest = B - first
            self.h_agent_buf[:rest] = h_agent[first:]
            self.ids_buf[:rest] = ids[first:]
            self.lengths_buf[:rest] = lengths[first:]

        self._write_ptr = (write + B) % cap
        self._size = min(self._size + B, cap)
        return B

    # ---- read ----

    def sample(
        self,
        n: Optional[int] = None,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> dict:
        """Uniform random batch of ``n`` pairs (default ``batch_size``).

        Returns:
            Dict with keys
                "h_agent": (n, d_agent),
                "ids":     (n, max_token_len),
                "lengths": (n,),
            all *clones* of the buffer rows (stable across subsequent adds).
        """
        n = n if n is not None else self.cfg.batch_size
        if n <= 0:
            raise ValueError(f"n must be positive; got {n}")
        if self._size < n:
            raise ValueError(
                f"buffer has {self._size} entries, requested {n}"
            )
        idx = torch.randint(0, self._size, (n,), generator=generator)
        return {
            "h_agent": self.h_agent_buf[idx].clone(),
            "ids": self.ids_buf[idx].clone(),
            "lengths": self.lengths_buf[idx].clone(),
        }


# ---- PairedCollector ----------------------------------------------------


# Type alias for the oracle — callable from a MazeState to optional Slots.
OracleFn = Callable[[MazeState], Optional[Slots]]


class PairedCollector:
    """Stride-based pair extractor from a rollout slice.

    Stateless: configured at construction with a tokenizer, oracle, and
    stride. ``extract_into`` walks a ``(T, N)`` grid of MazeStates and
    pushes one pair per ``(stride·k, n)`` step into the buffer (skipping
    cases where the oracle returns ``None`` or sprite detection failed).

    Surface-form randomisation (``render`` 's slot-order permutation,
    marker synonyms, optional connectives) is driven by an optional
    ``random.Random`` instance the caller passes to ``extract_into`` —
    keep it seeded for reproducibility.

    Args:
        tokenizer:     :class:`MazeTokenizer` (Phase 2).
        oracle:        callable ``state -> Optional[Slots]``. Defaults to
                       :func:`split_maze.language.describer_oracle`.
        stride:        sample every ``stride``-th timestep. P3-4 박제 → 4.
        max_token_len: row width of the ids buffer. Must match
                       ``PairBufferConfig.max_token_len``.
    """

    def __init__(
        self,
        tokenizer: MazeTokenizer,
        *,
        oracle: OracleFn = describer_oracle,
        stride: int = 4,
        max_token_len: int = 16,
    ):
        if stride < 1:
            raise ValueError("stride must be ≥ 1")
        if max_token_len < 1:
            raise ValueError("max_token_len must be ≥ 1")
        self.tokenizer = tokenizer
        self.oracle = oracle
        self.stride = stride
        self.max_token_len = max_token_len

    # ---- single-pair conversion ----

    def _make_pair(
        self,
        state: Optional[MazeState],
        rng: Optional[random.Random],
    ) -> Optional[tuple[torch.Tensor, int]]:
        """state → (padded ids (max_token_len,), length) or None.

        Returns ``None`` if:
        - ``state is None`` (sprite detection failed upstream),
        - ``oracle(state) is None`` (agent on cheese — undefined CHEESE_DIR,
          LANGUAGE_SPEC §5),
        - the rendered token sequence exceeds ``max_token_len``.
        """
        if state is None:
            return None
        slots = self.oracle(state)
        if slots is None:
            return None
        tokens = render(slots, rng=rng, include_bos_eos=True)
        ids_list = self.tokenizer.encode(tokens)
        length = len(ids_list)
        if length == 0 or length > self.max_token_len:
            return None
        padded = torch.full(
            (self.max_token_len,), self.tokenizer.pad_id, dtype=torch.long
        )
        padded[:length] = torch.tensor(ids_list, dtype=torch.long)
        return padded, length

    # ---- batch extraction ----

    def extract_into(
        self,
        buffer: PairBuffer,
        h_agent: torch.Tensor,
        maze_states: Sequence[Sequence[Optional[MazeState]]],
        *,
        rng: Optional[random.Random] = None,
    ) -> int:
        """Push (h_agent, ids) pairs into ``buffer`` for every valid
        ``(t, n)`` where ``t`` is a multiple of ``stride``.

        Args:
            buffer:      destination :class:`PairBuffer`.
            h_agent:     (T, N, d_agent). Float tensor; the rollout's
                         per-step IMPALA-CNN embeddings.
            maze_states: a ``T × N`` nested sequence of
                         ``Optional[MazeState]`` (``None`` for sprite
                         detection failures or pre-window steps).
            rng:         ``random.Random`` for surface-form variation.
                         If omitted, ``render`` uses its own default.

        Returns:
            The number of pairs added to the buffer (may be less than
            ``ceil(T/stride) * N`` due to None states / oracle skips /
            length overflow).
        """
        if h_agent.dim() != 3:
            raise ValueError(
                f"h_agent must be (T, N, d_agent); got {tuple(h_agent.shape)}"
            )
        T, N, d_a = h_agent.shape
        if d_a != buffer.cfg.d_agent:
            raise ValueError(
                f"d_agent mismatch: h_agent has {d_a}, buffer has "
                f"{buffer.cfg.d_agent}"
            )
        if buffer.cfg.max_token_len != self.max_token_len:
            raise ValueError(
                "buffer.max_token_len and collector.max_token_len must match"
                f" (got {buffer.cfg.max_token_len} vs {self.max_token_len})"
            )
        if len(maze_states) != T:
            raise ValueError(
                f"maze_states outer length must be T={T}; got {len(maze_states)}"
            )

        h_rows: List[torch.Tensor] = []
        id_rows: List[torch.Tensor] = []
        length_rows: List[int] = []

        for t in range(0, T, self.stride):
            row = maze_states[t]
            if len(row) != N:
                raise ValueError(
                    f"maze_states[{t}] must have N={N} entries; got {len(row)}"
                )
            for n in range(N):
                pair = self._make_pair(row[n], rng)
                if pair is None:
                    continue
                padded, length = pair
                h_rows.append(h_agent[t, n])
                id_rows.append(padded)
                length_rows.append(length)

        if not h_rows:
            return 0

        h_batch = torch.stack(h_rows, dim=0)
        id_batch = torch.stack(id_rows, dim=0)
        length_batch = torch.tensor(length_rows, dtype=torch.long)
        return buffer.add(h_batch, id_batch, length_batch)


# ---- module exports -----------------------------------------------------

__all__ = [
    "PairBufferConfig",
    "PairBuffer",
    "PairedCollector",
    "OracleFn",
]
