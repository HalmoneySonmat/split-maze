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

from .language import (
    CHEESE_DIR_VALUES,
    HEADING_VALUES,
    REGION_COLS,
    REGION_ROWS,
    parse,
)
from .lm import MazeTokenizer


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


__all__ = [
    "Build",
    "B3Probe",
    "N_ROW",
    "N_COL",
    "N_HEADING",
    "N_CHEESE",
    "IGNORE_INDEX",
]
