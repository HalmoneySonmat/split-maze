"""Unit tests for split_maze.builds (Phase 3.3.1 — Build base + B3Probe).

Organised by concern:

1. Build ABC — cannot instantiate; subclasses must implement.
2. B3Probe construction — head dims match slot class counts.
3. B3Probe.forward — per-slot logits shapes.
4. B3Probe._targets_from_ids — round-trip from a known Slots → render →
   tokenizer → ids → recovered indices.
5. B3Probe.update — loss dict, scalar/finite, (C-thin) boundary 1
   (h_agent grad None), CE decomposition.
6. B3Probe smoke — AdamW reduces probe loss on a fixed batch.
7. interpreter_parameters — yields all probe params.
8. _ce all-ignore safety.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from split_maze.builds import (  # noqa: E402
    IGNORE_INDEX,
    N_CHEESE,
    N_COL,
    N_HEADING,
    N_ROW,
    B3Probe,
    Build,
)
from split_maze.language import (  # noqa: E402
    CHEESE_DIR_VALUES,
    HEADING_VALUES,
    REGION_COLS,
    REGION_ROWS,
    Slots,
    render,
)
from split_maze.lm import MazeTokenizer  # noqa: E402


# ---- helpers ------------------------------------------------------------


def _ids_from_slots(slots: Slots, tokenizer: MazeTokenizer, rng):
    """Slots → render → encode → (ids_tensor (1, L), length)."""
    tokens = render(slots, rng=rng, include_bos_eos=True)
    ids = tokenizer.encode(tokens)
    return torch.tensor([ids], dtype=torch.long), len(ids)


# ---- 1. Build ABC -------------------------------------------------------


class TestBuildABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Build()

    def test_subclass_without_methods_fails(self):
        class Incomplete(Build):
            pass

        with pytest.raises(TypeError):
            Incomplete()


# ---- 2. B3Probe construction -------------------------------------------


class TestB3ProbeConstruction:
    def test_head_dims_match_slot_counts(self):
        tok = MazeTokenizer()
        probe = B3Probe(tok, d_agent=256, hidden=256)
        assert probe.head_row.out_features == N_ROW == 3
        assert probe.head_col.out_features == N_COL == 3
        assert probe.head_heading.out_features == N_HEADING == 9
        assert probe.head_cheese.out_features == N_CHEESE == 8

    def test_is_a_build(self):
        tok = MazeTokenizer()
        assert isinstance(B3Probe(tok), Build)

    def test_trunk_dims(self):
        tok = MazeTokenizer()
        probe = B3Probe(tok, d_agent=128, hidden=64)
        first = probe.trunk[0]
        assert isinstance(first, nn.Linear)
        assert first.in_features == 128
        assert first.out_features == 64


# ---- 3. B3Probe.forward -------------------------------------------------


class TestB3ProbeForward:
    def test_logits_shapes(self):
        tok = MazeTokenizer()
        probe = B3Probe(tok, d_agent=256, hidden=256)
        h = torch.randn(5, 256)
        out = probe(h)
        assert out["row"].shape == (5, N_ROW)
        assert out["col"].shape == (5, N_COL)
        assert out["heading"].shape == (5, N_HEADING)
        assert out["cheese"].shape == (5, N_CHEESE)


# ---- 4. target round-trip ----------------------------------------------


class TestB3ProbeTargets:
    def test_round_trip_known_slots(self):
        tok = MazeTokenizer()
        probe = B3Probe(tok)
        # top/right/up/down → indices 0 / 2 / 0 / 4
        slots = Slots(agent_row="top", agent_col="right",
                      heading="up", cheese_dir="down")
        ids, length = _ids_from_slots(slots, tok, random.Random(0))
        lengths = torch.tensor([length])
        row_t, col_t, head_t, chee_t = probe._targets_from_ids(ids, lengths)
        assert row_t[0].item() == REGION_ROWS.index("top") == 0
        assert col_t[0].item() == REGION_COLS.index("right") == 2
        assert head_t[0].item() == HEADING_VALUES.index("up") == 0
        assert chee_t[0].item() == CHEESE_DIR_VALUES.index("down") == 4

    def test_round_trip_various_surface_forms(self):
        """Different surface forms of the same Slots recover identical
        slot indices."""
        tok = MazeTokenizer()
        probe = B3Probe(tok)
        slots = Slots(agent_row="bottom", agent_col="center",
                      heading="still", cheese_dir="left")
        for seed in range(8):
            ids, length = _ids_from_slots(slots, tok, random.Random(seed))
            r, c, h, ch = probe._targets_from_ids(ids, torch.tensor([length]))
            assert r[0].item() == REGION_ROWS.index("bottom")
            assert c[0].item() == REGION_COLS.index("center")
            assert h[0].item() == HEADING_VALUES.index("still")
            assert ch[0].item() == CHEESE_DIR_VALUES.index("left")


# ---- 5. update + (C-thin) ----------------------------------------------


class TestB3ProbeUpdate:
    def _batch(self, tok, B=6):
        rng = random.Random(0)
        ids_list, lens = [], []
        for _ in range(B):
            slots = Slots(
                agent_row=rng.choice(REGION_ROWS),
                agent_col=rng.choice(REGION_COLS),
                heading=rng.choice(HEADING_VALUES),
                cheese_dir=rng.choice(CHEESE_DIR_VALUES),
            )
            tokens = render(slots, rng=rng, include_bos_eos=True)
            ids = tok.encode(tokens)
            ids_list.append(ids)
            lens.append(len(ids))
        T = max(lens)
        padded = torch.full((B, T), tok.pad_id, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        return padded, torch.tensor(lens)

    def test_update_returns_loss_dict(self):
        tok = MazeTokenizer()
        probe = B3Probe(tok)
        ids, lengths = self._batch(tok)
        h = torch.randn(ids.shape[0], 256)
        out = probe.update(h, ids, lengths)
        assert set(out) >= {"loss", "loss_row", "loss_col",
                            "loss_heading", "loss_cheese"}

    def test_loss_scalar_finite(self):
        tok = MazeTokenizer()
        probe = B3Probe(tok)
        ids, lengths = self._batch(tok)
        out = probe.update(torch.randn(ids.shape[0], 256), ids, lengths)
        assert out["loss"].ndim == 0
        assert torch.isfinite(out["loss"])

    def test_loss_is_mean_of_four(self):
        tok = MazeTokenizer()
        probe = B3Probe(tok)
        ids, lengths = self._batch(tok)
        out = probe.update(torch.randn(ids.shape[0], 256), ids, lengths)
        expected = (out["loss_row"] + out["loss_col"]
                    + out["loss_heading"] + out["loss_cheese"]) / 4.0
        assert torch.isclose(out["loss"], expected, atol=1e-6)

    def test_h_agent_receives_no_grad(self):
        """(C-thin) boundary 1: probe must not propagate grad to agent."""
        tok = MazeTokenizer()
        probe = B3Probe(tok)
        ids, lengths = self._batch(tok)
        h = torch.randn(ids.shape[0], 256, requires_grad=True)
        out = probe.update(h, ids, lengths)
        out["loss"].backward()
        assert h.grad is None

    def test_probe_params_receive_grad(self):
        tok = MazeTokenizer()
        probe = B3Probe(tok)
        ids, lengths = self._batch(tok)
        h = torch.randn(ids.shape[0], 256)
        out = probe.update(h, ids, lengths)
        out["loss"].backward()
        assert probe.head_row.weight.grad is not None
        assert probe.head_row.weight.grad.abs().sum().item() > 0.0


# ---- 6. smoke -----------------------------------------------------------


class TestB3ProbeSmoke:
    def test_optim_reduces_loss(self):
        torch.manual_seed(0)
        tok = MazeTokenizer()
        probe = B3Probe(tok)
        rng = random.Random(0)
        # Fixed batch with a fixed h_agent → slot mapping the probe can fit.
        B = 32
        ids_list, lens, hs = [], [], []
        slot_choices = [
            Slots("top", "left", "up", "right"),
            Slots("bottom", "right", "down", "left"),
            Slots("middle", "center", "still", "up-right"),
        ]
        for i in range(B):
            slots = slot_choices[i % len(slot_choices)]
            tokens = render(slots, rng=rng, include_bos_eos=True)
            ids = tok.encode(tokens)
            ids_list.append(ids)
            lens.append(len(ids))
            # h_agent deterministic per slot class so the probe can learn.
            hs.append(torch.full((256,), float(i % len(slot_choices))))
        T = max(lens)
        padded = torch.full((B, T), tok.pad_id, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        lengths = torch.tensor(lens)
        h = torch.stack(hs)

        opt = torch.optim.AdamW(probe.interpreter_parameters(), lr=1e-2)
        loss0 = probe.update(h, padded, lengths)["loss"].item()
        for _ in range(50):
            opt.zero_grad()
            out = probe.update(h, padded, lengths)
            out["loss"].backward()
            opt.step()
        loss1 = probe.update(h, padded, lengths)["loss"].item()
        assert loss1 < loss0


# ---- 7. interpreter_parameters -----------------------------------------


class TestInterpreterParameters:
    def test_yields_all_params(self):
        tok = MazeTokenizer()
        probe = B3Probe(tok)
        a = list(probe.interpreter_parameters())
        b = list(probe.parameters())
        assert len(a) == len(b) and len(a) > 0


# ---- 8. _ce all-ignore safety ------------------------------------------


class TestCESafety:
    def test_all_ignore_returns_zero_with_graph(self):
        logits = torch.randn(4, 3, requires_grad=True)
        target = torch.full((4,), IGNORE_INDEX, dtype=torch.long)
        loss = B3Probe._ce(logits, target)
        assert loss.item() == 0.0
        # Still differentiable (no crash on backward).
        loss.backward()
        assert logits.grad is not None

    def test_partial_ignore_computes(self):
        logits = torch.randn(4, 3)
        target = torch.tensor([0, IGNORE_INDEX, 2, IGNORE_INDEX])
        loss = B3Probe._ce(logits, target)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0
