"""Tests for src/split_maze/lm_train.py — Phase 2.2 training + gate."""

import json
import random
from pathlib import Path

import pytest
import torch

from split_maze.language import BOS, EOS, vocab
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer
from split_maze.lm_train import (
    LMTrainConfig,
    _strip_trailing_pad_and_after_eos,
    build_corpus_ids,
    canonical_sentence_for_combo,
    evaluate_72_combinations,
    evaluate_roundtrip,
    gate_pass,
    iter_batches,
    split_train_held,
    train_lm,
)


# ---- LMTrainConfig (sanity that PLAN P2-5..P2-6 defaults are not drifted) -

def test_lm_train_config_defaults_match_p2_5_p2_6():
    cfg = LMTrainConfig()
    # PLAN P2-5
    assert cfg.lr == 3e-4
    assert cfg.weight_decay == 0.01
    assert cfg.grad_clip == 1.0
    # PLAN P2-6
    assert cfg.epochs == 10
    assert cfg.batch_size == 64
    # PLAN P2-3
    assert cfg.lambda_ae == 1.0
    # PLAN P2-4 / spec
    assert cfg.train_frac == 0.9
    # POST-HOC-4 (2026-05-20)
    assert cfg.warmup_steps == 500


def test_train_lm_warmup_zero_runs_normally():
    """warmup_steps=0 disables warmup entirely; lr should never be scaled."""
    tok, model, train, held = _mini_setup()
    cfg = LMTrainConfig(epochs=1, batch_size=8, lr=1e-3,
                        device="cpu", seed=0, warmup_steps=0)
    metrics = train_lm(model, tok, train, held, cfg)
    assert len(metrics) == 1
    assert torch.isfinite(torch.tensor(metrics[0]["train_loss"]))


def test_train_lm_warmup_short_runs_normally():
    """A tiny warmup (a couple of steps) must not break training."""
    tok, model, train, held = _mini_setup()
    cfg = LMTrainConfig(epochs=1, batch_size=8, lr=1e-3,
                        device="cpu", seed=0, warmup_steps=2)
    metrics = train_lm(model, tok, train, held, cfg)
    assert len(metrics) == 1
    assert torch.isfinite(torch.tensor(metrics[0]["train_loss"]))


# ---- Corpus + split + batching ----------------------------------------

def test_build_corpus_ids_returns_n_sentences():
    tok = MazeTokenizer()
    ids = build_corpus_ids(100, seed=0, tokenizer=tok)
    assert len(ids) == 100
    # Every sentence starts with <BOS> and ends with <EOS>.
    for seq in ids:
        assert seq[0] == tok.bos_id
        assert seq[-1] == tok.eos_id


def test_build_corpus_ids_deterministic_with_same_seed():
    tok = MazeTokenizer()
    a = build_corpus_ids(50, seed=42, tokenizer=tok)
    b = build_corpus_ids(50, seed=42, tokenizer=tok)
    assert a == b


def test_split_train_held_respects_fraction():
    tok = MazeTokenizer()
    ids = build_corpus_ids(100, seed=0, tokenizer=tok)
    train, held = split_train_held(ids, train_frac=0.9, seed=0)
    assert len(train) == 90
    assert len(held) == 10


def test_split_train_held_partitions_input():
    tok = MazeTokenizer()
    ids = build_corpus_ids(50, seed=1, tokenizer=tok)
    train, held = split_train_held(ids, train_frac=0.8, seed=1)
    assert len(train) + len(held) == len(ids)
    # Each sentence (by identity) appears in exactly one side.
    keys_train = {tuple(s) for s in train}
    keys_held = {tuple(s) for s in held}
    assert keys_train.isdisjoint(keys_held)


def test_iter_batches_yields_padded_tensors():
    tok = MazeTokenizer()
    ids = build_corpus_ids(20, seed=0, tokenizer=tok)
    rng = random.Random(0)
    batches = list(iter_batches(ids, batch_size=8, tokenizer=tok,
                                 rng=rng, shuffle=False))
    # 20 / 8 = 3 batches (sizes 8, 8, 4).
    assert len(batches) == 3
    assert batches[0].shape[0] == 8
    assert batches[-1].shape[0] == 4
    assert batches[0].dtype == torch.long
    # All padded to common T within each batch.
    for b in batches:
        assert b.dim() == 2


def test_iter_batches_full_pass_covers_all_rows():
    tok = MazeTokenizer()
    ids = build_corpus_ids(17, seed=0, tokenizer=tok)
    rng = random.Random(0)
    total_rows = sum(b.size(0) for b in iter_batches(
        ids, batch_size=8, tokenizer=tok, rng=rng, shuffle=True
    ))
    assert total_rows == 17


# ---- Strip helper ------------------------------------------------------

def test_strip_trailing_pad_then_truncate_after_eos():
    out = _strip_trailing_pad_and_after_eos(
        [10, 11, 3, 12, 0, 0], pad_id=0, eos_id=3,
    )
    # First strip trailing pads → [10, 11, 3, 12]; then truncate after EOS
    # at index 2 → [10, 11, 3].
    assert out == [10, 11, 3]


def test_strip_no_eos_keeps_everything_modulo_pad():
    out = _strip_trailing_pad_and_after_eos(
        [10, 11, 12, 0], pad_id=0, eos_id=3,
    )
    assert out == [10, 11, 12]


# ---- Canonical 72-combo sentences -------------------------------------

def test_canonical_sentence_for_combo_explicit_layout():
    # POST-HOC-3: four-slot form (agent + column + heading + cheese).
    s = canonical_sentence_for_combo("up-right", "down-left")
    assert s[0] == BOS
    assert s[-1] == EOS
    assert s[1:3] == ["agent", "middle"]
    assert s[3:5] == ["column", "center"]
    assert s[5:7] == ["heading", "up-right"]
    assert s[7:9] == ["cheese", "down-left"]


def test_canonical_sentence_accepts_other_agent_region():
    s = canonical_sentence_for_combo("up", "down",
                                      agent_row="top", agent_col="right")
    assert s[1:3] == ["agent", "top"]
    assert s[3:5] == ["column", "right"]


# ---- helper builders ---------------------------------------------------

def _tiny_lm() -> tuple[MazeTokenizer, MazeLM]:
    tok = MazeTokenizer()
    cfg = LMConfig.from_tokenizer(
        tok, d_model=32, n_head=4, n_layer=2, d_ff=64,
        max_len=24, dropout=0.0,
    )
    torch.manual_seed(0)
    return tok, MazeLM(cfg)


# ---- evaluate_roundtrip -----------------------------------------------

def test_evaluate_roundtrip_empty_returns_zero():
    tok, model = _tiny_lm()
    rt = evaluate_roundtrip(model, tok, [], device=torch.device("cpu"))
    assert rt["num_sentences"] == 0
    assert rt["exact_match_rate"] == 0.0
    assert rt["slot_match_rate"] == 0.0
    # Per-slot keys must exist even on the empty path.
    for k in ("agent_match_rate", "agent_row_match_rate", "agent_col_match_rate",
              "heading_match_rate", "cheese_dir_match_rate"):
        assert rt[k] == 0.0


def test_evaluate_roundtrip_per_slot_breakdown_present():
    """The Post-hoc P2-7 metric reports each slot's match rate individually so
    we can diagnose which slot dominates the slot_match average. Phase 2.2
    first-run showed agent_region was the bottleneck and the row/col
    sub-breakdown lets us see whether *both* sub-tokens collapse to mode or
    just one of them."""
    tok, model = _tiny_lm()
    sentence_ids = [tok.encode(canonical_sentence_for_combo("up", "right"))]
    rt = evaluate_roundtrip(model, tok, sentence_ids,
                             device=torch.device("cpu"))
    for k in ("agent_match_rate", "agent_row_match_rate", "agent_col_match_rate",
              "heading_match_rate", "cheese_dir_match_rate"):
        assert k in rt
        assert 0.0 <= rt[k] <= 1.0
    # slot_match_rate must be the mean of the three top-level slots.
    expected = (rt["agent_match_rate"] + rt["heading_match_rate"]
                + rt["cheese_dir_match_rate"]) / 3.0
    assert abs(rt["slot_match_rate"] - expected) < 1e-9
    # agent_match_rate cannot exceed either of its components (both must
    # hold for region to match).
    assert rt["agent_match_rate"] <= rt["agent_row_match_rate"] + 1e-9
    assert rt["agent_match_rate"] <= rt["agent_col_match_rate"] + 1e-9


def test_evaluate_roundtrip_exact_match_with_mocked_echo(monkeypatch):
    """If generate is mocked to echo the input ids exactly, exact_match
    must be 1.0 and slot_match must be 1.0."""
    tok, model = _tiny_lm()
    gold = [tok.encode(canonical_sentence_for_combo("up", "right"))]
    gold_tensor = tok.collate(gold)

    def fake_generate(self, h_lm, max_len=16, eos_id=None):
        return gold_tensor

    monkeypatch.setattr(MazeLM, "generate", fake_generate)
    rt = evaluate_roundtrip(model, tok, gold, device=torch.device("cpu"))
    assert rt["exact_match_rate"] == pytest.approx(1.0)
    assert rt["slot_match_rate"] == pytest.approx(1.0)
    assert rt["num_sentences"] == 1


def test_evaluate_roundtrip_zero_when_generate_returns_garbage(monkeypatch):
    tok, model = _tiny_lm()
    gold = [tok.encode(canonical_sentence_for_combo("up", "right"))]

    def fake_generate(self, h_lm, max_len=16, eos_id=None):
        B = h_lm.size(0)
        # Return EOS immediately for each row — no real content.
        return torch.full((B, 1), tok.eos_id, dtype=torch.long)

    monkeypatch.setattr(MazeLM, "generate", fake_generate)
    rt = evaluate_roundtrip(model, tok, gold, device=torch.device("cpu"))
    assert rt["exact_match_rate"] == 0.0


# ---- evaluate_72_combinations ----------------------------------------

def test_evaluate_72_combinations_returns_72_total():
    tok, model = _tiny_lm()
    result = evaluate_72_combinations(
        model, tok, device=torch.device("cpu"),
    )
    assert result["num_total"] == 72
    # The model is random-init → most combos likely fail, but the call must
    # at least return a well-formed result.
    assert 0 <= result["num_passed"] <= 72
    assert 0.0 <= result["pass_rate"] <= 1.0
    assert isinstance(result["failed_examples"], list)


def test_evaluate_72_combinations_perfect_with_mocked_echo(monkeypatch):
    """When generate echoes the inputs, all 72 combos must round-trip."""
    tok, model = _tiny_lm()

    captured: dict = {}

    def fake_generate(self, h_lm, max_len=16, eos_id=None):
        # We need to know what sentence_ids the caller passed to encode;
        # rebuild them from the canonical schema.
        from split_maze.language import (
            CHEESE_DIR_VALUES, HEADING_VALUES,
        )
        seqs = [tok.encode(canonical_sentence_for_combo(h, c))
                for h in HEADING_VALUES for c in CHEESE_DIR_VALUES]
        return tok.collate(seqs)

    monkeypatch.setattr(MazeLM, "generate", fake_generate)
    result = evaluate_72_combinations(
        model, tok, device=torch.device("cpu"),
    )
    assert result["num_passed"] == 72
    assert result["pass_rate"] == 1.0
    assert result["failed_examples"] == []


# ---- gate_pass --------------------------------------------------------

def test_gate_pass_both_above_thresholds():
    v = gate_pass({"slot_match_rate": 0.97, "exact_match_rate": 0.40},
                  {"pass_rate": 1.0})
    assert v["pass"] is True
    assert v["slot_pass"] is True
    assert v["combo_pass"] is True


def test_gate_pass_slot_below_threshold_fails():
    v = gate_pass({"slot_match_rate": 0.80, "exact_match_rate": 0.40},
                  {"pass_rate": 1.0})
    assert v["pass"] is False
    assert v["slot_pass"] is False
    assert v["combo_pass"] is True


def test_gate_pass_combo_below_threshold_fails():
    v = gate_pass({"slot_match_rate": 0.99, "exact_match_rate": 0.40},
                  {"pass_rate": 0.97})
    assert v["pass"] is False
    assert v["slot_pass"] is True
    assert v["combo_pass"] is False


def test_gate_pass_threshold_inclusive_and_carries_diagnostics():
    """slot_match exactly at threshold passes; exact_match carried for diagnostics."""
    v = gate_pass({"slot_match_rate": 0.95, "exact_match_rate": 0.30,
                   "agent_match_rate": 0.90, "heading_match_rate": 0.98,
                   "cheese_dir_match_rate": 0.97},
                  {"pass_rate": 1.0})
    assert v["pass"] is True
    # Diagnostic-only carryovers must come through.
    assert v["roundtrip_exact_match_rate"] == pytest.approx(0.30)
    assert v["agent_match_rate"] == pytest.approx(0.90)
    assert v["heading_match_rate"] == pytest.approx(0.98)
    assert v["cheese_dir_match_rate"] == pytest.approx(0.97)


# ---- train_lm integration smoke --------------------------------------

def _mini_setup() -> tuple[MazeTokenizer, MazeLM, list, list]:
    """Tiny corpus + tiny model for quick train_lm smoke tests."""
    tok = MazeTokenizer()
    cfg = LMConfig.from_tokenizer(
        tok, d_model=24, n_head=4, n_layer=1, d_ff=48,
        max_len=20, dropout=0.0,
    )
    torch.manual_seed(0)
    model = MazeLM(cfg)
    ids = build_corpus_ids(48, seed=0, tokenizer=tok)
    train, held = split_train_held(ids, train_frac=0.75, seed=0)
    return tok, model, train, held


def test_train_lm_returns_one_record_per_epoch():
    tok, model, train, held = _mini_setup()
    cfg = LMTrainConfig(epochs=2, batch_size=8, lr=1e-3,
                        device="cpu", seed=0)
    metrics = train_lm(model, tok, train, held, cfg)
    assert len(metrics) == 2
    for m in metrics:
        assert "train_loss" in m
        assert "held_loss" in m
        assert "held_roundtrip_exact" in m
        assert torch.isfinite(torch.tensor(m["train_loss"]))


def test_train_lm_writes_jsonl_log(tmp_path: Path):
    tok, model, train, held = _mini_setup()
    log_path = tmp_path / "lm.jsonl"
    cfg = LMTrainConfig(epochs=2, batch_size=8, lr=1e-3,
                        device="cpu", seed=0)
    train_lm(model, tok, train, held, cfg, log_path=log_path)
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["epoch"] == 1
    assert "train_loss" in rec0


def test_train_lm_saves_checkpoint(tmp_path: Path):
    tok, model, train, held = _mini_setup()
    save_path = tmp_path / "lm.pt"
    cfg = LMTrainConfig(epochs=1, batch_size=8, lr=1e-3,
                        device="cpu", seed=0)
    train_lm(model, tok, train, held, cfg, save_path=save_path)
    assert save_path.exists()
    ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
    assert "model_state" in ckpt
    assert "config" in ckpt
    assert "lm_config" in ckpt


def test_train_lm_loss_decreases_across_epochs():
    """Tiny LM on a tiny corpus must reduce the average training loss
    from epoch 1 to the final epoch (otherwise the loop is broken)."""
    tok, model, train, held = _mini_setup()
    cfg = LMTrainConfig(epochs=3, batch_size=8, lr=3e-3,
                        device="cpu", seed=0)
    metrics = train_lm(model, tok, train, held, cfg)
    assert metrics[-1]["train_loss"] < metrics[0]["train_loss"]
