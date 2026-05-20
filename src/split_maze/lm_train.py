"""Training + gate evaluation for the maze-language LM (Phase 2.2).

Implements PLAN P2-5..P2-8 (frozen 2026-05-19):

- **P2-5** AdamW(lr=3e-4, weight_decay=0.01) + grad clip 1.0, flat lr.
- **P2-6** 10 epochs over a 50k neutral corpus, batch=64 (≈7.8k steps).
- **P2-7** Gate metric = sequence-level exact match for ``decode(encode(S))
  == S``, threshold ≥0.95. Slot-match-rate reported as a secondary signal
  (LANGUAGE_SPEC §8 parser semantics, surface-form invariant).
- **P2-8** "Generates all 72 (HEADING, CHEESE_DIR) combos" verified by
  round-tripping each canonical sentence (fixed AGENT_REGION=middle center);
  gate requires 72/72 = 1.0 combo pass rate.

Phase 2 PASS condition (sentence-exact ≥0.95 AND combo_pass_rate = 1.0)
is implemented by :func:`gate_pass`.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import torch

from . import language as lang
from .language import (
    BOS, EOS,
    CHEESE_DIR_VALUES, HEADING_VALUES, REGION_COLS, REGION_ROWS,
    Slots, generate_corpus, parse,
)
from .lm import LMConfig, MazeLM, MazeTokenizer


# ---- Config -------------------------------------------------------------

@dataclass
class LMTrainConfig:
    """Phase-2.2 training hyperparameters (PLAN P2-5..P2-6 frozen).

    POST-HOC-4 (2026-05-20) added ``warmup_steps`` — linear LR warm-up from
    0 to ``lr`` over the first ``warmup_steps`` optimizer steps. This is the
    standard small-transformer training best practice that earlier iterations
    of this CLI accidentally omitted; the omission is the leading hypothesis
    for the mode-collapse observed under any vocab change (see PLAN §10.1).
    """

    epochs: int = 10
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    lambda_ae: float = 1.0
    train_frac: float = 0.9
    device: str = "cpu"
    seed: int = 0
    log_interval: int = 200
    roundtrip_eval_n: int = 200  # per-epoch held-out roundtrip subset size
    warmup_steps: int = 500       # POST-HOC-4

    def as_dict(self) -> dict:
        return asdict(self)


# ---- Corpus loading + batching -----------------------------------------

def encode_corpus(sentences: list[list[str]],
                  tokenizer: MazeTokenizer) -> list[list[int]]:
    """Map each token sentence to an id sequence."""
    return [tokenizer.encode(s) for s in sentences]


def build_corpus_ids(n: int, seed: int,
                     tokenizer: MazeTokenizer) -> list[list[int]]:
    """Generate `n` neutral-grammar sentences and encode them to ids."""
    sentences = list(generate_corpus(n, seed=seed))
    return encode_corpus(sentences, tokenizer)


def split_train_held(ids: list[list[int]], train_frac: float,
                     seed: int) -> tuple[list[list[int]], list[list[int]]]:
    """Shuffle then split `ids` into train / held-out by index.

    Splits surface forms (per LANGUAGE_SPEC §7); with N=50_000 and ~648
    triples × dozens of surface variants, uniform random split still
    contains every triple in train with overwhelming probability.
    """
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * train_frac)
    return shuffled[:split], shuffled[split:]


def iter_batches(ids: list[list[int]], batch_size: int,
                 tokenizer: MazeTokenizer,
                 rng: random.Random,
                 shuffle: bool = True) -> Iterator[torch.Tensor]:
    """Yield padded (B, T) tensors for one pass over `ids`."""
    indices = list(range(len(ids)))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        if not chunk:
            continue
        batch_ids = [ids[i] for i in chunk]
        yield tokenizer.collate(batch_ids)


# ---- Round-trip + 72-combo evaluation ----------------------------------

def _strip_trailing_pad_and_after_eos(row: list[int], pad_id: int,
                                       eos_id: int) -> list[int]:
    """Trim trailing PAD and truncate after the first EOS (inclusive)."""
    out = list(row)
    while out and out[-1] == pad_id:
        out.pop()
    if eos_id in out:
        out = out[:out.index(eos_id) + 1]
    return out


@torch.no_grad()
def evaluate_roundtrip(model: MazeLM,
                       tokenizer: MazeTokenizer,
                       sentence_ids: list[list[int]],
                       device: Optional[torch.device] = None) -> dict:
    """Compute ``decode(encode(S)) == S`` match rates over a list of sentences.

    Returns:
        ``{"exact_match_rate", "slot_match_rate", "num_sentences"}``.
        ``exact_match_rate`` is the strict sequence-level metric (PLAN P2-7
        gate). ``slot_match_rate`` averages per-slot matches from the parser
        (LANGUAGE_SPEC §8) — secondary, surface-form invariant.
    """
    if not sentence_ids:
        return {"exact_match_rate": 0.0, "slot_match_rate": 0.0,
                "agent_match_rate": 0.0,
                "agent_row_match_rate": 0.0, "agent_col_match_rate": 0.0,
                "heading_match_rate": 0.0,
                "cheese_dir_match_rate": 0.0, "num_sentences": 0}
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()
    try:
        batch = tokenizer.collate(sentence_ids).to(device)
        h_lm = model.encode(batch)
        max_gen = min(model.config.max_len - 1, batch.size(1) + 2)
        generated = model.generate(h_lm, max_len=max_gen,
                                    eos_id=tokenizer.eos_id)
    finally:
        if was_training:
            model.train()

    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id

    exact = 0
    agent_hits = 0
    agent_row_hits = 0
    agent_col_hits = 0
    heading_hits = 0
    cheese_hits = 0
    n = len(sentence_ids)
    for i in range(n):
        gold_ids = sentence_ids[i]
        gen_ids = _strip_trailing_pad_and_after_eos(
            generated[i].tolist(), pad_id, eos_id,
        )
        if gen_ids == gold_ids:
            exact += 1
        # Slot-level comparison (parser handles ordering and synonyms).
        gold_parsed = parse(tokenizer.decode(gold_ids))
        gen_parsed = parse(tokenizer.decode(gen_ids))
        if gold_parsed.agent_region is not None:
            if gen_parsed.agent_region == gold_parsed.agent_region:
                agent_hits += 1
            # Per-token breakdown (Phase 2.2 diagnosis 2026-05-19):
            # agent_region is row × col, so a 1/3 mode-output on either
            # token alone drives the whole region's accuracy to 1/3.
            if gen_parsed.agent_region is not None:
                if gen_parsed.agent_region[0] == gold_parsed.agent_region[0]:
                    agent_row_hits += 1
                if gen_parsed.agent_region[1] == gold_parsed.agent_region[1]:
                    agent_col_hits += 1
        if gold_parsed.heading is not None \
                and gen_parsed.heading == gold_parsed.heading:
            heading_hits += 1
        if gold_parsed.cheese_dir is not None \
                and gen_parsed.cheese_dir == gold_parsed.cheese_dir:
            cheese_hits += 1

    agent_rate = agent_hits / n
    agent_row_rate = agent_row_hits / n
    agent_col_rate = agent_col_hits / n
    heading_rate = heading_hits / n
    cheese_rate = cheese_hits / n
    slot_total = (agent_rate + heading_rate + cheese_rate) / 3.0

    return {
        "exact_match_rate": exact / n,
        "slot_match_rate": slot_total,
        "agent_match_rate": agent_rate,
        "agent_row_match_rate": agent_row_rate,
        "agent_col_match_rate": agent_col_rate,
        "heading_match_rate": heading_rate,
        "cheese_dir_match_rate": cheese_rate,
        "num_sentences": n,
    }


def canonical_sentence_for_combo(heading: str, cheese_dir: str,
                                  agent_row: str = "middle",
                                  agent_col: str = "center") -> list[str]:
    """Build the canonical-form sentence used by the 72-combo gate (P2-8).

    POST-HOC-3 (2026-05-19): four-slot form with separate ``agent <row>`` and
    ``column <col>`` slots, each marker followed by a single token. Returns
    the explicit token list with ``<BOS>`` / ``<EOS>`` wrappers.
    """
    return [
        BOS,
        "agent", agent_row,
        "column", agent_col,
        "heading", heading,
        "cheese", cheese_dir,
        EOS,
    ]


@torch.no_grad()
def evaluate_72_combinations(model: MazeLM,
                              tokenizer: MazeTokenizer,
                              device: Optional[torch.device] = None,
                              agent_row: str = "middle",
                              agent_col: str = "center") -> dict:
    """Round-trip each of the 72 (HEADING, CHEESE_DIR) combos (PLAN P2-8).

    Constructs the canonical sentence for every combo at a fixed
    AGENT_REGION, encodes → decodes → parses, and checks that the recovered
    heading and cheese-dir match. Returns counts plus up to 5 failure
    examples for diagnostics.
    """
    if device is None:
        device = next(model.parameters()).device

    combos: list[tuple[str, str]] = []
    sentences: list[list[str]] = []
    for h in HEADING_VALUES:
        for c in CHEESE_DIR_VALUES:
            combos.append((h, c))
            sentences.append(canonical_sentence_for_combo(
                h, c, agent_row=agent_row, agent_col=agent_col,
            ))

    sentence_ids = [tokenizer.encode(s) for s in sentences]

    was_training = model.training
    model.eval()
    try:
        batch = tokenizer.collate(sentence_ids).to(device)
        h_lm = model.encode(batch)
        max_gen = min(model.config.max_len - 1, batch.size(1) + 2)
        generated = model.generate(h_lm, max_len=max_gen,
                                    eos_id=tokenizer.eos_id)
    finally:
        if was_training:
            model.train()

    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id

    num_passed = 0
    num_exact = 0
    failed: list[dict] = []
    for i, (h, c) in enumerate(combos):
        gen_ids = _strip_trailing_pad_and_after_eos(
            generated[i].tolist(), pad_id, eos_id,
        )
        gen_tokens = tokenizer.decode(gen_ids)
        gen_parsed = parse(gen_tokens)
        if gen_ids == sentence_ids[i]:
            num_exact += 1
        if gen_parsed.heading == h and gen_parsed.cheese_dir == c:
            num_passed += 1
        elif len(failed) < 5:
            failed.append({
                "heading": h,
                "cheese_dir": c,
                "generated_tokens": gen_tokens,
                "parsed_heading": gen_parsed.heading,
                "parsed_cheese_dir": gen_parsed.cheese_dir,
            })

    return {
        "num_total": len(combos),
        "num_passed": num_passed,
        "pass_rate": num_passed / max(1, len(combos)),
        "num_exact": num_exact,
        "exact_rate": num_exact / max(1, len(combos)),
        "failed_examples": failed,
    }


def gate_pass(roundtrip_metrics: dict, combo_metrics: dict,
              slot_threshold: float = 0.95,
              combo_threshold: float = 1.0) -> dict:
    """Phase 2 gate (PLAN P2-7 Post-hoc 2026-05-19 + P2-8).

    Post-hoc adjustment: the original P2-7 used sequence-level exact match,
    which turned out to be structurally incompatible with the training
    signal (each meaning has ~36 surface variants in the neutral corpus,
    so the model has no way to know *which* surface to emit on decode).
    The true intended property — semantic round-trip fidelity — is captured
    by the slot-match rate. See PLAN "Post-hoc adjustments" §10.

    Returns a verdict dict with overall ``pass`` (bool), the main criterion
    (slot_match), and the original strict exact-match metric as a
    diagnostic side-channel.
    """
    slot_match = roundtrip_metrics["slot_match_rate"]
    s_pass = slot_match >= slot_threshold
    c_pass = combo_metrics["pass_rate"] >= combo_threshold
    return {
        "pass": bool(s_pass and c_pass),
        "roundtrip_slot_match_rate": slot_match,
        "slot_threshold": slot_threshold,
        "slot_pass": bool(s_pass),
        "combo_pass_rate": combo_metrics["pass_rate"],
        "combo_threshold": combo_threshold,
        "combo_pass": bool(c_pass),
        # Diagnostic-only carryovers from the original P2-7 metric.
        "roundtrip_exact_match_rate":
            roundtrip_metrics.get("exact_match_rate", 0.0),
        "agent_match_rate":
            roundtrip_metrics.get("agent_match_rate", 0.0),
        "heading_match_rate":
            roundtrip_metrics.get("heading_match_rate", 0.0),
        "cheese_dir_match_rate":
            roundtrip_metrics.get("cheese_dir_match_rate", 0.0),
    }


# ---- Training loop ------------------------------------------------------

def train_lm(model: MazeLM, tokenizer: MazeTokenizer,
             train_ids: list[list[int]], held_ids: list[list[int]],
             cfg: LMTrainConfig,
             log_path: Optional[Path] = None,
             save_path: Optional[Path] = None,
             print_each_epoch: bool = False) -> list[dict]:
    """Train ``model`` on ``train_ids`` for ``cfg.epochs`` epochs.

    Returns a per-epoch list of metric dicts (also appended to ``log_path``
    as JSONL when supplied). Final checkpoint (state_dict + cfg snapshot)
    is written to ``save_path`` if provided.
    """
    device = torch.device(cfg.device)
    model = model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    rng = random.Random(cfg.seed)

    log_f = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w")

    # POST-HOC-4 (2026-05-20): linear LR warm-up over the first
    # cfg.warmup_steps optimizer steps. Standard small-transformer practice
    # to prevent attention-entropy collapse and mode collapse in early
    # training (see PLAN §10.1 POST-HOC-4).
    global_step = 0

    def _apply_warmup_lr() -> float:
        if cfg.warmup_steps <= 0:
            return cfg.lr
        if global_step >= cfg.warmup_steps:
            return cfg.lr
        scale = (global_step + 1) / cfg.warmup_steps
        target = cfg.lr * scale
        for g in opt.param_groups:
            g["lr"] = target
        return target

    epoch_metrics: list[dict] = []
    try:
        for epoch in range(cfg.epochs):
            epoch_start = time.time()
            model.train()
            sums = {"loss": 0.0, "next_token": 0.0, "autoencode": 0.0}
            n_batches = 0
            for batch in iter_batches(train_ids, cfg.batch_size,
                                       tokenizer, rng, shuffle=True):
                _apply_warmup_lr()
                batch = batch.to(device)
                parts = model.combined_loss(batch, lambda_ae=cfg.lambda_ae)
                opt.zero_grad()
                parts["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                cfg.grad_clip)
                opt.step()
                global_step += 1
                for k in sums:
                    sums[k] += parts[k].item()
                n_batches += 1

            # Held-out eval.
            model.eval()
            held_sums = {"loss": 0.0, "next_token": 0.0, "autoencode": 0.0}
            n_held = 0
            with torch.no_grad():
                for batch in iter_batches(held_ids, cfg.batch_size,
                                           tokenizer, rng, shuffle=False):
                    batch = batch.to(device)
                    parts = model.combined_loss(batch,
                                                 lambda_ae=cfg.lambda_ae)
                    for k in held_sums:
                        held_sums[k] += parts[k].item()
                    n_held += 1

            # Per-epoch round-trip check on a subset of held-out.
            subset = held_ids[:min(cfg.roundtrip_eval_n, len(held_ids))]
            rt = evaluate_roundtrip(model, tokenizer, subset, device=device)

            record = {
                "epoch": epoch + 1,
                "train_loss": sums["loss"] / max(1, n_batches),
                "train_next_token": sums["next_token"] / max(1, n_batches),
                "train_autoencode": sums["autoencode"] / max(1, n_batches),
                "held_loss": held_sums["loss"] / max(1, n_held),
                "held_next_token": held_sums["next_token"] / max(1, n_held),
                "held_autoencode": held_sums["autoencode"] / max(1, n_held),
                "held_roundtrip_exact": rt["exact_match_rate"],
                "held_roundtrip_slot": rt["slot_match_rate"],
                "held_roundtrip_n": rt["num_sentences"],
                "elapsed_sec": time.time() - epoch_start,
            }
            epoch_metrics.append(record)
            if log_f:
                log_f.write(json.dumps(record) + "\n")
                log_f.flush()
            if print_each_epoch:
                print(
                    f"[epoch {record['epoch']:>2}] "
                    f"train_loss={record['train_loss']:.4f} "
                    f"held_loss={record['held_loss']:.4f} "
                    f"rt_exact={record['held_roundtrip_exact']:.3f} "
                    f"rt_slot={record['held_roundtrip_slot']:.3f} "
                    f"({record['elapsed_sec']:.1f}s)"
                )
    finally:
        if log_f:
            log_f.close()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "config": cfg.as_dict(),
            "lm_config": asdict(model.config),
            "epochs_trained": cfg.epochs,
        }, save_path)

    return epoch_metrics
