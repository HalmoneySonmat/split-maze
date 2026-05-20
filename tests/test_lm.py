"""Tests for src/split_maze/lm.py — Phase 2 maze-language LM."""

import random

import pytest
import torch
import torch.nn as nn

from split_maze.language import (
    BOS, EOS, PAD, SUM,
    vocab,
    sample_slots, render, generate_corpus,
)
from split_maze.lm import LMConfig, MazeLM, MazeTokenizer


# ---- Tokenizer ---------------------------------------------------------

def test_tokenizer_vocab_size_matches_language_vocab():
    tok = MazeTokenizer()
    assert tok.vocab_size == len(vocab())
    assert tok.tokens == vocab()  # sorted, deterministic


def test_tokenizer_special_ids_resolve_to_special_strings():
    tok = MazeTokenizer()
    assert tok.tokens[tok.pad_id] == PAD
    assert tok.tokens[tok.bos_id] == BOS
    assert tok.tokens[tok.eos_id] == EOS
    assert tok.tokens[tok.sum_id] == SUM


def test_tokenizer_round_trip_on_corpus_sentence():
    tok = MazeTokenizer()
    rng = random.Random(0)
    sentence = render(sample_slots(rng), rng=rng)  # [<BOS>, ..., <EOS>]
    ids = tok.encode(sentence)
    assert tok.decode(ids) == sentence


def test_tokenizer_round_trip_handles_all_vocab_tokens():
    tok = MazeTokenizer()
    # Every single vocab token (including all specials) should encode/decode
    # to itself. This catches off-by-one bugs in the id table.
    ids = tok.encode(tok.tokens)
    assert tok.decode(ids) == tok.tokens


def test_collate_pads_to_max_length_with_pad_id():
    tok = MazeTokenizer()
    seqs = [[tok.bos_id, 1, 2, tok.eos_id],
            [tok.bos_id, 3, tok.eos_id]]
    out = tok.collate(seqs)
    assert out.shape == (2, 4)
    assert out.dtype == torch.long
    # Second sequence's last cell should be pad.
    assert int(out[1, 3]) == tok.pad_id
    # First sequence preserved exactly.
    assert out[0].tolist() == seqs[0]


def test_collate_empty_list_raises():
    tok = MazeTokenizer()
    with pytest.raises(ValueError):
        tok.collate([])


# ---- LMConfig ----------------------------------------------------------

def test_lmconfig_from_tokenizer_copies_special_ids():
    tok = MazeTokenizer()
    cfg = LMConfig.from_tokenizer(tok)
    assert cfg.vocab_size == tok.vocab_size
    assert cfg.pad_id == tok.pad_id
    assert cfg.bos_id == tok.bos_id
    assert cfg.eos_id == tok.eos_id
    assert cfg.sum_id == tok.sum_id


def test_lmconfig_overrides_apply():
    tok = MazeTokenizer()
    cfg = LMConfig.from_tokenizer(tok, d_model=64, n_layer=2, n_head=2,
                                  d_ff=128, max_len=16)
    assert cfg.d_model == 64
    assert cfg.n_layer == 2


# ---- helpers -----------------------------------------------------------

def _tiny_lm(max_len: int = 24) -> tuple[MazeTokenizer, LMConfig, MazeLM]:
    """A miniature MazeLM that exercises the full pipeline cheaply."""
    tok = MazeTokenizer()
    cfg = LMConfig.from_tokenizer(
        tok, d_model=32, n_head=4, n_layer=2, d_ff=64,
        max_len=max_len, dropout=0.0,
    )
    torch.manual_seed(0)
    model = MazeLM(cfg)
    model.eval()  # remove dropout randomness for deterministic tests
    return tok, cfg, model


def _sample_corpus_batch(tok: MazeTokenizer, n: int = 4,
                         seed: int = 1) -> torch.Tensor:
    seqs = [tok.encode(s) for s in generate_corpus(n, seed=seed)]
    return tok.collate(seqs)


# ---- MazeLM structure --------------------------------------------------

def test_mazelm_forward_shapes_match_appended_sum():
    tok, cfg, model = _tiny_lm()
    ids = _sample_corpus_batch(tok)
    logits, h_lm = model(ids)
    # forward appends <SUM>: logits length is T + 1.
    assert logits.shape == (ids.size(0), ids.size(1) + 1, cfg.vocab_size)
    assert h_lm.shape == (ids.size(0), cfg.d_model)


def test_encode_returns_only_summary_vector():
    tok, cfg, model = _tiny_lm()
    ids = _sample_corpus_batch(tok)
    h_lm = model.encode(ids)
    assert h_lm.shape == (ids.size(0), cfg.d_model)


def test_decode_logits_has_h_lm_plus_prefix_length():
    tok, cfg, model = _tiny_lm()
    B, T_p = 2, 5
    h_lm = torch.zeros(B, cfg.d_model)
    prefix = torch.randint(0, cfg.vocab_size, (B, T_p))
    logits = model.decode_logits(h_lm, prefix)
    assert logits.shape == (B, 1 + T_p, cfg.vocab_size)


def test_weight_tying_between_lm_head_and_tok_embed():
    _, _, model = _tiny_lm()
    assert model.lm_head.weight.data_ptr() == model.tok_embed.weight.data_ptr()


def test_h_lm_responds_to_sum_token_embedding():
    """The hidden producing h_lm sits at a position fed the <SUM> embedding.
    Mutating that embedding row must shift h_lm — confirming <SUM> was used."""
    tok, cfg, model = _tiny_lm()
    ids = _sample_corpus_batch(tok, n=2)
    h1 = model.encode(ids).clone()
    with torch.no_grad():
        model.tok_embed.weight[tok.sum_id].add_(1.0)
    h2 = model.encode(ids)
    assert not torch.allclose(h1, h2)


# ---- Causal mask -------------------------------------------------------

def test_causal_mask_no_future_leak():
    """Hidden at position t must be invariant to changes at positions > t."""
    tok, cfg, model = _tiny_lm()
    B, T = 1, 6
    ids_a = torch.randint(0, cfg.vocab_size, (B, T))
    ids_b = ids_a.clone()
    ids_b[:, -1] = (ids_b[:, -1] + 1) % cfg.vocab_size  # mutate last position

    # We need the *raw* per-position hidden before interface_proj, so we
    # replicate the transformer pipeline directly: tok_embed → _transform.
    with torch.no_grad():
        emb_a = model.tok_embed(ids_a)
        emb_b = model.tok_embed(ids_b)
        h_a = model._transform(emb_a)
        h_b = model._transform(emb_b)
    # Positions 0..T-2 must be identical (causal: they cannot see position T-1).
    assert torch.allclose(h_a[:, :-1], h_b[:, :-1], atol=1e-6)
    # Last position should differ (the only differing input).
    assert not torch.allclose(h_a[:, -1], h_b[:, -1], atol=1e-6)


# ---- Losses ------------------------------------------------------------

def test_next_token_loss_is_scalar_with_grad():
    tok, _, model = _tiny_lm()
    model.train()
    ids = _sample_corpus_batch(tok, n=4)
    loss = model.next_token_loss(ids)
    assert loss.dim() == 0
    assert loss.requires_grad
    assert loss.grad_fn is not None
    loss.backward()
    # The shared tok_embed weight should receive a gradient.
    assert model.tok_embed.weight.grad is not None


def test_autoencode_loss_is_scalar_with_grad():
    tok, _, model = _tiny_lm()
    model.train()
    ids = _sample_corpus_batch(tok, n=4)
    loss = model.autoencode_loss(ids)
    assert loss.dim() == 0
    assert loss.requires_grad
    loss.backward()
    # Interface projection should receive gradient (it sits on the encode side).
    assert model.interface_proj.weight.grad is not None


def test_combined_loss_components_sum_to_total():
    tok, _, model = _tiny_lm()
    model.train()
    ids = _sample_corpus_batch(tok, n=2)
    parts = model.combined_loss(ids, lambda_ae=1.0)
    expected = parts["next_token"] + 1.0 * parts["autoencode"]
    assert torch.allclose(parts["loss"], expected)


def test_combined_loss_lambda_ae_scales_autoencode_component():
    tok, _, model = _tiny_lm()
    model.train()
    ids = _sample_corpus_batch(tok, n=2)
    parts_05 = model.combined_loss(ids, lambda_ae=0.5)
    parts_10 = model.combined_loss(ids, lambda_ae=1.0)
    # Components themselves are deterministic in eval-like settings; here
    # we just check the total reflects the lambda properly relative to the
    # same components within the same call.
    expected_05 = parts_05["next_token"] + 0.5 * parts_05["autoencode"]
    expected_10 = parts_10["next_token"] + 1.0 * parts_10["autoencode"]
    assert torch.allclose(parts_05["loss"], expected_05)
    assert torch.allclose(parts_10["loss"], expected_10)


def test_cross_entropy_ignore_index_drops_pad_position():
    """The property next_token_loss / autoencode_loss actually rely on:
    perturbing the logits at an *ignored* position must not change the loss.
    This directly tests "pad does not contribute" without making assumptions
    about how the mean denominator is computed."""
    tok, cfg, _ = _tiny_lm()
    V = cfg.vocab_size
    logits_a = torch.randn(3, V)
    target = torch.tensor([3, 4, tok.pad_id])
    loss_a = torch.nn.functional.cross_entropy(
        logits_a, target, ignore_index=tok.pad_id,
    )
    # Perturb *only* the ignored row's logits — wildly.
    logits_b = logits_a.clone()
    logits_b[2] = torch.randn(V) * 1000.0
    loss_b = torch.nn.functional.cross_entropy(
        logits_b, target, ignore_index=tok.pad_id,
    )
    assert torch.allclose(loss_a, loss_b)


def test_autoencode_loss_finite_with_pad_in_batch():
    """Integration smoke: a batch with right-padded rows still yields a
    well-defined, finite autoencode loss."""
    tok, _, model = _tiny_lm()
    model.eval()
    rng = random.Random(7)
    s1 = render(sample_slots(rng), rng=rng)
    s2 = render(sample_slots(rng), rng=rng)
    ids = tok.collate([tok.encode(s1), tok.encode(s2)])
    loss = model.autoencode_loss(ids)
    assert torch.isfinite(loss)


# ---- Generation --------------------------------------------------------

def test_generate_returns_proper_shape():
    tok, cfg, model = _tiny_lm()
    h_lm = torch.zeros(3, cfg.d_model)
    out = model.generate(h_lm, max_len=8)
    assert out.dim() == 2 and out.size(0) == 3
    assert out.size(1) <= 8


def test_generate_caps_at_config_max_len_minus_one():
    tok, cfg, model = _tiny_lm(max_len=8)
    h_lm = torch.zeros(2, cfg.d_model)
    out = model.generate(h_lm, max_len=1000)
    # position 0 holds h_lm itself, leaving max_len-1 slots for tokens.
    assert out.size(1) <= cfg.max_len - 1


def test_generate_after_eos_emits_pad_only():
    """If a row hits <EOS>, all subsequent emitted tokens for that row must
    be <PAD>. We force this by patching the lm_head to always predict <EOS>."""
    tok, cfg, model = _tiny_lm()

    class _AlwaysEOS(nn.Module):
        def __init__(self, vocab_size: int, eos_id: int):
            super().__init__()
            self.weight = torch.zeros(vocab_size, cfg.d_model)
            self.eos_id = eos_id
            self.vocab_size = vocab_size

        def forward(self, x):
            B = x.shape[0] if x.dim() > 1 else 1
            # Return logits that strongly favour eos_id along the last dim.
            shape = x.shape[:-1] + (self.vocab_size,)
            logits = torch.full(shape, -1e4)
            logits[..., self.eos_id] = 0.0
            return logits

    model.lm_head = _AlwaysEOS(cfg.vocab_size, cfg.eos_id)
    h_lm = torch.zeros(1, cfg.d_model)
    out = model.generate(h_lm, max_len=5)
    # First token = <EOS>; rest must be pad.
    assert int(out[0, 0]) == cfg.eos_id
    assert all(int(out[0, i]) == cfg.pad_id for i in range(1, out.size(1)))


# ---- Parameter split (Phase 3 stop-grad readiness) --------------------

def test_core_and_interface_parameters_are_disjoint():
    _, _, model = _tiny_lm()
    iface = {id(p) for p in model.interface_parameters()}
    core = {id(p) for p in model.core_parameters()}
    assert iface and core
    assert iface.isdisjoint(core)


def test_core_plus_interface_covers_all_model_parameters():
    _, _, model = _tiny_lm()
    union = set()
    union.update(id(p) for p in model.interface_parameters())
    union.update(id(p) for p in model.core_parameters())
    all_ids = {id(p) for p in model.parameters()}
    assert union == all_ids


def test_interface_only_optim_does_not_change_core():
    """Phase 3 stop-grad sanity: optimising over interface_parameters() only
    must leave every core parameter exactly equal across a backward + step."""
    tok, _, model = _tiny_lm()
    opt = torch.optim.SGD(list(model.interface_parameters()), lr=1.0)
    core_before = {n: p.detach().clone() for n, p in model.named_parameters()
                   if id(p) not in {id(q) for q in model.interface_parameters()}}
    ids = _sample_corpus_batch(tok, n=2)
    loss = model.autoencode_loss(ids)
    loss.backward()
    opt.step()
    for n, p in model.named_parameters():
        if n in core_before:
            assert torch.equal(core_before[n], p.detach()), \
                f"core parameter {n} changed unexpectedly"


# ---- End-to-end with the real corpus generator -----------------------

def test_lm_consumes_real_corpus_sentences():
    """Sanity: the same tokenizer/LM happily processes whatever
    language.generate_corpus emits, including all surface variations."""
    tok, _, model = _tiny_lm()
    seqs = [tok.encode(s) for s in generate_corpus(8, seed=42)]
    ids = tok.collate(seqs)
    logits, h_lm = model(ids)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(h_lm).all()
