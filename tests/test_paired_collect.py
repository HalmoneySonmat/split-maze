"""Unit tests for split_maze.paired_collect.

Organised by concern:

1. PairBufferConfig defaults match Phase-3 박제값 (PLAN §10.2 P3-4, P3-3
   + §10.3 P3-2-2 / P3-2-3 박제).
2. PairBuffer construction — pre-allocated buffer shapes, initial state.
3. PairBuffer.add — single batch, multiple batches, FIFO wrap-around,
   B ≥ capacity edge.
4. PairBuffer.sample — shape, snapshot stability (clone), error paths.
5. PairBuffer.sample uniform-ish — every index gets some draws when
   buffer has small fixed size and we sample many times.
6. PairedCollector.__init__ — stride validation.
7. PairedCollector._make_pair — happy path, None paths (state=None,
   oracle returns None, length overflow).
8. PairedCollector.extract_into — happy path stride math, skips,
   buffer integration, surface-form variation via rng.
9. Integration — round-trip extract → sample → MazeLM.encode-shape
   sanity (no actual model needed for tests).
"""

from __future__ import annotations

import random
import pytest

torch = pytest.importorskip("torch")

from split_maze.language import (  # noqa: E402
    MazeState,
    Slots,
    describer_oracle,
)
from split_maze.lm import MazeTokenizer  # noqa: E402
from split_maze.paired_collect import (  # noqa: E402
    PairBuffer,
    PairBufferConfig,
    PairedCollector,
)


# ---- helpers ------------------------------------------------------------


def _maze_state_at(agent_xy=(0.2, 0.2), cheese_xy=(0.9, 0.1)):
    """A typical MazeState (agent off the cheese, well-defined heading)."""
    return MazeState(
        agent_xy=agent_xy,
        cheese_xy=cheese_xy,
        maze_size=(1.0, 1.0),
        recent_trajectory=((0.0, 0.0), (0.1, 0.1), (0.15, 0.15), (0.2, 0.2)),
    )


def _oracle_always_none(_state):
    """Oracle stub that always returns None (agent-on-cheese all the time)."""
    return None


# ---- 1. Config defaults --------------------------------------------------


class TestPairBufferConfigDefaults:
    def test_capacity_is_256000(self):
        # PLAN §10.2 P3-4-A.
        assert PairBufferConfig().capacity == 256_000

    def test_batch_size_is_128(self):
        # PLAN §10.2 P3-3-A.
        assert PairBufferConfig().batch_size == 128

    def test_d_agent_is_256(self):
        assert PairBufferConfig().d_agent == 256

    def test_max_token_len_is_16(self):
        assert PairBufferConfig().max_token_len == 16


# ---- 2. PairBuffer construction -----------------------------------------


class TestPairBufferConstruction:
    def test_empty_initial(self):
        buf = PairBuffer(PairBufferConfig(capacity=64, batch_size=4,
                                          d_agent=8, max_token_len=4))
        assert len(buf) == 0
        assert not buf.is_ready()
        assert buf.capacity == 64

    def test_buffer_shapes(self):
        cfg = PairBufferConfig(capacity=32, batch_size=4, d_agent=8,
                               max_token_len=6)
        buf = PairBuffer(cfg)
        assert buf.h_agent_buf.shape == (32, 8)
        assert buf.ids_buf.shape == (32, 6)
        assert buf.lengths_buf.shape == (32,)

    def test_ids_buf_initially_pad_filled(self):
        cfg = PairBufferConfig(capacity=8, batch_size=2, d_agent=4,
                               max_token_len=3)
        buf = PairBuffer(cfg, pad_id=7)
        assert (buf.ids_buf == 7).all()

    def test_is_ready_threshold(self):
        cfg = PairBufferConfig(capacity=16, batch_size=4, d_agent=4,
                               max_token_len=3)
        buf = PairBuffer(cfg)
        h = torch.randn(3, 4)
        ids = torch.zeros(3, 3, dtype=torch.long)
        lens = torch.tensor([1, 2, 3])
        buf.add(h, ids, lens)
        assert not buf.is_ready()           # 3 < 4
        assert buf.is_ready(n=3)            # custom n
        buf.add(torch.randn(1, 4),
                torch.zeros(1, 3, dtype=torch.long),
                torch.tensor([1]))
        assert buf.is_ready()                # 4 ≥ 4


# ---- 3. PairBuffer.add --------------------------------------------------


def _make_buf(capacity=8, batch_size=2, d_agent=4, max_token_len=3):
    cfg = PairBufferConfig(
        capacity=capacity,
        batch_size=batch_size,
        d_agent=d_agent,
        max_token_len=max_token_len,
    )
    return PairBuffer(cfg)


class TestPairBufferAdd:
    def test_add_returns_count(self):
        buf = _make_buf()
        h = torch.randn(3, 4)
        ids = torch.zeros(3, 3, dtype=torch.long)
        lens = torch.tensor([1, 1, 1])
        assert buf.add(h, ids, lens) == 3
        assert len(buf) == 3

    def test_add_increments_len(self):
        buf = _make_buf()
        buf.add(torch.randn(2, 4), torch.zeros(2, 3, dtype=torch.long),
                torch.tensor([1, 2]))
        buf.add(torch.randn(2, 4), torch.zeros(2, 3, dtype=torch.long),
                torch.tensor([1, 2]))
        assert len(buf) == 4

    def test_add_cap_at_capacity(self):
        buf = _make_buf(capacity=4)
        # Add 6 entries to a capacity-4 buffer.
        buf.add(torch.randn(6, 4),
                torch.zeros(6, 3, dtype=torch.long),
                torch.tensor([1, 2, 3, 4, 5, 6]))
        assert len(buf) == 4

    def test_add_fifo_wraparound_data(self):
        """After wrap, the buffer contains the last `capacity` entries."""
        buf = _make_buf(capacity=4, d_agent=2, max_token_len=2)
        # Entry i has h_agent = [i, i], length = i.
        for i in range(7):
            buf.add(torch.tensor([[float(i), float(i)]]),
                    torch.tensor([[i, i]], dtype=torch.long),
                    torch.tensor([i]))
        assert len(buf) == 4
        # The last four entries are i=3,4,5,6. They live in slots
        # determined by write_ptr at the time. Easier test: every length
        # in the buffer is between 3 and 6.
        lengths_in_buf = set(buf.lengths_buf.tolist())
        assert lengths_in_buf == {3, 4, 5, 6}

    def test_add_huge_batch_keeps_last_capacity(self):
        buf = _make_buf(capacity=3, d_agent=2, max_token_len=2)
        # Single add larger than capacity → keep last 3 entries.
        h = torch.tensor([[float(i), float(i)] for i in range(5)])
        ids = torch.tensor([[i, i] for i in range(5)], dtype=torch.long)
        lens = torch.tensor([0, 1, 2, 3, 4])
        n = buf.add(h, ids, lens)
        assert n == 3
        assert len(buf) == 3
        assert set(buf.lengths_buf.tolist()) == {2, 3, 4}

    def test_add_zero_batch_noop(self):
        buf = _make_buf()
        n = buf.add(
            torch.empty(0, 4),
            torch.empty(0, 3, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
        assert n == 0
        assert len(buf) == 0

    def test_add_shape_validation(self):
        buf = _make_buf(d_agent=4, max_token_len=3)
        with pytest.raises(ValueError, match="h_agent must be"):
            buf.add(torch.randn(2, 99),
                    torch.zeros(2, 3, dtype=torch.long),
                    torch.tensor([1, 1]))
        with pytest.raises(ValueError, match="ids must be"):
            buf.add(torch.randn(2, 4),
                    torch.zeros(2, 99, dtype=torch.long),
                    torch.tensor([1, 1]))
        with pytest.raises(ValueError, match="lengths must be"):
            buf.add(torch.randn(2, 4),
                    torch.zeros(2, 3, dtype=torch.long),
                    torch.tensor([1, 1, 1]))

    def test_add_detaches_h_agent(self):
        # Caller may pass a tensor with grad — buffer must store
        # a detached copy.
        buf = _make_buf()
        h = torch.randn(2, 4, requires_grad=True)
        ids = torch.zeros(2, 3, dtype=torch.long)
        lens = torch.tensor([1, 1])
        buf.add(h, ids, lens)
        # Reading back from the buffer: no grad.
        stored = buf.h_agent_buf[:2]
        assert not stored.requires_grad


# ---- 4. PairBuffer.sample ----------------------------------------------


class TestPairBufferSample:
    def _filled(self, n=5):
        buf = _make_buf(capacity=16, batch_size=2, d_agent=4, max_token_len=3)
        h = torch.arange(n * 4, dtype=torch.float).reshape(n, 4)
        ids = torch.arange(n * 3, dtype=torch.long).reshape(n, 3)
        lens = torch.arange(n, dtype=torch.long)
        buf.add(h, ids, lens)
        return buf

    def test_sample_default_batch(self):
        buf = self._filled()
        out = buf.sample()  # batch_size=2
        assert out["h_agent"].shape == (2, 4)
        assert out["ids"].shape == (2, 3)
        assert out["lengths"].shape == (2,)

    def test_sample_custom_n(self):
        buf = self._filled()
        out = buf.sample(n=3)
        assert out["h_agent"].shape == (3, 4)

    def test_sample_returns_clone(self):
        """Sampled rows must be stable across subsequent overwrites."""
        buf = self._filled(n=4)
        out = buf.sample(n=4)
        h_before = out["h_agent"].clone()
        # Overwrite buffer with fresh data
        new_h = torch.full((4, 4), 99.0)
        new_ids = torch.full((4, 3), 9, dtype=torch.long)
        new_lens = torch.tensor([9, 9, 9, 9])
        # Re-seat write pointer to 0 so we know we overwrite slots 0..3
        buf._write_ptr = 0
        buf.add(new_h, new_ids, new_lens)
        assert torch.equal(out["h_agent"], h_before)

    def test_sample_too_few_raises(self):
        buf = self._filled(n=2)
        with pytest.raises(ValueError, match="buffer has 2 entries"):
            buf.sample(n=5)

    def test_sample_zero_or_negative_raises(self):
        buf = self._filled(n=4)
        with pytest.raises(ValueError, match="n must be positive"):
            buf.sample(n=0)
        with pytest.raises(ValueError, match="n must be positive"):
            buf.sample(n=-1)

    def test_sample_is_uniformish(self):
        """Sample many times from a tiny buffer; every index should
        appear at least once."""
        buf = self._filled(n=4)
        gen = torch.Generator().manual_seed(0)
        seen = set()
        for _ in range(50):
            out = buf.sample(n=4, generator=gen)
            for length in out["lengths"].tolist():
                seen.add(length)
        # Lengths 0,1,2,3 should all show up.
        assert seen == {0, 1, 2, 3}


# ---- 5. PairedCollector.__init__ ---------------------------------------


class TestPairedCollectorInit:
    def test_stride_validation(self):
        tok = MazeTokenizer()
        with pytest.raises(ValueError, match="stride"):
            PairedCollector(tok, stride=0)

    def test_max_token_len_validation(self):
        tok = MazeTokenizer()
        with pytest.raises(ValueError, match="max_token_len"):
            PairedCollector(tok, max_token_len=0)

    def test_default_oracle_is_describer(self):
        tok = MazeTokenizer()
        coll = PairedCollector(tok)
        assert coll.oracle is describer_oracle


# ---- 6. PairedCollector._make_pair -------------------------------------


class TestPairedCollectorMakePair:
    def test_happy_path_returns_padded_ids_and_length(self):
        tok = MazeTokenizer()
        coll = PairedCollector(tok, max_token_len=16)
        rng = random.Random(0)
        pair = coll._make_pair(_maze_state_at(), rng)
        assert pair is not None
        padded, length = pair
        assert padded.shape == (16,)
        assert length >= 1
        # Trailing tail is padded.
        assert (padded[length:] == tok.pad_id).all()
        # Active head is not all pad.
        assert not (padded[:length] == tok.pad_id).all()

    def test_none_state_returns_none(self):
        tok = MazeTokenizer()
        coll = PairedCollector(tok)
        assert coll._make_pair(None, random.Random(0)) is None

    def test_oracle_none_returns_none(self):
        tok = MazeTokenizer()
        coll = PairedCollector(tok, oracle=_oracle_always_none)
        assert coll._make_pair(_maze_state_at(), random.Random(0)) is None

    def test_length_overflow_returns_none(self):
        """If we artificially set max_token_len smaller than a real sentence,
        the pair is skipped."""
        tok = MazeTokenizer()
        coll = PairedCollector(tok, max_token_len=1)  # impossibly small
        assert coll._make_pair(_maze_state_at(), random.Random(0)) is None


# ---- 7. PairedCollector.extract_into -----------------------------------


class TestPairedCollectorExtract:
    def _setup(self, T=8, N=2, stride=4, max_token_len=16):
        cfg = PairBufferConfig(
            capacity=64, batch_size=4, d_agent=8, max_token_len=max_token_len
        )
        buf = PairBuffer(cfg)
        tok = MazeTokenizer()
        coll = PairedCollector(tok, stride=stride, max_token_len=max_token_len)
        h = torch.randn(T, N, 8)
        states = [
            [_maze_state_at() for _ in range(N)] for _ in range(T)
        ]
        return coll, buf, h, states

    def test_stride_count_all_valid(self):
        coll, buf, h, states = self._setup(T=8, N=2, stride=4)
        n = coll.extract_into(buf, h, states, rng=random.Random(0))
        # ⌈8/4⌉ = 2 stride steps × N=2 envs = 4 pairs (assuming oracle ok).
        assert n == 4
        assert len(buf) == 4

    def test_skips_none_states(self):
        coll, buf, h, states = self._setup(T=8, N=2, stride=4)
        # Corrupt one stride-step state to None.
        states[0][1] = None
        n = coll.extract_into(buf, h, states, rng=random.Random(0))
        assert n == 3

    def test_skips_oracle_none(self):
        cfg = PairBufferConfig(
            capacity=64, batch_size=4, d_agent=8, max_token_len=16
        )
        buf = PairBuffer(cfg)
        tok = MazeTokenizer()
        coll = PairedCollector(
            tok, oracle=_oracle_always_none, stride=4, max_token_len=16
        )
        h = torch.randn(8, 2, 8)
        states = [[_maze_state_at() for _ in range(2)] for _ in range(8)]
        assert coll.extract_into(buf, h, states) == 0
        assert len(buf) == 0

    def test_returns_zero_when_no_valid_pairs(self):
        coll, buf, h, states = self._setup(T=4, N=2, stride=4)
        states = [[None] * 2 for _ in range(4)]
        assert coll.extract_into(buf, h, states) == 0

    def test_extract_validates_h_agent_dims(self):
        coll, buf, _, states = self._setup(T=8, N=2)
        with pytest.raises(ValueError, match="h_agent must be"):
            coll.extract_into(buf, torch.randn(8, 2), states)  # missing d_a

    def test_extract_validates_d_agent_match(self):
        coll, buf, _, states = self._setup(T=8, N=2)
        wrong_h = torch.randn(8, 2, 99)
        with pytest.raises(ValueError, match="d_agent mismatch"):
            coll.extract_into(buf, wrong_h, states)

    def test_extract_validates_max_token_len_match(self):
        cfg = PairBufferConfig(
            capacity=8, batch_size=2, d_agent=4, max_token_len=8
        )
        buf = PairBuffer(cfg)
        tok = MazeTokenizer()
        coll = PairedCollector(tok, stride=4, max_token_len=16)
        with pytest.raises(ValueError, match="max_token_len"):
            coll.extract_into(buf, torch.randn(4, 1, 4),
                              [[_maze_state_at()]] * 4)

    def test_extract_validates_states_outer_length(self):
        coll, buf, h, _ = self._setup(T=4, N=2)
        states = [[_maze_state_at()] * 2 for _ in range(3)]   # wrong T
        with pytest.raises(ValueError, match="maze_states outer length"):
            coll.extract_into(buf, h, states)

    def test_extract_validates_states_inner_length(self):
        coll, buf, h, _ = self._setup(T=4, N=2)
        states = [[_maze_state_at()] for _ in range(4)]   # wrong N (1 not 2)
        with pytest.raises(ValueError, match=r"maze_states\[0\] must have N"):
            coll.extract_into(buf, h, states)

    def test_extract_preserves_h_agent_values(self):
        """A pair's h_agent slot in the buffer should match h_agent[t,n]."""
        coll, buf, h, states = self._setup(T=4, N=1, stride=4)
        coll.extract_into(buf, h, states, rng=random.Random(0))
        assert len(buf) == 1
        stored = buf.h_agent_buf[0]
        assert torch.equal(stored, h[0, 0])


# ---- 8. Surface-form variation through rng -----------------------------


class TestSurfaceFormVariation:
    def test_same_rng_seed_same_output(self):
        tok = MazeTokenizer()
        coll = PairedCollector(tok, max_token_len=16)
        rng_a = random.Random(7)
        rng_b = random.Random(7)
        state = _maze_state_at()
        a = coll._make_pair(state, rng_a)
        b = coll._make_pair(state, rng_b)
        assert a is not None and b is not None
        assert torch.equal(a[0], b[0])
        assert a[1] == b[1]

    def test_different_rng_seeds_eventually_differ(self):
        tok = MazeTokenizer()
        coll = PairedCollector(tok, max_token_len=16)
        state = _maze_state_at()
        seen = set()
        for seed in range(30):
            pair = coll._make_pair(state, random.Random(seed))
            if pair is None:
                continue
            ids, length = pair
            seen.add(tuple(ids[:length].tolist()))
        # At least two distinct surface forms across 30 seeds.
        assert len(seen) >= 2


# ---- 9. Integration / smoke --------------------------------------------


class TestIntegrationSmoke:
    def test_extract_then_sample_round_trip(self):
        cfg = PairBufferConfig(
            capacity=64, batch_size=4, d_agent=8, max_token_len=16
        )
        buf = PairBuffer(cfg)
        tok = MazeTokenizer()
        coll = PairedCollector(tok, stride=2, max_token_len=16)
        T, N = 8, 4
        h = torch.randn(T, N, 8)
        states = [[_maze_state_at() for _ in range(N)] for _ in range(T)]
        added = coll.extract_into(buf, h, states, rng=random.Random(0))
        # T/stride = 4 stride-steps × N=4 = 16
        assert added == 16
        assert len(buf) == 16
        assert buf.is_ready()
        out = buf.sample()
        assert out["h_agent"].shape == (4, 8)
        assert out["ids"].shape == (4, 16)
        # Every sampled length is at least the minimum maze sentence
        # (4 markers + 4 values + BOS + EOS = 10) — sanity that no row
        # leaked from initial pad-filled buffer.
        assert (out["lengths"] >= 8).all()
        assert (out["lengths"] <= 16).all()

    def test_default_p32_hyperparams_match_session_handoff(self):
        cfg = PairBufferConfig()
        # PLAN §10.2 P3-4 + §10.3 P3-2-2.
        assert cfg.capacity == 256_000
        assert cfg.batch_size == 128
        assert cfg.d_agent == 256
        # PairedCollector default stride is implicit per ctor — verify.
        tok = MazeTokenizer()
        coll = PairedCollector(tok)
        assert coll.stride == 4
        assert coll.max_token_len == 16
