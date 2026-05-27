# SPLIT-MAZE

[🇰🇷 한국어](README.md) · 🇬🇧 English · [🇨🇳 中文](README.zh.md) · [🇯🇵 日本語](README.ja.md)

*A procgen-maze RL agent and a from-scratch maze-language LM coupled by an artificial corpus callosum — testing whether the LM reports a goal-misgeneralized agent's true internal goal faithfully (not by rationalizing), both sides from scratch.*

Third in the lineage: SPLIT-9 (post-hoc adapter, negative) → SPLIT-MNIST (co-trained reconstruction ACC V2, homogeneous-toy positive) → this project. It re-tests that V2 pattern in a *heterogeneous real environment* (maze RL agent + maze-language LM, both from scratch), using *goal misgeneralization* (an agent raised where cheese is always top-right keeps going top-right OOD, ignoring the real cheese) as the faithful-vs-rationalization discriminator.
Three phases. **Phase 4** — the core hypothesis (separation-reconstruction V2 > next-token adapter B4 in faithfulness) was rejected (Scenario C); but a controlled 2×2 resolved the confound and yielded three rich negatives (goal misgen readable at the representation level · a "faithful ≠ rationalization" reframe · the ACC's single-vector bottleneck). **Phase 5** — a record-based callosum (CCM) grows to the trained-translator ceiling when both brains meet halfway (co-adaptation +0.064±0.020, 5 seeds, generalizing across 3 decider brains). **Phase 6** — live bidirectional feedback failed to make the interpreter more grounded (Δ +0.037 < threshold +0.05 + parrot-guard failure; the feedback only biases/offloads toward the goal-prior). A clean negative the pre-registered guards caught.

**Headline:** Phase 4 V2 rejected (per-slot 0.375 / swap 0.42 vs B4 0.66 / 0.83) · Phase 5 bridge grows +0.064±0.020 (5-seed · 3-brain) · Phase 6 R2 negative (Δ +0.037 < +0.05, in-dist guard failed).

**[See the details → 3-phase overview + full results & charts (KO/EN)](https://halmoneysonmat.github.io/split-maze/)**

---

### Research lineage

[SPLIT-9](https://github.com/HalmoneySonmat/split-9) → [SPLIT-MNIST](https://github.com/HalmoneySonmat/split-mnist) → **SPLIT-MAZE**

Reproduction & stack are on the details page. Tests: `pytest tests/ -q`.

---

Claude was used to build this project.
