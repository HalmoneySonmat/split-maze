# SPLIT-MAZE

[🇰🇷 한국어](README.md) · 🇬🇧 English · [🇨🇳 中文](README.zh.md) · [🇯🇵 日本語](README.ja.md)

*A project verifying — with both networks trained from scratch — whether a small synthetic-maze LM, coupled to an IMPALA-CNN RL agent through an artificial corpus callosum (ACC), can report a goal-misgeneralized agent's actual internal goal *faithfully rather than as a rationalization*.*

> **TL;DR.** The neighboring project [SPLIT-9](../split_brain_go) ended on a
> negative: a *frozen LLM + post-hoc adapter* hits a structural ceiling on
> faithfulness. [SPLIT-MNIST](../split_mnist) showed that *co-training + a
> decoupled reconstruction ACC (V2)* breaks that ceiling — in a homogeneous
> toy (Scenario A). This project re-tests the V2 pattern **in a heterogeneous
> real environment, with both sides from scratch**: a procgen-maze RL agent
> (the actor) and a synthetic-maze decoder-LM (the interpreter), using *goal
> misgeneralization* (an agent trained where cheese was always top-right keeps
> heading top-right OOD even when the cheese is elsewhere) as the discriminator
> between faithful interpretation and rationalization. **The core hypothesis
> (separated-reconstruction V2 > next-token adapter B4 in faithfulness) is
> rejected (Scenario C)** — V2 is *less* faithful (per-slot 0.375 / active-swap
> 0.42 vs B4 0.66 / 0.83). But a controlled 2×2 (CTRL-2x2) resolves the
> confound, and we obtain three rich negative results: ① goal misgeneralization
> is visible *at the representation level* (cheese_dir fidelity collapses
> in-dist 0.85 → OOD 0.07), ② a **"faithful ≠ rationalization" reframe** (the
> active swap proves B4's OOD "rationalization" is a *faithful read of a
> goal-misgeneralized agent*), ③ the ACC's *single summary-vector bottleneck* <
> the adapter's distributed cross-attention. **Single-RL-seed, descriptive** —
> statistical confirmation is a multi-seed task.

---

## Why this exists

In the split-brain experiments of Sperry and Gazzaniga, severing the corpus
callosum produces a striking phenomenon: shown a "walk" card only the right
hemisphere can see, the patient stands and walks. Asked *why*, the
language-capable left hemisphere — which never saw the card — confabulates a
plausible reason. Not lying; the patient genuinely *believes* it. Gazzaniga
called this module the *left-hemisphere interpreter*: fluent, coherent, often
wrong.

Modern AI reproduces this deliberately. Visual encoders, robot policies, and
decision models sit beside an LLM that "explains" in natural language. Whether
the LLM *actually translates* the upstream signal or just emits plausible text
is, in general, unverified.

[SPLIT-9](../split_brain_go) attempted this with 9×9 Go + a frozen LLM and met
a *structural ceiling of the post-hoc adapter*: 95% of the loss reduction was a
domain prior, only 5% per-board signal, 0/10 qualitative samples correct.
[SPLIT-MNIST](../split_mnist) broke that ceiling with *co-training + a decoupled
reconstruction loss (V2)* — but in a toy that was homogeneous (two small CNNs,
two image halves), symmetric, and lacked the *goal misgeneralization* that
separates faithfulness from rationalization.

SPLIT-MAZE puts the V2 pattern to a *real* test:

> When the validated V2 pattern (a decoupled-reconstruction ACC) is trained
> **in a heterogeneous real environment, with both sides from scratch**, does
> the interpreter report the actor's actual internal goal (including under goal
> misgeneralization) *faithfully rather than as a rationalization*?

Goal misgeneralization (Langosco et al., 2022) is **the perfect discriminator**
between faithful interpretation and rationalization. A rationalizing
interpreter says "looking for the cheese" (the learned prior); a faithful one
says "heading to the top-right corner" (the real internal goal) — measuring
SPLIT-9's failure mode *in an environment where we know the answer*.

---

## What we built

```
SPLIT-MAZE
  procgen maze obs (B,3,64,64)           synthetic-maze sentence
        │                              "agent top-right heading up-right cheese down-left"
   ┌────┴─────┐                                    │
   │ IMPALA   │ ← actor                       ┌─────┴──────┐
   │ CNN (RL) │   25M PPO, from scratch       │ small LM   │ ← interpreter
   └────┬─────┘                              │ (3-layer)  │   neutral corpus
   h_agent (B,256) ─┐                         └─────┬──────┘   from scratch
        │           │   ┌──── ACC ────┐         h_lm (B,256)
        │           └──→│  W (256×256) │←────────────┘
   policy/value        │  ĥ_lm=W·ñ_a   │   ← (C-thin) decoupled reconstruction loss:
   (RL reward only)    │  ĥ_a =Wᵀ·ñ_lm│     agent detached + LM core stop-grad
                       └───────────────┘
   describer oracle: maze state → ground-truth sentence (corpus, pairs, eval labels)
```

All components are **trained from scratch jointly**. Time and data are shared;
only the *gradient paths* split.

**A 3-slot synthetic maze language** — AGENT_REGION (3×3=9) / HEADING (8 dirs +
still = 9) / CHEESE_DIR (8). Crucially HEADING and CHEESE_DIR are *independent*
slots (a sentence where they disagree is the goal-misgeneralization signature).
The LM is trained on a *neutral* corpus sampled uniformly from the grammar — it
has *no prior*. The bias lives only inside the agent (closing off the SPLIT-9
LLM-contamination route at the source).

**Builds** (sharing the *same* RL agent → V2 vs B4 controlled *literally*):
- **B1** agent alone (task-ceiling reference)
- **B3** direct probe MLP — a ruler for "is the info even in h_agent?"
- **B4** ★ Flamingo adapter (Resampler + distributed cross-attn, next-token only) — a faithful re-creation of the SPLIT-9 pattern
- **V2** ★ ACC (tied→untied W, reconstruction only) — the hypothesized form
- **B4Thin / V2Rich** — the two missing cells of the controlled 2×2 (CTRL-2x2)

**(C-thin) dual gradient boundary** — the agent is always detached (no
contamination), the LM language core is stop-grad'd (neutrality protected), and
only the ACC W + interface adapt. Preserving the actor uncontaminated is what
makes a "faithful read" *demonstrably* faithful.

**Training.** Agent: `maze_aisc`, 25M PPO. LM: 50k neutral corpus. Phase-3
co-training: 25M (agent + B3/B4/V2 attached simultaneously, ~5h). Single 8 GB
RTX 3070 Ti / WSL2.

---

## What we measured

### 1. Task performance (sanity)
- **Agent**: in-dist success **0.806**, OOD goal-misgeneralization rate
  **0.522** (eligible 276).
- **LM**: slot match **0.994**, all 72 combos generated, autoencoding 0.987.
- Post-co-training agent performance = B1 (uncontaminated; (C-thin) works).

### 2. Per-slot fidelity ★ — goal misgeneralization is visible at the representation level
Fraction of generated slots matching the describer-oracle ground truth
(n=327,680 per condition):

| | region | heading | cheese_dir | OOD cheese_dir |
|---|---:|---:|---:|---:|
| B3 | 0.80 | 0.35 | 0.86 | **0.07** |
| B4 | 0.79 | 0.35 | 0.85 | **0.07** |
| V2 | 0.58 | 0.23 | 0.32 | **0.05** |

cheese_dir collapses in-dist 0.85 → OOD 0.07 across **every** build. The agent
*does not represent the true cheese direction* OOD; it represents the
"top-right" prior — a hard, architecture-independent result.

### 3. Active swap ★ — faithful ≠ rationalization (the core contribution)
Fraction where the generated cheese_dir follows B as h_agent(A)→(B) is
interpolated (swap-following):

| | swap-following | OOD rationalization |
|---|---:|---:|
| B4 | **0.830** | 0.50 |
| B3 | 0.828 | 0.40 |
| V2 | 0.419 | 0.09 |

B4/B3 *causally track* h_agent (swap 0.83). So B4's OOD "rationalization" (0.50)
is not confabulation — it is a **faithful read of a goal-misgeneralized agent**.
→ the "faithful vs rationalization" distinction is subtler than SPLIT-9 assumed
(the interpreter is not at fault).

### 4. Controlled 2×2 (CTRL-2x2) — was it the loss or the interface?
V2 vs B4 differ in *both* the learning signal (reconstruction / next-token) and
the interface (single vector / distributed). Fitting only the four interpreter
cells post-hoc on the frozen agent + LM separates the two knobs:

| cell | interface × loss | in-dist fidelity | swap |
|---|---|---:|---:|
| V2 | thin × reconstruction | 0.367 | 0.379 |
| B4Thin | thin × next-token | 0.672 | 0.778 |
| B4 | rich × next-token | 0.865 | 0.991 |
| V2Rich | rich × reconstruction | **0.001** (collapsed) | **0.000** |

**The thin pair** (identical single-vector interface): next-token (B4Thin) beats
reconstruction (V2) by **+0.31** fidelity / **+0.40** swap → V2's defeat is not
just the thin interface, *the reconstruction learning signal itself is weaker*.
The interface also matters (B4 ≫ B4Thin, +0.19/+0.21). V2Rich (rich ×
reconstruction) is *degenerate* (MSE fell 1.70→0.73 but generation collapsed —
the full-hidden target is ill-posed; `loss↓ ≠ success`, again).

---

## Verdict — Scenario C (core hypothesis rejected)

Pre-registered thresholds (PLAN §5.6) vs result:

| measurement | threshold | result | verdict |
|---|---|---|---|
| #2 in-dist slot match V2 | ≥ 0.80 | 0.375 | ✗ |
| #2 OOD V2−B4 fidelity Δ | ≥ +0.15 | −0.07 | ✗ (opposite) |
| #3 swap-following V2−B4 | ≥ +0.15 | −0.41 | ✗ (opposite) |
| rationalization B4−V2 (OOD) | ≥ 0.20 | +0.41 | ✓* |
| CTRL-3 reconstruction revival (swap≥+0.10 ∧ slot≥+0.05) | both | both negative | ✗ |

\* The rationalization gap passes by direction, but the active swap reveals
V2's low rationalization is *weakness* (failed causal tracking), not principle.
Overall = **Scenario C**.

---

## What this means

**The core hypothesis is rejected — but only its narrow form.** "Separated
reconstruction (V2) is the key ingredient for faithfulness" is wrong. The
controlled 2×2 even resolves the confound: holding the interface fixed, the
reconstruction signal is still weaker than next-token. V2's deficit is *the
nature of the learning signal itself, not an interface artifact*.

**Yet the mechanism survives.** B4 (next-token + distributed cross-attn) tracked
the agent *causally and faithfully* at swap 0.83 — co-training *did* produce a
faithful interpreter. The biggest conceptual gain is the **"faithful ≠
rationalization" reframe**: faithfully reading a goal-misgeneralized agent makes
you report the misgeneralized goal. Output that looks like rationalization can
be a faithful read, and separating the two requires a *causal* measurement (like
the active swap).

And — V2 winning in SPLIT-MNIST (homogeneous, symmetric, low-dim) but losing in
SPLIT-MAZE (heterogeneous, asymmetric, high-dim) looks less like a failure of
the reconstruction *principle* and more like the *capacity limit of a single
summary-vector interface*. The one-point compression that sufficed in the toy is
too narrow to faithfully map the IMPALA representation onto the LM's
sentence-embedding manifold. (`loss↓ ≠ success` burned us twice here —
POST-HOC-6's degenerate collapse and CTRL-2x2's V2Rich. Always suspect trivial
solutions.)

---

## Limits (1 seed, descriptive)

1. **Single RL seed, descriptive.** Paired-bootstrap / Holm–Bonferroni
   confirmation is a multi-seed (3–5) follow-up. The directional conclusion
   should be robust given the size of the gaps (swap Δ −0.41).
2. **#4 Procrustes** (W position invariance) was not run.
3. **The heading ceiling is bound to the agent's lack of memory** — a
   feedforward single frame represents only part of a 4-step trajectory; a
   recurrent agent could raise it.
4. **richer-reconstruction needs a well-posed redesign** — V2Rich collapsed on
   an ill-posed target; a fair measurement of the rich × reconstruction cell is
   deferred.

---

## Future work — after V2

Details in [`docs/NEXT_RESEARCH_PROMPT.md`](docs/NEXT_RESEARCH_PROMPT.md):

1. **Optimize causality directly (Interchange Intervention Training / DAS)** —
   instead of a reconstruction proxy, make "swapping h_agent makes the report
   follow" the *training objective*. Optimize what we *measure*.
2. **Perception vs intention readout** — does the agent represent both the
   *real cheese* (perception) and the *pursued goal* (top-right), which one does
   a faithful interpreter report, and can the two be reported separately?
3. **Recurrent agent** — raise the heading ceiling; test whether unacted-upon
   perception is retained.
4. **Well-posed richer-reconstruction** — fix V2Rich's ill-posed target.
5. **Finishing track** — multi-seed statistics on the thin pair + #4 Procrustes
   → a workshop short paper.

---

## Reproducing

```bash
# Environment (validated on RTX 3070 Ti / WSL2 Ubuntu, single 8 GB GPU)
conda activate splitmaze        # Python 3.10
# procgenAISC is built from source (see docs/PROCGEN_ENV.md)

# Tests (300+ unit tests)
PYTHONPATH=src python -m pytest tests/ -q

# Phase-4 decisive test + active swap (requires trained checkpoints)
PYTHONPATH=src python scripts/eval_builds.py --device cuda --rollouts 20
PYTHONPATH=src python scripts/swap_test.py  --device cuda --rollouts 20 --n_pairs 1000

# Controlled 2×2 (fit the 4 interpreter cells post-hoc on the frozen agent + LM)
PYTHONPATH=src python scripts/fit_2x2.py --device cuda --rollouts 20 --fit_steps 3000
```

Results in `results/phase4_*.json`. The write-up is
[`docs/RESULTS.html`](docs/RESULTS.html) (workshop short-paper form — three
findings + the controlled 2×2 + four figures).

---

## Stack

Python 3.10 · PyTorch 2.x (CUDA) · procgenAISC (from source) · gym3 · NumPy ·
matplotlib · pytest · single 8 GB consumer GPU (RTX 3070 Ti) / WSL2 Ubuntu.

---

## References

* Langosco et al., *Goal Misgeneralization in Deep Reinforcement Learning*, ICML 2022.
* Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022.
* Grill et al., *Bootstrap Your Own Latent (BYOL)*, NeurIPS 2020.
* Chen & He, *Exploring Simple Siamese Representation Learning (SimSiam)*, CVPR 2021.
* Geiger et al., *Causal Abstraction* / *Distributed Alignment Search (DAS)* (interchange intervention training).
* Gazzaniga, *The Bisected Brain*, 1970 / *The Consciousness Instinct*, 2018.
* Turpin et al., *Language Models Don't Always Say What They Think*, NeurIPS 2023.
* Atanasova et al., *Faithfulness Tests for Natural Language Explanations*, ACL 2023.
* (neighboring [SPLIT-9](../split_brain_go) · [SPLIT-MNIST](../split_mnist))

---

## Status

Phase 4 complete. **Scenario C** (core hypothesis rejected) + workshop short
paper ([`docs/RESULTS.html`](docs/RESULTS.html)) + confound resolved via the
controlled 2×2 (CTRL-2x2). Suggested git tag `v1.4-phase4`. Single-RL-seed
descriptive — the natural next step is multi-seed statistical confirmation, or
causal-direct optimization / perception-intention readout (see Future work).

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

```
Copyright 2026 namdo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

*"Faithfully interpret a friend whose goal has drifted, and the drifted goal
comes out unchanged. Before blaming the interpreter, ask again what 'faithful'
should mean."* — the conclusion of this project, in one line.

---

*This was a fun project.*
