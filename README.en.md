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
>
> **Follow-up (Phase 5 — CCM): the bridge grows.** After V2 was rejected, we tried
> a bridge that is neither a *trained translator* nor thin reconstruction —
> **CCM (Co-activation Callosal Memory)**, which merely *records (remembers) which
> nodes co-fire* when both nets see the same scene, and drives agent→LM from that
> record. ① *Recording alone* (zero backprop) gives active swap **0.445 = 52% of
> B4**; the key ingredient is **common-mode (mean) removal** (centering *or*
> whitening alone suffices; raw Hebbian collapses). ② Trying to *grow* the bridge
> in a closed loop (step2) failed (push hard → task collapse, gently → it stalls;
> `loss↓ ≠ success`). ③ But make the bridge *plastic* (memory → seed) and let both
> brains meet it halfway (**step3**: agent gentle + one LM block + W) and the thin
> bridge grows to the *trained-translator ceiling (~0.80)* — pure co-adaptation
> gain **+0.064±0.020 (5-seed confirmed, 5/5 positive)**, task & language preserved.
> Evidence for the original vision ("two brains co-adapt and the bridge grows") —
> and it **also generalizes to 3 other decider brains** (GENERALIZES, mean +0.081,
> 3/3 positive). The interpreter LM is shared, so generalization is established only
> for the *decider* brain; and the effect is brain-dependent — it vanishes in the one
> brain whose task collapsed ("it doesn't all hold when you shake it").

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

## Phase 5 — CCM (Co-activation Callosal Memory): the bridge grows

After V2 was rejected, we tried a *completely different* bridge. Instead of
*training* a translator, we merely **record (remember) the correspondence of
nodes that co-fire** when both nets see the same scene, and drive agent→LM from
that record. **CCM (Co-activation Callosal Memory).** The interface and
generation path are byte-identical to B4Thin, but the bridge `W` is filled by a
*closed-form statistic* (or a plastic refinement seeded by it) rather than by
gradient. "Memory, not learning" is the identity. It starts from the biology of
the corpus callosum — *activity-dependent plasticity* and *inhibitory
normalization*.

**step1 — half the skill, from recording alone.** On the frozen agent + LM, fit
`W` in closed form over 245,760 pairs. With *zero* backprop, **active swap 0.445 =
52% of the trained adapter B4.** But *raw* co-activation (the Hebbian outer
product) is degenerate (the mean term dominates → constant collapse). The key
ingredient is **common-mode (mean) removal** — a clean 2×2 ablation shows
centering *or* whitening **alone** recovers almost all of it (the two are
redundant routes). → **"The bridge must remember *what is different*, not *what
co-fires*."** (We first thought "whitening is decisive"; the ablation corrected
that as a confound — recorded honestly.)

**step2 — closed loop (recorded W, agent only): negative.** We let the two brains
adapt to the bridge so it would *grow*. Push hard → task collapses (return
10→3.75); gently enough to survive → the bridge stalls (swap −0.05). Both
unsupported — a textbook `loss↓ ≠ success` (training loss falls, held-out
fidelity does not rise).

**step3 — make the bridge *live* and let both brains meet it: the happy ending.**
Two changes — (i) make `W` *plastic* (a trainable parameter) but warm-start it
from the recorded value ("memory is the seed"); (ii) let the LM meet halfway too:
unfreeze just one decoder block `blocks[0]` (on the bridge generate path) at a
small lr, with a **language KL anchor** (to a frozen reference) preserving
language. Staged — A1 (both brains frozen, W only = the trained-translator
ceiling / control) → A2 (co-adaptation).

| bridge | active swap | note |
|---|---:|---|
| recorded memory (step1) | 0.445 | the seed |
| A1 plastic W (frozen brains) | 0.683 | memory → translator (80% of B4) |
| A1-long (W-budget control) | 0.725 | plateaus even with more W (< A2) |
| **A2 co-adaptation** | **0.784** | reaches the converged translator ceiling (~0.78) |

A2 **preserves the task (return 10→8.3) and language (KL≈0)**. **Confirmed by a
5-seed procedure multi-seed**: the pure co-adaptation gain (A2 − W-budget control)
= **+0.064 ± 0.020, positive in 5/5 seeds** (SEM≈0.009 → not noise). Decisively,
even the *recorded* ridge (zero W-training) rose 0.445→0.508 on the adapted
system — a gain unexplainable by W-training, i.e. **evidence that the two brains'
representations genuinely grew more aligned**. This is *evidence-grounded support*
for the original vision — "two brains adapt simultaneously, remember the stimulus,
and the bridge grows." (Confirmed on the *same* frozen RL agent = "real for this
brain".)

**And it generalizes across decider brains (GENERALIZES).** We re-ran the step3
co-adaptation pipeline on **3 *different* decider brains** (each an independently
trained 25M-PPO RL agent; the interpreter LM is *shared* across them because it is
neutral). Per-brain effect = swap(A2) − swap(A1-long) = **+0.137 / +0.004 / +0.102,
mean +0.081 ± 0.069, positive in 3/3 brains**. The pre-registered *frozen* criterion
— GENERALIZES iff (≥ N−1 brains positive) AND (mean ≥ +0.03), i.e. for N=3: ≥2/3
positive AND mean≥+0.03 — is met on **both** counts → **verdict = GENERALIZES**. The
W-independent corroboration also holds across brains: the adapted *recorded* ridge
swap (mean 0.500) > the frozen baseline (0.445). But **honestly, the effect is
heterogeneous.** Brains 1 and 3 show a clear residual above the W-budget control
(+0.10 to +0.14), but **brain 2 is essentially null (+0.004) and its task collapsed**
(return 10 → 7.66, −23%). For brain 2 the raw A2 swap (0.804) ≈ the A1-long control
(0.801), so its apparent "gain" is almost entirely just extra W-training — **the
W-budget control correctly absorbed that false positive** (without the control it
would have looked like a confound). The verdict does *not* hinge on brain 2's
hairline-positive sign: even treating brain 2 as null, positives = 2/3 ≥ N−1, so it
still passes. → "the bridge grows when both brains meet halfway" replicates as a
*direction* across decider brains, but the *magnitude* is brain-dependent and the
effect *vanishes if the decider's task is broken*. This is the first empirical
evidence on the "does it hold when you shake it?" question — it holds in 2/3, not in
the brain whose task we (inadvertently) broke. **Scope:** N=3 is a probe (5 would be
sturdier); only the *decider* (RL-seed) was varied while the interpreter LM was
shared, so generalization is established only for the decider brain — varying the
*interpreter* brain (LM-seed) is the next queued item ("queue of the queue").
Artifacts: `results/phase5_ccm_brain_{a2,a1long}_s{1..3}.json`.

> Phase 5 in one line: **the bridge grows — when both brains meet it. The effect is
> modest (+0.06) but confirmed real by a 5-seed procedure *and* generalization to 3
> decider brains.** step3 (plastic, bidirectional) honestly reversed step2's negative
> (recorded, one-directional), and the effect replicates across decider brains — but
> it's brain-dependent, so "it holds when you shake it" is brain-dependent (it
> vanishes in the brain whose task collapsed).

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
5. **CCM (Phase 5).** step1's positive (52% of B4) and the common-mode finding are
   1-seed descriptive. step3's co-adaptation (+0.064±0.020, 5/5) is confirmed by a
   *5-seed procedure* multi-seed, and **RL-seed generalization is now done**:
   re-running step3 on 3 *different* RL brains **GENERALIZES** (mean +0.081, 3/3
   positive, pre-registered frozen criterion met). But the interpreter LM is *shared*
   across those brains → generalization is established only for the *decider* brain;
   *interpreter-brain variation* (LM-seed) is still to do ("queue of the queue"). The
   effect is also brain-dependent (brain 2 is null + its task collapsed; the W-budget
   control absorbed that false positive). *Full bidirectional* (all-at-once) is not
   done. In step-seed 4 and brain 2 the A2 return breached the guard (−23% vs −20%).

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
6. **CCM full bidirectional (queued)** — step3 is staged (A1→A2) and limited to
   one LM block; train agent+LM+W all at once from the start (higher collapse
   risk; the staged success is the baseline).
7. **CCM RL-seed generalization ✅ DONE** — the step3 co-adaptation gain was
   confirmed across 3 *different RL brains* (GENERALIZES, mean +0.081, 3/3 positive).
   *Next, queue of the queue*: **interpreter (LM-seed) brain variation** — so far only
   the decider was varied while the interpreter LM was shared; "both brains differ and
   the bridge still grows" is the real complete version.

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

Phase 4 (Scenario C) + **Phase 5 (CCM)** complete. Phase 4: core hypothesis (V2)
rejected + confound resolved via the controlled 2×2 + workshop short paper.
**Phase 5 CCM**: recorded bridge = 52% of B4 (step1) · common-mode-removal
mechanism (ablation, correcting step1's interpretation) · closed-loop negative
(step2) · **the bridge grows (step3): co-adaptation +0.064±0.020, confirmed by a
5-seed procedure + generalization to 3 decider brains (GENERALIZES, mean +0.081,
3/3)**. All reflected in [`docs/RESULTS.html`](docs/RESULTS.html) §3.5.
Decider-brain generalization is confirmed — the natural next step is
interpreter-brain (LM-seed) variation or CCM full bidirectional (see Future work).

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

Claude was used to build and run this experiment. It was a great help. Starting from my 1% far-fetched idea, Claude made something genuinely impressive possible.
