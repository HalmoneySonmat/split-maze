# SPLIT-MAZE

🇰🇷 한국어 · [🇬🇧 English](README.en.md) · [🇨🇳 中文](README.zh.md) · [🇯🇵 日本語](README.ja.md)

*procgen 미로를 푸는 IMPALA-CNN RL 에이전트와 from-scratch 합성 미로언어 LM을 인공 뇌량(ACC)으로 잇고, 목표 오일반화한 에이전트의 실제 내부 목표를 LM이 *합리화가 아니라 충실하게* 말할 수 있는지 — 둘 다 from scratch로 — 검증한 프로젝트.*

> **TL;DR.** 인접 프로젝트 [SPLIT-9](../split_brain_go)는 *동결 LLM + post-hoc
> 어댑터*가 충실성에서 구조적 천장에 부딪힌다는 negative result로 끝났고,
> [SPLIT-MNIST](../split_mnist)는 *공동 학습 + 분리된 재구성 ACC(V2)*가 그 천장을
> 뚫는다는 것을 동질적 toy에서 검증했다(Scenario A). 본 프로젝트는 그 V2 패턴을
> **이질적인 진짜 환경에서, 양쪽 다 from scratch로** 재검증한다 — procgen 미로
> RL 에이전트(결정자) + 합성 미로언어 소형 트랜스포머 LM(해석자), 그리고 *목표
> 오일반화*(치즈가 늘 우상단에 있던 미로에서 자란 에이전트가 OOD에선 치즈를 두고도
> 우상단으로 감)를 충실 vs 합리화의 판별기로. **핵심 가설(분리-재구성 V2 > next-token
> 어댑터 B4 충실도)은 기각됐다(Scenario C)** — V2가 *덜* 충실(per-slot 0.375 / 활성
> 스왑 0.42 vs B4 0.66 / 0.83). 그러나 통제 2×2(CTRL-2x2)로 confound까지 해소했고,
> 세 가지 풍부한 negative result를 얻었다: ① 목표 오일반화가 *표상 수준*에서 읽힌다
> (cheese_dir 충실도 in-dist 0.85 → OOD 0.07 붕괴), ② **"충실 ≠ 합리화" 재프레임**
> (활성 스왑이 B4의 OOD "합리화"가 *오일반화 에이전트를 충실히 읽은 것*임을 인과
> 증명), ③ ACC의 *단일 요약-벡터 병목* < 어댑터의 분산 cross-attn. **단일 RL seed의
> descriptive 결과** — 통계 확정은 multi-seed 과제.

---

## 왜 이걸 만들었나

Sperry와 Gazzaniga의 분리뇌 실험 — 좌·우반구를 잇는 corpus callosum을 절단한
환자에게 우반구만 보이는 자리에 "걸어가시오" 카드를 보여주면 환자는 일어나 걷는다.
"왜 걷냐?" 물으면 *진짜 이유*를 못 본 좌반구가 *그럴듯한 이야기*를 지어낸다. 거짓말이
아니라 진짜 *그렇다고 믿는다*. Gazzaniga가 *좌반구 통역사*라 부른 모듈 — 유창하고,
일관되고, 자주 틀린다.

현대 AI는 이 구조를 의도적으로 반복한다. 시각 인코더·로봇 정책·결정 모델 옆에 LLM을
붙여 "지금 무슨 일인지" 자연어로 설명하게 한다. 그런데 LLM이 *진짜로 상위 신호를
번역*하는지, 아니면 *그럴듯한 텍스트만 뱉는지*는 대체로 검증되지 않는다.

[SPLIT-9](../split_brain_go)는 이 질문을 9×9 바둑 + 동결 LLM으로 풀려다 *post-hoc
어댑터의 구조적 천장*을 만났다 — loss 개선의 95%가 도메인 prior, 5%만 보드별 신호,
정성 10/10 오답. [SPLIT-MNIST](../split_mnist)는 그 천장을 *공동 학습 + 분리된 재구성
loss(V2)*로 뚫었다 — 단, 동질적(둘 다 작은 CNN·이미지 절반)이고, 대칭적이며, 충실
vs 합리화를 가를 *목표 오일반화*가 없는 toy였다.

SPLIT-MAZE는 그 V2 패턴을 *진짜로* 시험한다:

> 검증된 V2 패턴(분리-재구성 ACC)이 **이질적인 진짜 환경에서, 양쪽 다 from
> scratch로** 학습될 때도, 결정자의 실제 내부 목표(목표 오일반화 포함)를 해석자가
> *합리화가 아니라 충실하게* 말하게 하는가?

목표 오일반화(Langosco et al., 2022)는 **충실한 해석 vs 합리화의 완벽한 판별기**다.
합리화하는 해석자는 "치즈 찾는 중"(학습 prior)이라 하고, 충실한 해석자는 "우상단으로
향함"(실제 내부 목표)이라 해야 한다 — *정답을 아는 환경에서* SPLIT-9의 실패 양식을 잰다.

---

## 무엇을 만들었나

```
SPLIT-MAZE
  procgen 미로 관측 (B,3,64,64)          합성 미로언어 문장
        │                              "agent top-right heading up-right cheese down-left"
   ┌────┴─────┐                                    │
   │ IMPALA   │ ← 결정자                      ┌─────┴──────┐
   │ CNN (RL) │   25M PPO, from scratch       │ 소형 LM    │ ← 해석자
   └────┬─────┘                              │ (디코더 3층)│   중립 코퍼스
   h_agent (B,256) ─┐                         └─────┬──────┘   from scratch
        │           │   ┌──── ACC ────┐         h_lm (B,256)
        │           └──→│  W (256×256) │←────────────┘
   policy/value        │  ĥ_lm=W·ñ_a   │   ← (C-thin) 분리 재구성 loss:
   (RL 보상만)          │  ĥ_a =Wᵀ·ñ_lm│     에이전트 detach + LM core stop-grad
                       └───────────────┘
   describer oracle: 미로 상태 → 정답 문장 (코퍼스·페어·평가 라벨)
```

모두 *from scratch 공동 학습*. 학습 *시간/데이터*는 같이, 학습 *gradient 경로*만 분리.

**3슬롯 합성 미로언어** — AGENT_REGION(3×3=9) / HEADING(8방위+still=9) / CHEESE_DIR(8).
HEADING과 CHEESE_DIR이 *독립 슬롯*인 것이 핵심(둘이 어긋나는 문장 = 목표 오일반화
시그니처). LM은 문법에서 균등 샘플링한 중립 코퍼스로 학습 — *prior가 없다*. 편향은
오직 에이전트 안에만 산다(SPLIT-9式 LLM 오염 원천 봉쇄).

**빌드** (같은 RL 에이전트 공유 → V2 vs B4 *문자 그대로* 통제):
- **B1** 에이전트 단독 (과제 상한 reference)
- **B3** 직접 probe MLP — "정보가 h_agent에 있긴 한가" 측정자
- **B4** ★ Flamingo 어댑터 (Resampler + 분산 cross-attn, next-token only) — SPLIT-9 패턴 충실 재현
- **V2** ★ ACC (tied→untied W, 분리 재구성 only) — 본 가설 본형
- **B4Thin / V2Rich** — 통제 2×2(CTRL-2x2)의 빠진 두 칸

**(C-thin) 이중 grad 경계** — 에이전트는 항상 detach(오염 없음), LM 언어 코어는
stop-grad(중립성 보호), ACC W와 인터페이스만 적응. 결정자를 오염 없이 보존해야
"충실한 읽기"가 *명백히* 충실하다.

**학습.** 에이전트 `maze_aisc` 25M PPO. LM 중립 코퍼스 50k. Phase 3 공동학습 25M
(에이전트 + B3/B4/V2 동시 부착, ~5h). RTX 3070 Ti 8GB 1대 / WSL2.

---

## 무엇을 측정했나

### 1. 과제 성능 (sanity)
- **에이전트**: in-dist 성공률 **0.806**, OOD 목표 오일반화율 **0.522**(eligible 276).
- **LM**: 슬롯 일치 **0.994**, 72조합 생성 1.0, 오토인코딩 0.987.
- 공동학습 후 에이전트 성능 = B1(미오염 확인, (C-thin) 정상 작동).

### 2. per-slot 충실도 ★ — 목표 오일반화가 표상 수준에서 읽힌다
생성 문장 슬롯이 describer oracle 정답과 일치하는 비율 (조건당 n=327,680):

| | region | heading | cheese_dir | OOD cheese_dir |
|---|---:|---:|---:|---:|
| B3 | 0.80 | 0.35 | 0.86 | **0.07** |
| B4 | 0.79 | 0.35 | 0.85 | **0.07** |
| V2 | 0.58 | 0.23 | 0.32 | **0.05** |

cheese_dir이 in-dist 0.85 → OOD 0.07로 **전 빌드 붕괴**. 에이전트가 OOD에서 *실제 치즈
방향을 표상하지 않고* "우상단" prior를 표상한다 — 아키텍처 무관한 단단한 결과.

### 3. 활성 스왑 ★ — 충실 ≠ 합리화 (핵심 기여)
h_agent(A)→(B) α-보간 시 생성 cheese_dir이 B를 따르는 비율(swap-following):

| | swap-following | OOD 합리화율 |
|---|---:|---:|
| B4 | **0.830** | 0.50 |
| B3 | 0.828 | 0.40 |
| V2 | 0.419 | 0.09 |

B4/B3가 h_agent를 *강하게 인과 추적*(swap 0.83). 따라서 B4의 OOD "합리화"(0.50)는
둘러대기가 아니라 **목표 오일반화한 에이전트를 충실히 읽은 결과**다. → "충실 vs
합리화" 구분이 SPLIT-9 가정보다 미묘하다(통역사 잘못이 아니다).

### 4. 통제 2×2 (CTRL-2x2) — 패배의 원인은 loss인가 인터페이스인가
V2 vs B4는 *학습 신호*(재구성/next-token)와 *인터페이스*(단일벡터/분산)가 동시에 다르다.
얼린 에이전트·LM에 통역사 4셀만 post-hoc 적합해 분리:

| cell | 인터페이스 × loss | in-dist 충실도 | swap |
|---|---|---:|---:|
| V2 | thin × 재구성 | 0.367 | 0.379 |
| B4Thin | thin × next-token | 0.672 | 0.778 |
| B4 | rich × next-token | 0.865 | 0.991 |
| V2Rich | rich × 재구성 | **0.001** (붕괴) | **0.000** |

**얇은 쌍**(동일 단일벡터 인터페이스): next-token(B4Thin)이 재구성(V2)을 충실도
**+0.31** / swap **+0.40**로 압도 → V2의 패배는 얇은 인터페이스 탓만이 아니라
*재구성 학습 신호 자체가 약하기 때문*. 인터페이스도 효과(B4 ≫ B4Thin, +0.19/+0.21).
V2Rich(rich×재구성)는 *degenerate*(loss↓ 1.70→0.73이나 생성 붕괴 — full-hidden 타깃
ill-posed; `loss↓ ≠ 성공` 재발).

---

## 시나리오 판정 — Scenario C (핵심 가설 기각)

사전 등록 임계(PLAN §5.6) vs 결과:

| 측정 | 임계 | 결과 | 판정 |
|---|---|---|---|
| #2 in-dist 슬롯 일치 V2 | ≥ 0.80 | 0.375 | ✗ |
| #2 OOD V2−B4 충실도 Δ | ≥ +0.15 | −0.07 | ✗ (반대) |
| #3 swap-following V2−B4 | ≥ +0.15 | −0.41 | ✗ (반대) |
| 합리화율 B4−V2 (OOD) | ≥ 0.20 | +0.41 | ✓* |
| CTRL-3 재구성 부활 (swap≥+0.10 ∧ slot≥+0.05) | 둘 다 | 둘 다 음수 | ✗ |

\* 합리화율은 방향대로 충족하나, 활성 스왑이 V2의 낮은 합리화가 *원칙*이 아니라
*약함*(인과 추적 실패)임을 드러냄. 종합 판정 = **Scenario C**.

---

## 이게 무슨 의미인가

**핵심 가설은 기각됐다 — 단, 좁은 형태만.** "분리-재구성(V2)이 충실도의 핵심
재료"는 틀렸다. 통제 2×2가 confound까지 해소했다: 인터페이스를 동일하게 맞춰도
재구성 신호가 next-token보다 약하다. 즉 V2의 열세는 *인터페이스 아티팩트가 아니라
학습 신호 자체의 성질*이다.

**그러나 메커니즘 자체는 산다.** B4(next-token + 분산 cross-attn)는 활성 스왑 0.83으로
에이전트를 *인과적으로 충실히* 추적했다 — 동시 학습으로 *충실한 통역사*가 실제로
나왔다. 가장 큰 개념적 수확은 **"충실 ≠ 합리화" 재프레임**이다: 목표가 오일반화된
에이전트를 충실히 읽으면 오일반화된 목표를 보고하게 된다. 합리화처럼 보이는 출력이
실은 충실한 reading일 수 있고, 둘을 가르려면 (활성 스왑 같은) *인과* 측정이 필요하다.

그리고 — SPLIT-MNIST(동질·대칭·저차원)에서 이긴 V2가 SPLIT-MAZE(이질·비대칭·고차원)
에서 진 것은, 분리-재구성 *원리*의 실패라기보다 *단일 요약-벡터 인터페이스의 용량
한계*로 보인다. toy에서 충분했던 한 점 압축이, IMPALA 표상 → LM 문장 임베딩 다양체를
충실히 얹기엔 좁다. (`loss↓ ≠ 성공`은 이 프로젝트에서 두 번 데었다 — POST-HOC-6의
degenerate collapse, CTRL-2x2의 V2Rich. trivial 해는 늘 의심할 것.)

---

## 한계 (1 seed descriptive)

1. **단일 RL seed · descriptive.** paired-bootstrap·Holm–Bonferroni 통계 확정은
   multi-seed(3~5) 후속 과제. 단 핵심 격차가 커서(swap Δ−0.41) 방향 결론은 견고할 전망.
2. **#4 Procrustes**(W 위치 무관)는 미실행.
3. **heading 천장이 에이전트 메모리 부재에 묶임** — 피드포워드 단일 프레임이라 4스텝
   궤적을 부분만 표상. recurrent 에이전트면 천장이 오를 여지.
4. **richer-reconstruction은 well-posed 재설계 필요** — V2Rich는 ill-posed 타깃으로
   붕괴. rich×재구성 칸의 공정한 측정은 Deferred.

---

## 향후 연구 — V2 이후

상세는 [`docs/NEXT_RESEARCH_PROMPT.md`](docs/NEXT_RESEARCH_PROMPT.md):

1. **인과를 직접 최적화 (Interchange Intervention Training / DAS)** — 재구성 proxy 대신
   "h_agent를 스왑하면 보고가 따라간다"를 *학습 목표*로. 우리가 *재는 것*을 *최적화*.
2. **지각 vs 의도 분리 readout** — 에이전트가 *실제 치즈*(지각)와 *추구 목표*(우상단)를
   둘 다 표상하는지, 충실한 통역사가 어느 쪽을 보고하는지 + 둘을 분리 보고 가능한지.
3. **recurrent 에이전트** — heading 천장 ↑, 행동 안 하는 지각의 보존 여부.
4. **well-posed richer-reconstruction** — V2Rich ill-posed 타깃 수선.
5. **마무리 트랙** — 얇은 쌍 multi-seed 통계 + #4 Procrustes → 워크숍 short-paper.

---

## 재현 방법

```bash
# 환경 (RTX 3070 Ti / WSL2 Ubuntu 검증, 8 GB GPU 1대)
conda activate splitmaze        # Python 3.10
# procgenAISC는 from-source 빌드 (docs/PROCGEN_ENV.md 참조)

# 테스트 (300+ 단위 테스트)
PYTHONPATH=src python -m pytest tests/ -q

# Phase 4 결정적 테스트 + 활성 스왑 (학습된 체크포인트 필요)
PYTHONPATH=src python scripts/eval_builds.py --device cuda --rollouts 20
PYTHONPATH=src python scripts/swap_test.py  --device cuda --rollouts 20 --n_pairs 1000

# 통제 2×2 (얼린 에이전트·LM에 4셀 post-hoc 적합)
PYTHONPATH=src python scripts/fit_2x2.py --device cuda --rollouts 20 --fit_steps 3000
```

결과는 `results/phase4_*.json`. 결과 문서는 [`docs/RESULTS.html`](docs/RESULTS.html)
(워크숍 short-paper 골격 — 세 발견 + 통제 2×2 + 그림 4장).

---

## 디렉터리

```
split_maze/
├── PLAN.md                  ← 사전 등록 설계 문서 (v1.0 + §10.1 정밀화 로그)
├── README.md                ← 본 문서 (한국어 메인)
├── README.en.md / .ja.md / .zh.md
├── LICENSE                  ← Apache 2.0
├── docs/
│   ├── PROCGEN_ENV.md       ← procgen 빌드·차원·환경
│   ├── LANGUAGE_SPEC.md     ← 합성 미로언어 + describer oracle
│   ├── SESSION_HANDOFF.md   ← 진행 상황·러닝·인벤토리
│   ├── RESULTS.html         ← 워크숍 short-paper (세 발견 + 통제 2×2)
│   └── NEXT_RESEARCH_PROMPT.md ← V2 이후 후속 연구 킥오프
├── src/split_maze/
│   ├── agent.py, ppo.py, train.py, train_phase3.py, evaluate.py
│   ├── env.py, language.py, lm.py, lm_train.py
│   ├── acc.py               ← ACC (tied→untied W, (C-thin), 양방향 MSE)
│   ├── adapter.py           ← Resampler + Gated cross-attn (Flamingo)
│   ├── builds.py            ← Build ABC + B3/B4/V2 + B4Thin/V2Rich
│   └── paired_collect.py    ← (h_agent, 문장) 페어 수집 + replay buffer
├── scripts/                 ← train_agent/lm/phase3, evaluate, eval_builds,
│                              swap_test, diagnose_v2, ceiling_v2, fit_2x2 …
├── tests/                   ← 300+ 단위 테스트 (13 파일)
├── checkpoints/             ← 학습 산출물 (gitignored, *.pt)
└── results/                 ← phase4_{builds,swap,ctrl2x2}.json 등
```

---

## 스택

Python 3.10 · PyTorch 2.x (CUDA) · procgenAISC (from source) · gym3 · NumPy ·
matplotlib · pytest · 8 GB 소비자용 GPU 1대 (RTX 3070 Ti) / WSL2 Ubuntu.

---

## 참고 문헌

* Langosco et al., *Goal Misgeneralization in Deep Reinforcement Learning*, ICML 2022.
* Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022.
* Grill et al., *Bootstrap Your Own Latent (BYOL)*, NeurIPS 2020.
* Chen & He, *Exploring Simple Siamese Representation Learning (SimSiam)*, CVPR 2021.
* Geiger et al., *Causal Abstraction* / *Distributed Alignment Search (DAS)* (interchange intervention training).
* Gazzaniga, *The Bisected Brain*, 1970 / *The Consciousness Instinct*, 2018.
* Turpin et al., *Language Models Don't Always Say What They Think*, NeurIPS 2023.
* Atanasova et al., *Faithfulness Tests for Natural Language Explanations*, ACL 2023.
* (인접 폴더 [SPLIT-9](../split_brain_go) · [SPLIT-MNIST](../split_mnist))

---

## 상태

Phase 4 완료. **Scenario C** (핵심 가설 기각) + 워크숍 short-paper 작성([`docs/RESULTS.html`](docs/RESULTS.html))
+ 통제 2×2(CTRL-2x2)로 confound 해소. 권장 git tag `v1.4-phase4`. 단일 RL seed
descriptive — 다음 자연스러운 단계는 multi-seed 통계 확정 또는 인과-직접 최적화/지각-의도
readout(향후 연구 참조).

---

## 라이선스

Apache License 2.0. [LICENSE](LICENSE) 참조.

```
Copyright 2026 namdo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

*"목표가 어긋난 친구를 충실히 통역하면, 어긋난 목표가 그대로 나온다. 통역사를 탓하기
전에, 무엇을 '충실'이라 부를지부터 다시 물어라."* — 본 프로젝트의 결론을 한 문장으로.

---

*재밌는 프로젝트였다.*
