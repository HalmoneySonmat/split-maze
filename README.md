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
>
> **후속 (Phase 5 — CCM): 다리가 자란다.** V2 기각 뒤, 학습된 번역기도 얇은 재구성도
> 아닌 — 두 망이 같은 장면에서 *함께 켜진 노드의 대응을 기록(기억)*만 해 에이전트→LM을
> 구동하는 **CCM(공동활성 기억 뇌량)** 을 시험했다. ① *기록만으로*(backprop 0) 활성 스왑
> **0.445 = B4의 52%**, 비결은 **"공통모드(평균) 제거"**(중심화·화이트닝 중 *하나면 충분*,
> 날것 Hebbian은 붕괴). ② 닫힌고리로 다리를 키우려다(step2) 실패(세게=task 붕괴, 살살=정체,
> `loss↓≠성공`). ③ 그러나 다리를 *가소성*으로 풀고(기억→씨앗) 두 뇌가 마중하면(**step3**:
> agent gentle + LM 한 블록 + W) 얇은 다리가 *학습 번역기 천장(~0.80)* 까지 자란다 — 공동적응
> 순수 이득 **+0.064±0.020 (5-seed 확정, 5/5 양성)**, task·언어 보존. 사용자 원래 비전("두 뇌가
> 동시에 적응하며 다리가 자란다")의 지지 증거 — **다른 결정자 뇌 3개로도 일반화**(GENERALIZES,
> 평균 +0.081, 3/3 양성). 단 해석자 LM은 공유라 결정자 뇌 한정이고, 효과는 뇌-의존(과제가 붕괴한
> 1개 뇌에선 사라짐 = "흔들면 일부는 안 버틴다").

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

## Phase 5 — CCM (공동활성 기억 뇌량): 다리가 자란다

V2가 기각된 뒤, *전혀 다른* 다리를 시험했다. 번역기를 *학습*하지 않고 — 두 망이 같은
장면을 받을 때 **함께 켜진 노드의 대응을 기록(기억)** 만 해서 그 기록으로 에이전트→LM을
구동한다. **CCM(Co-activation Callosal Memory).** 인터페이스·생성 경로는 B4Thin과
byte-identical하되, 다리 `W`는 gradient가 아니라 *닫힌형 통계*(또는 그것을 씨앗으로 한
가소성 학습)로 채운다. "학습이 아니라 기억"이 정체성. 생물 뇌량의 *활동의존 가소성*과
*억제 정규화* 문헌에서 출발했다.

**step1 — 기록만으로 절반.** frozen 에이전트+LM 위에서 245,760 페어로 `W`를 닫힌형 적합.
backprop 0회인데 **활성 스왑 0.445 = 학습 어댑터 B4의 52%.** 단 *날것* 공동활성(Hebbian
외적)은 degenerate(평균항 지배 → 상수 붕괴). 작동의 비결은 **"공통모드(평균) 제거"** —
깨끗한 2×2 ablation 결과, 중심화 *또는* 화이트닝 중 **하나만으로** 거의 완전 복원(둘은
중복 경로). → **"다리는 *무엇이 같이 켜졌나*가 아니라 *무엇이 다른가*를 기억해야 한다."**
(초기엔 "화이트닝이 결정적"이라 봤으나 ablation이 confound로 교정 — 정직 기록.)

**step2 — 닫힌고리(기록 W·에이전트만): 음성.** 두 뇌가 다리에 적응하며 *다리가 자라나*
봤다. 세게 밀면 task 붕괴(return 10→3.75), 안 깨질 만큼 살살 하면 다리 정체(swap −0.05).
양쪽 모두 미지지 — `loss↓ ≠ 성공`의 교과서 사례(학습 loss는 떨어져도 held-out 충실도는 안 오름).

**step3 — 다리를 *살아있게* 만들고 두 뇌가 마중: 해피엔드.** 둘을 바꿨다 — (i) `W`를
*가소성*(학습 파라미터)으로 풀되 기록값에서 warm-start("기억이 씨앗"), (ii) LM도 마중:
생성 경로 디코더 블록 `blocks[0]`만 작은 lr로 + **언어 KL 앵커**(frozen 참조)로 언어 보존.
단계적 — A1(두 뇌 고정·W만 = 번역기 천장 control) → A2(공동적응).

| 다리 | 활성 스왑 | 비고 |
|---|---:|---|
| 기록 기억 (step1) | 0.445 | 씨앗 |
| A1 plastic W (고정 뇌) | 0.683 | 기억→번역기 성장 (B4의 80%) |
| A1-long (W-budget control) | 0.725 | W를 더 줘도 plateau (< A2) |
| **A2 공동적응** | **0.784** | 수렴 번역기 천장(~0.78) 도달 |

A2는 **task(return 10→8.3)·언어(KL≈0) 보존**. **5-seed 절차 multi-seed로 확정**: 공동적응
순수 이득(A2 − W-budget control) = **+0.064 ± 0.020, 5/5 양성**(SEM≈0.009 → 노이즈 아님).
결정적으로, gradient가 0인 *기록* ridge조차 적응된 시스템에서 0.445→0.508로 올랐다 — W
학습으로 설명 불가한, **두 뇌의 표상이 서로 정렬되며 자랐다는 증거**. 사용자 원래 비전 —
"두 뇌가 동시에 다르게 학습하면서 자극을 기억하고, 다리가 자란다" — 의 *증거 기반 지지*.

**그리고 다른 결정자 뇌로도 일반화한다 (GENERALIZES).** *다른* RL 에이전트 3개(각 독립 25M
PPO; 해석자 LM 공유)에서 per-brain 효과 = swap(A2)−swap(A1-long) = **+0.137 / +0.004 /
+0.102, 평균 +0.081 ± 0.069, 3/3 양성**. 동결 기준(≥2/3 양성 AND mean≥+0.03) 둘 다 충족 →
"이 뇌에서 진짜"를 넘어 **결정자 뇌를 가로질러 재현**. 뇌-간 기록 ridge 평균 0.500>0.445(W-무관
정렬 유지). 단 **정직하게**: 효과는 *뇌-의존*이다 — brain 1·3은 control 위 분명한 잔차(+0.10~+0.14)지만
**brain 2는 사실상 null(+0.004)이며 동시에 과제가 붕괴**(return 10→7.66). brain 2의 raw A2(0.804)≈
A1-long(0.801)이라 그 "이득"은 거의 전부 W 추가학습 — **W-budget control이 가짜 양성을 정확히 흡수**했다.
(판정은 brain 2를 null로 봐도 2/3≥N−1로 유지되어 그 부호에 의존하지 않는다.) → "마중하면 다리가
자란다"는 *방향*은 뇌 간 재현되나, 크기는 뇌마다 다르고 **결정자 과제를 깨면 사라진다**. 해석자 LM은
공유라 일반화는 *결정자 뇌*까지 — 해석자 뇌 변동은 미시도(큐의 큐).

> Phase 5 한 줄: **다리는 자란다 — 두 뇌가 마중할 때. 효과는 modest(+0.06)하지만 5-seed 절차 +
> 3-뇌 일반화로 실재 확정.** step2의 음성(기록·한 방향)을 step3(가소성·양방향)가 정직하게 뒤집었고,
> 다른 결정자 뇌로도 재현된다 — 단 과제를 깨면 사라지므로 "흔들어봐도 버틴다"는 뇌-의존이다.

---

## 한계 (1 seed descriptive)

1. **단일 RL seed · descriptive.** paired-bootstrap·Holm–Bonferroni 통계 확정은
   multi-seed(3~5) 후속 과제. 단 핵심 격차가 커서(swap Δ−0.41) 방향 결론은 견고할 전망.
2. **#4 Procrustes**(W 위치 무관)는 미실행.
3. **heading 천장이 에이전트 메모리 부재에 묶임** — 피드포워드 단일 프레임이라 4스텝
   궤적을 부분만 표상. recurrent 에이전트면 천장이 오를 여지.
4. **richer-reconstruction은 well-posed 재설계 필요** — V2Rich는 ill-posed 타깃으로
   붕괴. rich×재구성 칸의 공정한 측정은 Deferred.
5. **CCM(Phase 5).** step1 양성(B4의 52%)·공통모드 제거는 1-seed descriptive. step3 공동적응
   (+0.064±0.020, 5/5)은 *5-seed 절차* multi-seed로 확정, **다른 RL 뇌 3개로의 일반화도
   GENERALIZES**(평균 +0.081, 3/3 양성). 단 **해석자 LM은 공유** → 일반화는 *결정자 뇌*까지이고
   *해석자 뇌 변동*은 미시도(큐의 큐). 효과는 뇌-의존(brain 2는 null+과제붕괴; control이 흡수).
   *완전 양방향*(처음부터 동시)도 미시도. 절차 seed 4 & brain 2에서 return이 가드(−20%)를 초과(−23%).

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
6. **CCM 완전 양방향(큐)** — step3는 단계적(A1→A2)·LM 한 블록 한정. 처음부터 agent+LM+W
   동시 학습(붕괴 위험 ↑, 단계적 성공이 비교 기준).
7. **CCM RL-seed 일반화 ✅ 완료** — step3 공동적응 효과를 다른 RL 뇌 3개에서 확인(GENERALIZES,
   평균 +0.081, 3/3). *다음 큐의 큐*: **해석자(LM) 뇌 변동** — 지금까지 결정자 뇌만 바꿨고
   해석자는 공유했다. "두 뇌가 둘 다 달라도 자라나"가 진짜 완성판.

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
│   ├── paired_collect.py    ← (h_agent, 문장) 페어 수집 + replay buffer
│   └── ccm.py               ← Phase 5 CCM (CoActAccumulator + W 사다리/ablation + plastic CCMBridge)
├── scripts/                 ← train_agent/lm/phase3, evaluate, eval_builds, swap_test,
│                              fit_2x2 · (Phase 5) ccm_sanity, fit_ccm, train_ccm_step2,
│                              train_ccm_step3, run_step3_multiseed.sh, agg_step3_seeds …
├── tests/                   ← 330+ 단위 테스트 (14 파일, test_ccm 포함)
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

Phase 4 (Scenario C) + **Phase 5 (CCM)** 완료. Phase 4: 핵심 가설(V2) 기각 + CTRL-2x2
confound 해소 + 워크숍 short-paper. **Phase 5 CCM**: 기록 다리 = B4의 52%(step1) · 공통모드
제거 메커니즘(ablation, step1 해석 교정) · 닫힌고리 음성(step2) · **다리가 자란다(step3):
공동적응 +0.064±0.020, 5-seed 절차 확정 + 다른 결정자 뇌 3개로 일반화(GENERALIZES, 평균
+0.081, 3/3)**. 모두 [`docs/RESULTS.html`](docs/RESULTS.html) §3.5에 반영. 결정자 뇌
일반화까지 확정 — 다음 자연스러운 단계는 **해석자(LM) 뇌 변동** 또는 CCM 완전 양방향(향후 연구 참조).

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

이 실험을 구축하고 실행하는 데 Claude를 사용했다. 정말 많은 도움이 되었다. 내 1%의 망상으로 Claude는 정말 멋진 일을 가능하게 했다.

*"다리는 학습이 아니라 기억에서 시작될 수 있고, 두 뇌가 마중하면 자란다 — 단, '자랐다'는
부풀린 승리가 아니라 흔들어봐도 버티는 것이어야 한다."* — Phase 5의 결론을 한 문장으로.

---

*재밌는 프로젝트였다.*
