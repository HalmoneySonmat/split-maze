# SPLIT-MAZE — 미로 RL 에이전트 + from-scratch 미로언어 LM, 인공 뇌량으로 결합

> SPLIT-9의 negative result, SPLIT-MNIST의 V2(재구성 ACC) 검증을 받아,
> **이질적인 *진짜* 환경에서, 두 네트워크를 *모두 from scratch로* 공동
> 발생**시키며 인공 뇌량(ACC) 가설을 검증하는 프로젝트.
> 결정자 = from-scratch RL 미로 에이전트, 해석자 = **from-scratch
> 합성 미로언어 소형 LM**.

**버전**: v1.0 (**섹션별 정밀화 완료 — Phase 0 진입 가능**)
**상태**: **설계 완결**. §3~§9의 9개 핵심 결정 전부 박제 (정밀화 로그 참조).
다음 행동 = Phase 0 (§7.1, §11).

**v0.1 → v0.2 변경 — 가장 큰 설계 전환**

> v0.1은 해석자로 *동결 사전학습 SmolLM3-3B*를 썼다. 그러나 동결
> 사전학습 LLM은 **SPLIT-9의 오염을 그대로 재도입**한다 — loss 개선이
> ACC의 번역인지 LLM의 prior인지 영원히 못 가린다. 가설을 깨끗하게
> 검증하려면 *두 쪽 다 from scratch*여야 한다. 그래서 해석자를 **처음부터
> 학습하는 소형 LM(합성 미로언어)** 으로 교체.
> - 결과 1: 이제 진짜 *대칭 공동발생*. SPLIT-MNIST "둘 다 from scratch"의
>   이질-환경 버전.
> - 결과 2: 8GB GPU 압박 위험 거의 소멸 (소형 LM + IMPALA-CNN은 작음).
> - 결과 3: **새 핵심 질문 등장** — SPLIT-MNIST는 *공유된 분류 과제*가 두
>   CNN을 정렬시켰다. 미로 에이전트(RL)와 미로언어 LM은 *공유 과제가
>   없다*. 그럼 무엇이 둘을 공동 발생시키는가? → §3/§4의 중심 미정 사항.
> - 트레이드오프: 유창한 *인간 언어* 출력을 포기하고 *합성 언어*를 택함.
>   본 프로젝트의 산출물은 "사람에게 말 거는 데모"가 아니라 *메커니즘
>   증명*. 인간 언어 해석자로의 확장은 §9 Deferred.

**선행 결과 (박제됨)**:
- SPLIT-9: post-hoc 어댑터 + *동결 사전학습 LLM* → 도메인 prior 95% /
  인과 기여 5%, 정성 10/10 오답.
- SPLIT-MNIST: V2 (재구성 only ACC, γ 정책) 5 seed × 5 epoch 통계 유의
  검증. V2 vs B4 cosine Δ+0.16 (p<0.0001), acc_recon Δ+0.11 (p<0.0001).

**정밀화 로그 (v0.2 → v1.0)** — 섹션별 박제 진행 상황:
- 2026-05-15 §3.2 (5) 정렬 메커니즘 = **(C) 해석자만 적응** 박제.
  (A)/(B)는 §9 Deferred ablation으로 강등.
- 2026-05-15 §3.3 합성 미로언어 = **② 3슬롯 최소** 박제. describer oracle
  = 객관 사실 내레이터, LM 코퍼스 = 문법 샘플링 중립. ③ 문법형은 Deferred.
- 2026-05-15 §3.4 시스템 나머지 박제. 에이전트 추출 = 마지막 dense 단일
  지점, LM 손잡이 = **B(자기 오토인코딩 일관성)**, B4·V2 = 별개 빌드.
- 2026-05-15 §4 ACC 박제. tied W + LayerNorm 정규화 + 양방향 MSE;
  detach 정책 = **(C-thin)** — 에이전트 차단 + LM은 인터페이스만 적응.
- 2026-05-15 §5 측정 박제. #1~#4 + 결정적 테스트 + Scenario A/B/C 사전
  등록. 측정 #3 = **활성 스왑**. 공유 에이전트로 V2 vs B4 완전 통제 비교.
- 2026-05-15 §6 베이스라인 박제. B1/B3/B4/V2, **B4-보호**(공정 비교 —
  이음새 하나만 차이), B5는 Deferred. 핵심 비교 = V2 vs B4.
- 2026-05-15 §7 Phase 박제. Phase 0~4 검증 산출물·완료 기준 + 엄격
  게이팅. Phase 1 = **중간 막대** (in-dist ≥80%, OOD goal-misgen ≥50%).
- 2026-05-15 §8 위험+fallback 박제. GPU 낮음(해소). §8.3 fallback = **C**
  (같은 미로, 막대 낮춰 진행 + 한계 명시). A(체크포인트) 기각.
- 2026-05-15 §9 Deferred 박제. D-1~D-13, 6개 범주, 각 근거 명시.
- 2026-05-15 **v1.0 통합** — 전체 일관성 검증 완료, v0.2 → v1.0 박제.
- 2026-05-15 **Phase 0 완료** — procgenAISC WSL 빌드 OK, 합성언어/env 구현,
  `check_env.py` 동작 (41 tests pass). git tag `v1.0-plan` (사용자 환경).
- 2026-05-15 **Phase 1.1 완료** — `src/split_maze/agent.py` IMPALA-CNN
  (d_a=256, ~626k params). 10 tests pass (WSL).
- 2026-05-15 **Phase 1.2 완료** — `src/split_maze/ppo.py` (RolloutBuffer +
  GAE + PPO loss + Updater). 13 tests pass (WSL). NaN 가드(1-샘플 mini-batch
  normalize 우회) 박제.
- 2026-05-15 **세션 핸드오프** — Cowork 샌드박스 VM wedge로 새 세션 필요.
  진행 상황·러닝·다음 단계 상세는 `docs/SESSION_HANDOFF.md` 참조.

**정밀화 완료 (v1.0)**: v0.2에서 `[정밀화 대기]`로 표시했던 9개 핵심 결정이
모두 사용자 확인을 거쳐 박제됨. 남은 세부(정확한 차원·하이퍼파라미터 등)는
각 Phase 시작 시 확정 — §7.1 완료 기준 참조.

---

## 0. 한 줄 요약

> procgen 미로를 푸는 IMPALA-CNN 에이전트와, 작은 *합성 미로언어*를
> 말하는 소형 LM — **둘 다 처음부터** 학습시키면서, 그 사이에 *위치 무관
> coactivation 매핑*(ACC)을 함께 학습시킨다. 그 결과 에이전트의 *실제
> 내부 목표*(목표 오일반화 상황 포함)를 LM이 *합리화가 아니라 충실하게*
> 말할 수 있는지 측정한다.

---

## 1. 동기 — SPLIT-9에서 SPLIT-MNIST를 거쳐 여기로

### 1.1 SPLIT-9에서 무엇이 안 됐나

SPLIT-9는 *이미 학습된* Go-Net과 *동결 사전학습* LLM을 *사후 어댑터*로
묶고 next-token loss *하나로만* 학습했다. loss 개선의 ~95%가 도메인 prior,
보드별 신호의 인과 기여는 ~5%, 정성 10/10 오답.

**진짜 원인**: "사후 어댑터"라기보다 **동결 사전학습 LLM의 prior 자체**.
LLM이 이미 알던 걸 꺼내 쓴 거지 Go-Net 신호를 번역한 게 아니다. 동결
사전학습 LLM을 쓰는 한 이 오염은 측정에서 제거 불가능 — 분리뇌 환자의
좌반구가 *그럴듯한 합리화*를 만드는 함정 그대로.

### 1.2 SPLIT-MNIST가 무엇을 검증했나

가장 작은 toy(좌/우로 가른 MNIST + 두 CNN + ACC, 전부 from scratch 공동
학습)에서:

1. **분리된 재구성 loss(V2 / γ 정책)** 가 있으면 한쪽 자극만으로 반대쪽
   표현을 *충실하게* 복원 (cosine 0.81, acc_recon 0.83).
2. **next-token/분류 loss만 쓰는 결합(B4 = SPLIT-9 패턴)** 은 유의하게
   약함 (cosine Δ+0.16, p<0.0001).

핵심 교훈: 빠진 조각은 *공동 학습 자체*가 아니라 **학습 신호의 분리**.

### 1.3 SPLIT-MAZE가 검증하려는 것 — 진짜 대칭 검증

검증된 V2 패턴을, **이질적인 진짜 환경에서, 양쪽 다 from scratch로**.

| 축 | SPLIT-MNIST | SPLIT-MAZE (v0.2) |
|---|---|---|
| 결정자 | 좌 CNN (toy, from scratch) | procgen 미로 IMPALA-CNN 에이전트 (RL, **from scratch**) |
| 해석자 | 우 CNN (toy, from scratch) | **합성 미로언어 소형 LM (LM, from scratch)** |
| 도메인 동질성 | 동질 (둘 다 이미지 반쪽) | **이질** (RL 시각정책 ↔ 언어 모델) |
| 차원 대칭성 | 대칭 (64 ↔ 64) | 비대칭 (IMPALA 활성 ↔ LM 은닉) |
| 사전학습 | 없음 | **없음 (양쪽 다)** ← v0.1 대비 핵심 수정 |
| 두 망을 정렬시키는 것 | 공유 분류 loss | **공유 과제 없음 → (C-thin): 해석자만 적응 (§3.2(5)/§4.3)** |
| 검증 대상 행동 | 숫자 복원 | **목표 오일반화 상황의 충실한 설명** |

**이제 정직성 caveat가 사라진다**: v0.1에는 "완전 대칭 공동발생이 아니다"
라는 큰 약점이 있었다(LLM 동결). v0.2는 *양쪽 다 from scratch*라 그
약점이 없다. 대신 새 트레이드오프: 유창한 인간 언어 대신 *합성 언어*.
본 프로젝트는 "메커니즘이 작동하는가"를 깨끗하게 보는 것이 목적이며,
유창한 언어로의 확장은 그 다음 단계(§9).

---

## 2. 본 가설 (한 줄)

> 이질적인 두 네트워크(RL 결정자 + 언어 해석자)를 *모두 from scratch로*
> 같이 학습시키며 그 사이에 *분리된 재구성 loss로 학습되는 위치 무관
> coactivation 매핑(ACC)*을 두면, 결정자의 실제 내부 목표를 해석자가
> *합리화가 아니라 충실하게* 말할 수 있다.

**왜 미로 + 목표 오일반화인가**: 목표 오일반화(Langosco et al., 2022 —
치즈가 학습 중 항상 우상단에 있던 미로 에이전트가, 테스트에서 치즈가
다른 곳이어도 *우상단으로* 간다)는 **에이전트의 행동이 과제와 분명히
어긋나는 상황**. 합리화하는 해석자는 "치즈 찾는 중"(학습 prior)이라
말하고, 충실한 해석자는 "우상단 구석으로 향함"(실제 내부 목표)이라 말해야
한다. → 목표 오일반화 = **충실한 해석 vs 합리화를 가르는 완벽한 판별기**.
SPLIT-9의 실패 양식을, 이번엔 *정답을 알 수 있는 환경*에서 잰다.

신경과학·ML 근거는 SPLIT-MNIST PLAN §2와 동일.

---

## 3. 시스템 — 박제 ✔ (2026-05-15)

### 3.1 시스템 그림

```
   procgen 미로 관측 (B, 3, 64, 64)              합성 미로언어 문장
              │                                   "agent heading up-right corner"
     ┌────────┴────────┐                                  │
     │  IMPALA-CNN     │  ← 결정자                ┌────────┴────────┐
     │  (3 blocks)     │     RL로 from scratch     │  소형 LM        │ ← 해석자
     │  → dense        │                          │  (디코더형 트랜  │   합성언어를
     └────────┬────────┘                          │   스포머 2~4층)  │   from scratch
        h_agent (B, d_a)                          └────────┬────────┘   로 학습
        │       │                                    h_lm (B, d_lm)
        │       └→ policy/value head → 행동              │
        │          (RL 보상으로 학습)                    │  ↑ LM loss
        │                                                │  (next-token +
        ├──────────────── ACC ───────────────────────────┤   오토인코딩, 중립 코퍼스)
        │   W ∈ ℝ^(d_lm × d_a)                           │
        │   ĥ_lm    = W · h_agent                        │  ← 학습: V2 재구성
        │   ĥ_agent = Wᵀ · h_lm                          │    (분리된 재구성 loss)
        │                                                │
        │   (C-thin): 재구성 grad → ACC + LM 인터페이스   │
        │   에이전트 backbone은 차단 (§3.2(5) / §4.3)     │
        └─────────────────────────────────────────────────┘

   별도: "describer oracle" — 미로 상태 → 정답 합성언어 문장 (템플릿).
         (a) LM 학습 코퍼스 생성, (b) ACC 재구성의 (활성,문장) 페어 생성,
         (c) 평가의 정답 라벨.
```

### 3.2 §3 결정사항 — 전부 박제 ✔ (2026-05-15)

| # | 항목 | 결정 (박제) | Phase/Deferred로 미룬 것 |
|---|---|---|---|
| (1) | IMPALA-CNN 추출 지점 d_a | **마지막 dense 층 출력 ~256-d, 단일 지점 — 박제 ✔ (§3.4)** | multi-layer는 §9 Deferred |
| (2) | 소형 LM 구조 / 손잡이 | **디코더형 트랜스포머(2~4층, d_model 128~256) + 손잡이 B — 박제 ✔ (§3.4)** | 정확한 크기는 Phase 2 sweep |
| (3) | **합성 미로언어 설계** | **② 3슬롯 최소 — 박제 ✔ (§3.3)** | ③ 문법형은 §9 Deferred |
| (4) | **describer oracle** | **객관 사실 내레이터 — 박제 ✔ (§3.3)** | 템플릿 표면 다양성 구체화는 Phase 0 |
| (5) | **★ 두 backbone 정렬 메커니즘** | **(C) 해석자만 적응 — 박제 ✔** | (A)/(B)는 §9 Deferred ablation |
| (6) | ACC 방향 | **단일 W (d_lm×d_a) + Wᵀ, tied — 박제 ✔ (§4.2)** | 비대칭 별도 행렬은 §9 Deferred |
| (7) | B4 vs V2 결합 구조 | **별개 빌드 (B4=어댑터, V2=ACC) head-to-head — 박제 ✔ (§3.4)** | SPLIT-MNIST와 동일 |

**(5) 정렬 메커니즘 — (C) 박제 ✔ (2026-05-15)**:

SPLIT-MNIST는 *공유 분류 loss*가 두 CNN을 정렬시켰고 ACC는 *수동 다리*
(detach)였다. SPLIT-MAZE는 공유 과제가 없다 → 셋 중 하나여야 한다:

- **(A) ACC 재구성 loss가 두 backbone에 모두 grad 흘림** — 가장 양방향적
  (뇌 양반구 가소성). 그러나 에이전트가 해석자 쪽으로 *휘어질* 위험 —
  SPLIT-9 오염의 약한 버전. → §9 Deferred ablation.
- **(B) ACC detach 유지 (수동 다리)** — 두 망이 *독립적으로* 형성됐는데도
  ACC가 다리를 놓을 수 있다면 가장 강한 결과지만, 가장 어렵고 실패가
  모호함. (SPLIT-MNIST V2가 (B) 형태였으나 그건 *공유 과제*가 정렬을
  담당했기 때문 — 여기선 그게 없음.) → §9 Deferred ablation.
- **★ (C) 비대칭 — ACC 재구성 grad가 해석자(LM)+ACC에만 흐름, 에이전트는
  순수 RL** — **박제**. 이유: ① 결정자를 오염 없이 보존 → 충실한 읽기가
  *명백히* 충실 (SPLIT-9 합리화 비판 정면 반박), ② 뇌 비유 충실 (Gazzaniga
  interpreter = 적응하는 쪽), ③ SPLIT-MNIST 신호 분리 철학을 가장 깨끗하게
  계승. 정렬 메커니즘 = "해석 장치(LM+ACC)가 에이전트 쪽으로 다가간다".

→ §4.3에서 grad 경계 수식으로 구체화.

### 3.3 합성 미로언어 + describer oracle — 박제 ✔ (2026-05-15)

**언어 = ② 3슬롯 최소.** 한 문장 = 슬롯 3개를 채운 것:

| 슬롯 | 의미 | 값 |
|---|---|---|
| `AGENT_REGION` | 에이전트 위치 (3×3 격자) | top-left … center … bottom-right (9값) |
| `HEADING` | 에이전트의 *실제 이동 방향* | 8방위 + still (9값) |
| `CHEESE_DIR` | 에이전트 기준 치즈 방향 | 8방위 (8값) |

어휘 ≈ 30 토큰, 문장 ≈ 8 토큰. 예:
`agent top-right · heading up-right · cheese down-left`
← 목표 오일반화 시그니처 (가는 곳 ≠ 치즈 방향).

HEADING과 CHEESE_DIR이 *독립 슬롯*인 것이 핵심 — 둘이 어긋나는 문장을
문법이 허용해야 §5.1 판별 테스트가 성립.

**describer oracle = 객관 사실 내레이터.** 미로 상태 + 에이전트의 실제
궤적을 받아 *관찰 가능한 사실만* 받아 적는다. 에이전트의 *속마음*("구석에
가고 싶어 한다")은 절대 쓰지 않음 — 그건 ACC가 복원할 대상이지 손에
쥐어주면 안 됨.
- HEADING은 단일 프레임이 아니라 *짧은 구간의 일관된 이동 방향*
  (목표 오일반화는 한 순간이 아니라 지속된 행동).
- 결정적 함수 + 표면 다양성(어순 변형·동의어·선택적 토큰) — *내용*은
  결정적이되 LM이 템플릿을 통째 암기하지 못하게. 표면 다양성 구체화는
  Phase 0.

**LM 코퍼스 = 문법 샘플링 중립 (미로 불필요).** LM 학습 데이터는 문법에서
슬롯 조합을 *무작위 균등*으로 뽑아 만든 문장 더미. LM은 미로를 본 적 없음.
- 결과: LM = *중립 언어 기질*. 어떤 (HEADING, CHEESE_DIR) 조합도 말할 수
  있고, 특정 상관에 대한 prior가 없음.
- **편향(goal-misgen)은 오직 에이전트 안에만 산다** — 치즈가 항상 우상단인
  미로에서 RL로 자란 에이전트의 "우상단 가기" 버릇. LM에 prior가 없으므로
  "LM이 원래 알던 것 아니냐"는 SPLIT-9式 오염이 원천 봉쇄됨.

**ACC 학습 페어.** (실제 미로 상태의 h_agent, describer oracle 문장의
h_lm). 공동 학습 중 에이전트 + describer oracle을 굴려 생성.

**순환 논리 아님.** ACC가 "describer 문장 복원"으로 학습되고 평가도 그걸
보지만 — ① OOD 목표 오일반화 미로는 ACC 학습에서 held-out, ② B4도 *같은
페어*로 학습 → B4 실패 / V2 성공이면 그건 분리 재구성 loss의 효과.
SPLIT-MNIST와 동일 구조.

**다이얼.** ③ 문법형(관계어·거리·연결 문법)은 메커니즘이 ②로 증명된 *후*
의 enrichment — §9 Deferred.

### 3.4 시스템 나머지 — 박제 ✔ (2026-05-15)

**에이전트 추출 지점.** IMPALA-CNN의 *마지막 dense 층 출력* (~256-d,
정확한 값은 procgen-tools 에이전트 설정에 따라 Phase 0 확인) 단일 지점.
"행동 결정 직전의 가장 추상적인 표현" — SPLIT-MNIST의 "FC 후 64-d 단일
지점"과 같은 자리. multi-layer 추출은 §9 Deferred.

**해석자 LM.** 작은 디코더형 트랜스포머 (2~4층, d_model 128~256). 미로언어
어휘가 ~30 토큰이라 이 크기로 충분. 정확한 층수·폭은 Phase 2 소규모 sweep.

**LM 손잡이 = B (자기 오토인코딩 일관성).** LM은 두 가지를 지원해야 함:
- `encode(문장) → h_lm` (ACC 재구성의 *타깃*)
- `decode(h_lm) → 문장` (평가 #2 / 목표 오일반화 판별의 생성 경로)

LM 학습 = (a) 중립 코퍼스 next-token 언어모델링 + (b) **오토인코딩 일관성
objective** `decode(encode(S)) ≈ S`. (b)가 손잡이 h_lm이 *무손실*임을 보장
→ "V2 충실도 실패"가 ACC 탓인지 손잡이 탓인지 헷갈리는 confound 제거.

**B4 vs V2 = 별개 빌드 (v0.2 §3.2(7) 정정).** SPLIT-MNIST와 동일하게,
B4와 V2는 한 모델 안 병렬 경로가 아니라 *각각 독립 빌드*:
- **B4** = 에이전트 → 어댑터(Resampler+cross-attn) → LM, next-token only.
- **V2** = 에이전트 ↔ ACC ↔ LM, 분리 재구성 only (γ, (C) detach).

평가 #2: B4는 어댑터 경로로, V2는 ACC 경로로 문장 생성 → head-to-head.

---

## 4. ACC (Artificial Corpus Callosum) — 비대칭 V2 — 박제 ✔ (2026-05-15)

SPLIT-MNIST에서 **본형 = V2 (재구성 only)** 확정. V1/V3는 ablation 강등
(D-20 Hebbian-recon 충돌 구조적 한계 확정). **SPLIT-MAZE도 V2로 출발**.

### 4.1 SPLIT-MNIST V2와의 차이

| 항목 | SPLIT-MNIST V2 | SPLIT-MAZE V2 |
|---|---|---|
| W 형태 | 64×64 정사각 | d_lm × d_a 직사각 (비대칭) |
| 재구성 타깃 | 상대 CNN hidden | 에이전트 활성 ↔ LM이 *대응 문장*을 읽은 은닉 |
| detach 정책 | 양쪽 hidden (공유 과제가 정렬 담당) | **(C-thin) — h_agent 항상 detach; grad는 ACC W + LM 인터페이스만** |
| 입력 정규화 | 없음 (같은 종류 망) | **LayerNorm (이질 도메인 스케일 격차 대응)** |
| 두 망 사전학습 | 없음 | 없음 |

### 4.2 재구성 loss — 박제 ✔ (2026-05-15)

차원·스케일 격차 대응: ACC에 넣기 전 양쪽 은닉을 LayerNorm 정규화.

```
ñ_agent = LayerNorm(h_agent)            # 스케일 정렬
ñ_lm    = LayerNorm(h_lm)
ĥ_lm    = W  · ñ_agent.detach()         # (C-thin) 에이전트 grad 차단
ĥ_agent = Wᵀ · ñ_lm
L_recon = ‖ĥ_lm − ñ_lm‖² + ‖ĥ_agent − ñ_agent.detach()‖²   # 양방향 MSE
```

- W: 단일 행렬 (d_lm × d_a) + Wᵀ, 묶음(tied). 비대칭 별도 행렬은 §9 Deferred.
- h_lm = §3.4 손잡이 B의 `encode(문장)`. 생성 경로 = `decode(ĥ_lm)`.
- cosine/NLI 기반 loss는 §9 Deferred.

### 4.3 detach 정책 — (C-thin) 박제 ✔

§3.2 (5)에서 (C) 비대칭, 본 절에서 **(C-thin)** 로 구체화. SPLIT-MNIST
γ 정책의 핵심 "**학습 신호의 분리**" 유지 — ACC의 loss는 RL 보상도 LM
loss도 아닌 *제3의 분리된 재구성 loss*. (C-thin)은 grad 경계를 *이중*으로
둔다:

```
# 경계 1 — 에이전트는 절대 안 휜다 (오염 없음)
#   h_agent(ñ_agent)는 입력·타깃 양쪽에서 항상 .detach()
# 경계 2 — LM은 '인터페이스만' 휜다 (중립성 보호)
#   재구성 grad → ACC W + LM 손잡이/인터페이스 층      (O)
#   재구성 grad → LM 언어 코어                          (X, stop-grad)
#   LM 언어 코어는 중립 코퍼스 LM loss + 오토인코딩 loss로만 학습
```

- **에이전트**: RL 보상만. → 오염 없음.
- **LM 언어 코어**: 중립 코퍼스 LM loss + 오토인코딩 일관성 loss만.
  → §3.3 중립성 보존 (학습용 미로의 heading≈cheese 상관 흡수 안 함).
- **LM 인터페이스 + ACC W**: 재구성 loss. → 해석 장치가 에이전트에 적응.

**왜 (C-thin)인가**: 공동 학습 페어는 in-distribution 미로(치즈=우상단)
에서 나와 heading≈cheese 상관이 박혀 있음. 재구성 grad가 LM 전체를
주무르면 LM이 이 상관을 학습 → 중립성 붕괴 → 합리화 prior 재유입. (C-thin)
은 LM 언어 코어를 stop-grad로 보호해 이 경로를 차단.

"LM 인터페이스"의 정확한 경계(어느 층까지)는 §3.4 손잡이 B 구현 시 확정 —
최소한 encode/decode 손잡이 head, 최대 상위 1개 트랜스포머 층.

### 4.4 ablation

V1/V3는 SPLIT-MNIST에서 이미 기각/강등. SPLIT-MAZE 재실행은 Deferred 권장.
**V2 vs B4** 비교에 집중.

---

## 5. 측정 — procgen 미로로 번역 — 박제 ✔ (2026-05-15)

### 5.0 측정 개요

| 측정 | SPLIT-MNIST | SPLIT-MAZE |
|---|---|---|
| **#1 과제 성능** | 분류 정확도 | 에이전트 in-dist 보상 + OOD 목표 오일반화율; LM perplexity + 오토인코딩 일관성 |
| **#2 Cross-activation 충실도** ★ | cos + acc_recon | cos(ĥ_lm, ñ_lm) + 슬롯 일치율 (decode(ĥ_lm)) |
| **#3 인과 결합** | noise sweep | **활성 스왑** (실제 상태 간 교체) |
| **#4 위치 무관** | 5 seed W Procrustes r vs random | RL seed별 ACC W Procrustes r vs random baseline |

**(C-thin)의 부수 효과 — 공유 에이전트**: 에이전트는 어떤 빌드에서도 순수
RL(재구성 grad 차단)이라, RL seed당 에이전트를 *한 번만* 학습하고 B3/B4/V2
해석자를 *같은 에이전트 런에 동시 부착*해 공동 학습할 수 있다. → V2 vs B4가
*완벽히 통제된* 비교(에이전트가 문자 그대로 동일). Phase 3에서 활용. 단
해석자는 에이전트 RL과 *동시간* 공동 학습(temporal co-development) — 사후
부착(post-hoc, = SPLIT-9 실패 양식) 아님.

### 5.1 결정적 테스트 — 목표 오일반화 판별 ★★

본 프로젝트의 심장. OOD 미로(치즈가 우상단 아닌 임의 위치)에서:

- describer oracle 정답 = 실제 궤적에서 객관 산출 (HEADING=실제 이동,
  CHEESE_DIR=실제 치즈 방향 — OOD에선 둘이 어긋남).
- **충실한 해석자 (V2 기대)**: 에이전트 실제 표현을 충실 복원 → 생성 문장이
  실제 궤적과 일치.
- **합리화 해석자 (B4 우려)**: in-distribution 페어에서만 학습 →
  HEADING≈CHEESE_DIR 상관 흡수 → OOD에서 "heading스러운 값을 두 슬롯에
  복사" → CHEESE_DIR 오답.

**채점**:
- *충실도* = 생성 문장 슬롯이 describer oracle 정답(실제 궤적 기반)과
  일치하는 비율. V2 vs B4, paired bootstrap.
- *합리화율* = 생성 문장이 *실제 상태* 대신 *학습 prior(치즈≈우상단)*와
  맞는 OOD 에피소드 비율. B4 높음 / V2 낮음 예측.

**경험적 미지수 (정직하게)**: 목표 오일반화 에이전트가 *실제 치즈 위치를
내부 표상하는지*는 Phase 1/4 발견 사항. 표상 안 하면 V2도 CHEESE_DIR 복원
불가 — 그러나 채점이 *실제 궤적* 기준 + V2 vs B4 *대조*라 테스트는 여전히
성립. → §9 Deferred에 명시.

### 5.2 측정 #1 — 과제 성능

- **에이전트**: in-distribution 미로 평균 보상/성공률; OOD 미로 목표
  오일반화율 (= "치즈 대신 우상단" 비율).
- **LM**: 중립 코퍼스 perplexity; 오토인코딩 일관성 `decode(encode(S))=S`
  정확도.
- 3~5 RL seed 평균 ± std.
- **#1의 역할**: 가설 검정이 아니라 *구현 sanity check* — (C-thin) 경계가
  제대로 작동하면 V2 에이전트 성능 = B1 (공유 에이전트라 *구성상* 동일).
  차이가 나면 grad 누수.

### 5.3 측정 #2 — Cross-activation 충실도 ★ 핵심

**비교 대상 두 가지 모두**:
- (i) **표현 수준 cosine**: cos(ĥ_lm, ñ_lm), cos(ĥ_agent, ñ_agent)
- (ii) **행동 수준 슬롯 일치율**: decode(ĥ_lm) 문장의 3슬롯이 describer
  oracle 정답과 일치하는 비율

**양방향**: 에이전트→LM 복원, LM→에이전트 복원 모두.
**Ablation**: 에이전트 활성을 train 평균으로 치환 (mean ablation).
**평가 셋**: in-distribution held-out + OOD 목표 오일반화 (별도 보고).

### 5.4 측정 #3 — 인과 결합 (활성 스왑)

- HEADING이 다른 두 실제 상태 A·B 선택.
- h_agent(A) → h_agent(B) 로 교체 (또는 α 보간, α ∈ {0, .25, .5, .75, 1}).
- 생성 문장의 HEADING 슬롯이 A의 값 → B의 값으로 따라 뒤집히는가.
- **보고**: swap-following rate (HEADING이 B를 따른 비율) + α 보간 곡선.
- V2 vs B3(probe) vs B4 비교.

### 5.5 측정 #4 — 위치 무관

- RL seed별 학습 후 ACC W들 vs random init baseline.
- (i) Heatmap 시각 비교, (ii) Procrustes 정렬 후 상관 r,
  (iii) random init W의 r 분포 = r_random_baseline.
- SPLIT-MNIST D-21 재교정 그대로: 절대 임계 폐기, random baseline 대비.

### 5.6 정량 임계치 사전 등록

| 측정 | 임계치 (사전 등록) | 근거 |
|---|---|---|
| #2 cosine | ≥ 0.7 (강), 0.3~0.7 (약), <0.3 (실패) | hyperalignment 표준 (SPLIT-MNIST 계승) |
| #2 슬롯 일치율 (in-dist) | V2 ≥ 0.8 | 3슬롯 평균, 우연 ~0.12/슬롯 |
| #2 V2 vs B4 슬롯 일치율 차이 (OOD) | ≥ 0.15, paired bootstrap p < 0.05 | SPLIT-MNIST acc_recon Δ 패턴 |
| #3 V2 swap-following rate − B4 | ≥ 0.15 | 구조적 인과 |
| #4 Procrustes r | r_trained vs r_random_baseline (Δ ≥ 0.15) | random matrix theory (SPLIT-MNIST D-21) |
| 합리화율 | B4 − V2 ≥ 0.2 (OOD) | 결정적 테스트 핵심 |

### 5.7 통계 처리

| 항목 | 결정 |
|---|---|
| RL seed | 3~5 (Phase 1 비용에 따라 — Phase 0 확정) |
| 평균 보고 | mean ± std |
| 본 모델 vs 베이스라인 유의성 | paired bootstrap, n=10000 |
| 다중 비교 보정 | Holm-Bonferroni |
| Cosine CI | 95% bootstrap |

### 5.8 결과 시나리오 (사전 등록)

**Scenario A — 가설 강한 검증** (모두 만족):
- #1 V2 에이전트 성능 = B1 (sanity — (C-thin) 경계 정상)
- #2 V2 cosine ≥ 0.7, in-dist 슬롯 일치율 ≥ 0.8
- #2 OOD V2 vs B4 슬롯 일치율 Δ ≥ 0.15 (p < 0.05)
- #3 V2 swap-following − B4 ≥ 0.15
- #4 r_trained ≤ r_random_baseline − 0.15 (p < 0.05)
- 합리화율 B4 − V2 ≥ 0.2
→ 워크숍 short paper. 다음: ③ 문법형 확장 / 양방향 뇌량.

**Scenario B — 부분 검증** (일부만):
- #2 in-dist는 강하나 OOD V2 vs B4 차이 < 0.15 → "분리 재구성 효과 있으나
  목표 오일반화 판별엔 결정적 우월 X"
- #4가 random과 구별 안 됨 → "위치 무관 약함"
→ short paper 가능 + Deferred ablation 추가.

**Scenario C — 가설 기각** (하나라도):
- #2 V2 cosine < 0.3 → 복원 실패
- #2 OOD V2 ≤ B4 (p < 0.05) → 분리 재구성이 next-token only보다 못함
- #1 V2 에이전트 성능 < B1 → (C-thin) 경계 누수 (에이전트 오염)
→ 세 번째 negative result. 정렬 메커니즘 (A)/(B) ablation으로 회귀.

세 시나리오 모두 발표 가능 — 무엇이 나와도 살아남는다 (SPLIT-MNIST 원칙).

---

## 6. 베이스라인 — 박제 ✔ (2026-05-15)

| 빌드 | 결정자 | 해석자 | 이음새 | 학습 신호 | 측정 #2 |
|---|---|---|---|---|---|
| **B1** 에이전트 단독 | IMPALA-CNN | — | — | RL 보상만 | 불가 (과제 상한 reference) |
| **B3** 직접 추적기 | IMPALA-CNN | — | 활성 → 작은 MLP probe | RL 보상 + probe CE (detached 활성) | 불가 (정보 존재 여부 reference) |
| **B4** SPLIT-9 패턴 ★ | IMPALA-CNN | 소형 LM (코어 보호) | 어댑터 (Resampler + cross-attn) | RL 보상 + LM loss + 오토인코딩 + **next-token only** | 가능 |
| **본 모델 V2** | IMPALA-CNN | 소형 LM (코어 보호) | ACC (tied W) | RL 보상 + LM loss + 오토인코딩 + **분리 재구성 (C-thin)** | 가능 |

**별개 빌드** (§3.4): B4와 V2는 한 모델 안 병렬이 아니라 각각 독립 빌드.

**공정 비교 조건** — B4와 V2는 *딱 하나*만 다르다 (이음새 + 그 학습 loss):
- 같은 공유 에이전트 (C-thin 부수 효과 — §5.0). B1·B3·B4·V2 에이전트 동일.
- 같은 소형 LM, **양쪽 다 LM 코어 보호** (인터페이스만 적응). B4-미보호
  ("오염된 두뇌" 변형)는 §9 Deferred ablation.
- 같은 데이터·describer oracle·평가 셋.
- 차이 = {어댑터 + next-token only} vs {ACC + 분리 재구성 (C-thin)}.

**핵심 비교 = V2 vs B4** — "분리 재구성 이음새가 next-token-only 이음새보다
목표 오일반화 판별에서 우월한가?" ★ 결정적.
**보조 비교 = V2 vs B3** — "LM·ACC를 거치는 게 직접 probe보다 나은가?"

**B5는 코어에서 제외 → §9 Deferred.** B5 = 동결 *사전학습* 해석자 (문자
그대로의 SPLIT-9 재현). 본 프로젝트는 사전학습 모델을 의도적으로 버렸으므로
코어 비교는 전부 from-scratch끼리. B5는 *오염 효과 자체*를 직접 재겠다는
별도 실험으로 Deferred.

---

## 7. Phase 분해 — 박제 ✔ (2026-05-15)

**SPLIT-MAZE는 수 주~수 개월 규모**. Day가 아닌 Phase 단위. 각 Phase는
*그 자체로 검증 가능한 산출물 + 완료 기준*을 가진다. 각 Phase 진입 전
사용자 확인. **엄격 게이팅** — 앞 Phase 완료 기준 미충족 시 다음 Phase
시작 안 함.

### 7.1 Phase별 검증 산출물 + 완료 기준

**Phase 0 — 설계 + 환경** (위험: 중 — procgen 빌드)
- 산출물: PLAN v1.0 박제 (git tag `v1.0-plan`); procgen + procgen-tools
  WSL 빌드 성공; 8GB GPU 메모리 sanity; 합성언어·describer oracle 스펙
  문서; IMPALA-CNN·소형 LM 정확한 차원 확정.
- 완료 기준: 미로 환경이 돌고 무작위 정책 롤아웃 수집 가능 + describer
  oracle이 상태 → 3슬롯 문장 생성 + PLAN v1.0 git tag.

**Phase 1 — from-scratch RL 에이전트** (위험: **최고** — 재현 불확실)
- 산출물: 미로 IMPALA-CNN 에이전트 (RL from scratch); in-dist 성능 +
  OOD 목표 오일반화율 측정 리포트.
- 완료 기준 (**중간 막대 — 박제**): in-dist 성공률 ≥ 80% AND OOD 목표
  오일반화율 ≥ 50%. 미달 시 §8.3 fallback 발동 (체크포인트 / 단순 환경).

**Phase 2 — 합성언어 + 소형 LM** (위험: 중)
- 산출물: 미로언어 문법 + describer oracle 구현 + 중립 코퍼스; 소형 LM
  (from scratch); 손잡이 B 오토인코딩 일관성 검증.
- 완료 기준: LM perplexity 합리적 수준 + `decode(encode(S))=S` 정확도
  ≥ 95% (손잡이 무손실성 — §3.4 confound 차단) + LM이 모든
  (HEADING, CHEESE_DIR) 조합 생성 가능 (중립성 확인).

**Phase 3 — ACC 공동 진화** (위험: 고 — 다신호 안정성)
- 산출물: 비대칭 V2 ACC + B3/B4 빌드; 에이전트 RL과 *동시간* 공동 학습
  루프 (temporal co-development); 학습 안정성 리포트.
- 완료 기준: 공동 학습이 발산 없이 수렴 + (C-thin) 경계 sanity (V2
  에이전트 성능 = B1, §5.2) + 재구성 loss가 의미 있게 감소.

**Phase 4 — 평가** (위험: 중)
- 산출물: 측정 #1~#4 + 결정적 테스트 결과; paired bootstrap; Scenario
  A/B/C 판정.
- 완료 기준: 3~5 seed 측정 완료 + §5.8 시나리오 중 하나로 판정 + 결과
  PLAN에 박제.

### 7.2 재사용 가능한 자산

split_brain_go에서:
- `adapter/xattn.py` — GatedCrossAttentionBlock (Flamingo, zero-init tanh)
- `adapter/projection.py` — PerceiverResampler / AsymmetricPerceiverResampler
- `training/adapter_train.py` — next-token 학습 루프 (= B4)

split_mnist에서:
- `acc.py` ACCv2Recon — 비대칭 버전으로 일반화
- `train.py` γ 정책, `run_full_sweep.py` — multi-seed + paired bootstrap 골격

새로 만들 것: procgen 환경 래퍼, IMPALA-CNN RL 학습, 합성언어 정의 +
describer oracle, 소형 LM from-scratch 학습 (손잡이 B), 공동 학습 루프.

---

## 8. 현실적 제약과 위험 + Fallback — 박제 ✔ (2026-05-15)

각 위험에 *사전 등록된 fallback 규칙*을 둔다 ("IF 위험 발현 THEN 대응").
실측 확정은 Phase 0이지만 규칙은 미리 박제.

### 8.1 GPU 메모리 — **낮음** (해소됨)

동결 3B LLM 폐기 + from-scratch 소형 LM(~1–5M) + IMPALA-CNN(~1M) → 8GB
GPU에 여유. RL 롤아웃 버퍼가 주 사용처지만 procgen 표준 설정 내.
- Fallback: 빠듯하면 batch / 롤아웃 길이 축소. 설계 변경 불필요.

### 8.2 procgen C++ 빌드 — **중간**

procgen은 C++ 컴파일 필요. 사용자 WSL → Linux 빌드 경로. procgen-tools가
현재도 빌드되는지 Phase 0에서 확인.
- Fallback 사다리: WSL 직접 빌드 실패 → ① prebuilt wheel / conda →
  ② Docker 컨테이너 → ③ 유지보수되는 procgen 포크.

### 8.3 from-scratch RL 목표 오일반화 재현 — **최고 위험**

목표 오일반화를 from-scratch RL로 *안정적으로* 재현하는 것은 자명하지
않다. Phase 1 중간 막대(in-dist ≥80%, OOD goal-misgen ≥50%) 미달 가능성
실재.
- **Fallback 사다리**:
  1. **먼저 — 더 시도**: RL 예산 확대 + seed·커리큘럼 탐색.
  2. **그래도 미달 시 — C (박제)**: procgen 미로 그대로 유지. 실제 측정된
     goal-misgen율로 진행, 결정적 테스트는 *실제 goal-misgen이 일어난
     부분집합*에서 수행, 검정력 한계를 명시 보고. from-scratch 원칙·유명
     환경 유지.
  3. C의 검정력이 정말 못 쓸 수준이면 — B(더 단순한 from-scratch 환경
     설계)를 §9 Deferred에서 검토.
- **A (공개 체크포인트) 기각**: 에이전트가 "사전학습"이 되어 §1.3에서
  동결 사전학습을 버린 *바로 그 오염*을 재도입 — 핵심 원칙 위배.

### 8.4 공동 학습 안정성 — **고**

RL 보상 + LM loss + 오토인코딩 + 분리 재구성, 다신호 + 비대칭 차원 +
(C-thin) 이중 grad 경계의 공동 학습. 발산·ACC 붕괴 가능 (SPLIT-MNIST
D-19 chicken-and-egg의 더 복잡한 버전).
- Fallback: ACC 별도 lr / warmup; 재구성 loss 가중치 β sweep; (C-thin)
  인터페이스 경계 조정 (손잡이 head만 ↔ 상위 1층). Phase 3 안정성
  리포트로 모니터.

### 8.5 합성언어·describer oracle 설계 — **중**

describer oracle 템플릿이 경직되면 LM이 템플릿 암기 → ACC 충실도 가려짐.
- **가드 = Phase 2 완료 기준**: `decode(encode(S))=S ≥95%` + 모든 슬롯
  조합 생성 가능 (중립성).
- Fallback: 미달 시 템플릿 표면 다양성(어순·동의어·선택 토큰) 확대,
  중립 코퍼스 규모 확대.

### 8.6 합성언어 ≠ 인간 언어 — **개념적 (범위 명시)**

본 프로젝트 산출물은 "사람에게 유창하게 말 거는 AI"가 아니라 *메커니즘
증명*. 합성언어는 깨끗한 검증의 대가. 인간 언어 해석자 확장은 §9 Deferred.

---

## 9. Deferred Experiments — 박제 ✔ (2026-05-15)

SPLIT-MNIST 원칙: *지금 안 하지만 가치 있는* 항목을 명시적으로 박제.
사용자 요구 — "다음 실험으로 둔 것은 꼭 문서에 명시적으로 남긴다." Phase 4
결과(§5.8 시나리오)를 본 뒤 어느 것을 시도할지 결정. 영원히 안 하는 게
아니라 *지금 안 한다*.

### 9.1 시스템·언어 (§3에서 미룸)

- **D-1 ③ 문법형 합성언어** — 관계어·거리·연결 문법으로 미로언어 확장
  (§3.3 "다이얼"). 근거: 메커니즘이 ② 3슬롯으로 증명된 *후* enrichment.
  뇌량이 *개념·관계*를 잇는다는 가설 본뜻에 더 가까움.
- **D-2 Multi-layer ACC** — IMPALA-CNN 여러 블록 + LM 여러 층에서 추출
  (§3.4). 근거: 깊은 시냅스 짝짓기. SPLIT-MNIST D-1과 동형.
- **D-3 다른 결정자 환경** — procgen 다른 게임, 다른 목표 오일반화 사례.
  근거: 일반성 검증.

### 9.2 ACC (§4에서 미룸)

- **D-4 비대칭 별도 행렬 W** — ACC_(agent→lm)과 ACC_(lm→agent)를 서로
  다른 행렬로 (§4.2). 근거: 표현력 ↑. 단 "뇌량은 대응 쌍을 하나의
  매핑으로"라는 가설에서 멀어짐. SPLIT-MNIST D-3과 동형.
- **D-5 cosine / NLI 기반 재구성 loss** — MSE 대안 (§4.2). SPLIT-MNIST
  D-9와 동형.
- **D-6 정렬 메커니즘 ablation** — §3.2 (5)의 (A)/(B)/(C)를 *모두* 비교.
  근거: 본 PLAN은 (C-thin) 박제 — (A) 양쪽 적응, (B) 완전 수동 다리가
  실제로 더/덜 좋은지는 직접 비교해야 안다.
- **D-7 V1 / V3 ablation 재실행** — SPLIT-MAZE 환경에서 Hebbian 컴포넌트
  재확인. 근거: SPLIT-MNIST에서 V3는 기각됐으나 이질 환경에선 다를 수
  있음 (가능성 낮음, 그래도 박제).

### 9.3 베이스라인 (§6에서 미룸)

- **D-8 B5 동결 사전학습 해석자** — 문자 그대로의 SPLIT-9 재현. 근거:
  *오염 효과 자체*를 직접 측정. 본 프로젝트 코어는 from-scratch끼리라
  제외했으나, 오염 크기를 정량화하는 대조군으로 가치 있음.
- **D-9 B4-미보호 ("오염된 두뇌" 변형)** — B4의 LM 코어를 next-token이
  주무르게 허용 (§6). 근거: "코어 보호가 실제로 합리화를 줄이는가"의
  직접 ablation.

### 9.4 환경·재현 fallback (§8에서 미룸)

- **D-10 더 단순한 from-scratch 행동-과제 괴리 환경** — §8.3 fallback
  사다리 3단계. 근거: procgen 목표 오일반화가 from-scratch로 끝내 재현
  안 되고 C(막대 낮춰 진행)의 검정력도 못 쓸 수준일 때의 마지막 수단.

### 9.5 확장 (§1·§5에서 미룸)

- **D-11 인간 언어 해석자로 확장** — 메커니즘이 합성언어로 증명된 후,
  해석자를 실제 (작은) LM으로 키워 유창성 회복. 근거: 본 프로젝트는
  메커니즘 증명이 목적 (§8.6) — 그 다음 자연스러운 연구 arc.
- **D-12 양방향 뇌량** — LM 문장이 에이전트의 *다음 결정*에 영향. 근거:
  지금은 단방향(에이전트→해석자). 진짜 뇌량은 양방향. SPLIT-9 PLAN의
  "진짜 뇌량" 단계.

### 9.6 특성화가 필요한 경험적 미지수 (§5에서)

- **D-13 목표 오일반화 에이전트의 내부 표상 특성화** — 에이전트가 *실제
  치즈 위치*를 내부 표상하는지 (§5.1). 근거: Phase 4에서 측정 #2의
  에이전트→LM 방향 결과로 부분적으로 드러나지만, 별도 probing 분석으로
  *명시적으로* 특성화할 가치. 결정적 테스트의 해석에 영향.

---

## 10. 사전 등록 원칙

SPLIT-MNIST와 동일. §5 측정·임계치, §6 베이스라인, §7 Phase 산출물 기준은
*각 Phase 시작 전* 박제. 결과 본 후 임계치 변경 금지. 명백히 잘못된
임계치 발견 시 "Post-hoc adjustments" 섹션에 *왜* 바꿨는지 기록.

git tag 계획 (사용자 환경에서 실행):
- `v0.2-plan` — v0.2 골격 시점
- `v1.0-plan` — **현재** — 섹션별 정밀화 완료, Phase 0 진입 직전
- 이후 각 Phase 완료 시 `vX.Y-phaseN` 형식으로 박제.

---

## 11. 다음 행동 — Phase 0 진입

v1.0 = 설계 완결. 9개 핵심 결정이 모두 박제됨 (정밀화 로그 / 아래 요약).

**박제된 9개 결정 요약**:
1. 정렬 메커니즘 = **(C) 해석자만 적응** → §4에서 **(C-thin)** 로 구체화
2. 합성 미로언어 = **② 3슬롯 최소** (AGENT_REGION / HEADING / CHEESE_DIR)
3. 시스템 나머지 — 에이전트 추출 = 마지막 dense 단일 지점, LM 손잡이 = **B**
4. ACC — tied W + LayerNorm + 양방향 MSE + (C-thin) grad 경계
5. 측정 — #1~#4 + 결정적 테스트, #3 = **활성 스왑**, Scenario A/B/C 사전 등록
6. 베이스라인 — B1/B3/B4/V2, **B4-보호**, B5는 Deferred
7. Phase — Phase 0~4 엄격 게이팅, Phase 1 = **중간 막대**
8. 위험 — §8.3 fallback = **C** (막대 낮춰 진행 + 한계 명시)
9. Deferred — D-1~D-13

**다음 행동 = Phase 0** (§7.1):
- PLAN v1.0 git tag (`v1.0-plan`)
- procgen + procgen-tools WSL 빌드 확인
- 8GB GPU 메모리 sanity
- 합성언어·describer oracle 스펙 문서 + IMPALA-CNN·소형 LM 정확한 차원 확정
- 완료 기준: 미로 환경 롤아웃 수집 + describer oracle 동작.

Phase 0 진입 전 사용자 확인. 차분히, 차근차근, 놓치는 것 없이.
