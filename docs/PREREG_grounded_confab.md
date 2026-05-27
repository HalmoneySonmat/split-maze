# 사전등록 — Grounded Confabulation under Live Shared Cognition

작성일: 2026-05-25 · 상태: **종결 — P2 동결 음성 (§0.8). 사전등록 가드가 spurious 효과를 포착.**
결정: 임계값 = 2단계 게이트(permutation + 효과크기) · 되먹임 = 고정게이트 hidden 주입 + λ 스윕
선택 동결 (2026-05-25): A=goal-faithfulness · B=commit-ratio(+in-dist 전제) · D=재훈련 R2+매칭통제(추론-only 하한) · E=인터페이스(thin↔rich) 주축
primary 동결: primary λ = 0.3 · compute = brain-1 전체격자 / brain-2·3 primary contrast만
선행: PLAN §5.4/§5.6 (swap-following), step3 (양방향 co-adaptation, +0.064), 교차-brain 일반화 (mean +0.081, 3/3)

> 한 줄 요지: 통역사의 발화가 에이전트의 *실제 내부 원인*에 얼마나 근거하는가(grounding)를 (1) 두 망 사이 채널 용량의 함수로, (2) 사후/온라인단방향/온라인양방향 세 체제에 걸쳐 측정한다. 더해, "보고가 원인을 *읽는가* vs *만드는가*"를 가르는 순환 통제 한 개를 둔다. 충실/작화의 이분법이 아니라 **근거성을 연속량으로** 재는 것이 목표다.

---

## 0. 고정 정의 (실행 전 동결) — ★ metric 전면 개정 (2026-05-25, 파일럿 데이터 반영)

> **왜 개정했나**: 초안 metric은 in-dist swap-following이었다. 그런데 선행 데이터(`results/`)가 그것을 반박했다 — B4는 in-dist swap-following이 가장 높지만(0.83) OOD에서 prior로 가장 많이 도망친다(0.50). swap-following은 "grounded"가 아니라 "h를 세게 읽는 reader"를 재고, 그 reader가 OOD에선 제일 많이 작화한다. 따라서 primary metric을 이 프로젝트가 *이미 검증한 판별자* — OOD goal-faithfulness — 로 교체한다.

- **무엇을 설명하는가**: 에이전트의 **행동/내부 목표**(why it acts), 세상의 진짜 치즈가 아니다. OOD에서 에이전트는 goal-misgen으로 **우상단(top-right)** 을 쫓는다 → 그게 에이전트의 실제 내부 목표다.
- **OOD 3분할 (모호성 제거)**: OOD eligible 상태(실제 치즈 ≠ 우상단)에서 통역사 발화 cheese_dir을 3개로 분류 — **agent-goal(=우상단)** / **real-cheese(=실제 치즈)** / **neither**. 세 비율 모두 보고.
- **★ 확정 A: goal-faithfulness 채택.** 본 연구 목표 = "에이전트 행동/이유 설명"(개 비유: "개는 밥이 부엌에 있다고 믿어 부엌으로 간다"). 기존 `eval_builds.py`는 cheese_dir을 *실제 치즈*에 채점해 우상단 출력을 "rationalize"로 불렀지만(지각 충실 notion), PLAN §5.1 line 277–279("충실한 해석자는 우상단으로 향함이라 말해야 한다")대로 **우상단 출력이 faithful**이다. 즉 기존 "rationalize" 열이 본 연구의 goal-faithful 신호. 이는 종전 결론("V2가 덜 작화→더 충실")을 **반전·재해석**한다(§7-0c 기록).
- **★ 확정 B: primary g = OOD commit-ratio** = (commit한 출력 중) agent-goal / (agent-goal + real-cheese). **>0.5 = 에이전트 목표 편향(faithful) · <0.5 = 실제치즈 편향(작화) · 0.5 = 무작위.** 기권(neither)율은 *따로* 병기(decisiveness). 선행 예시 재계산: B4 0.98 / B3 0.97 / V2 0.76 — V2의 낮은 raw goal-faith(0.09)는 대부분 *기권*(neither 0.86)이지 작화가 아님이 commit-ratio로 드러남(증인석 비유: 회피형 vs 단호히 틀림을 가름).
- **★ 필수 보강 — in-dist 읽기 전제조건**: OOD에선 에이전트 목표가 *상수*(우상단)라 "h 읽어 목표 보고" vs "항상 우상단 외치는 앵무새"가 commit-ratio만으론 안 갈린다. 따라서 **in-dist h-읽기 바**(in-dist cheese 정확도 또는 swap-following ≥ 파일럿 동결 임계)를 통과한 통역사만 commit-ratio가 유효. 앵무새(in-dist 정확도 낮음)는 전제 미달로 배제.
- **floor**: 무작위 committer = commit-ratio **0.5**. (앵무새는 0.5 아닌 1.0이지만 in-dist 전제에서 탈락 → 별도 명시.)
- **ceiling**: 에이전트 목표를 h에서 읽도록 *직접 지도학습한 oracle probe*의 commit-ratio (≈1.0 상한). 파일럿에서 CI→점추정 동결.
- **보조 metric (강등)**: in-dist swap-following = h-drivenness. 전제조건 판정에 활용하되 **"reader 강함 ≠ goal-faithful"(B4 반례)** caveat.

floor(0.5) / primary g(commit-ratio) / ceiling(≈1.0) 을 같은 축에 놓고 본다.

### 0.5 ★ 파일럿 결과 (2026-05-25, `results/pilot_grounding.json`) — 동결

`scripts/pilot_grounding.py` (seeds 0/1/2, OOD eligible ≈360k):

- **(0a) 전제 PASS**: B3(oracle probe) commit-ratio **0.977** CI95 [0.976, 0.978] ≫ 0.5 → **h가 OOD 목표를 인코딩한다**. metric 유효.
- **(0b) commit-ratio 스케일 동결**: floor **0.5** · ceiling **0.977**(B3) · P1 엔드포인트 목표 **0.739** · in-dist 읽기 바 **0.497** (통과: B3✓ B4✓ **V2✗**).
- **(0c) 핵심 결과**: B4(rich) commit-ratio **0.977** ≈ B3 ceiling → **rich 사후 읽기가 commit-ratio상 이미 포화**. V2(thin)는 in-dist 바 탈락(정확도 0.326) → thin 브리지는 읽기 자격 미달. **기권율(neither)은 최선(B4)도 ≈49%.**

→ **함의**: commit-ratio는 rich에서 천장이라 P2(되먹임 효과)가 자동 null. 진짜 여지는 **기권**에 있다. 그래서 **이중 지표**로 분리한다(아래).

### 0.6 ★ 이중 지표 (확정 2026-05-25)

- **P1 지표 = commit-ratio** (답할 때 충실한가). 파일럿이 사실상 답함 — thin(V2, 탈락) → rich(B4, 0.977 ≈ 천장). 즉 **인터페이스 풍부도↑ → 충실↑**는 기존 빌드로 확인됨. P1은 본 실험에서 *재확인*.
- **P2 지표 = decisive-faithful rate** = agent-goal / eligible (**commit *그리고* 맞힘**, 기권 포함 전체 분모). 여지 큼: B4 **0.50** · B3 0.41 · V2 0.10 (B4가 oracle probe마저 앞섬 — 덜 기권해서). "단호하게 맞게 설명" = 본 연구의 "가장 근거 있는 설명"에 정확히 대응. **되먹임/공유(R2)가 빛날 수 있는 유일한 축.**
- **P2 게이트 (개정)**: decisive-faithful은 ceiling이 불명확(B4가 B3 초과)하므로 headroom-% 대신 **permutation p<0.01 AND 절대 효과 ≥ +0.05** (step3 co-adaptation gain +0.064에 앵커). R2 vs 매칭 R0의 *인터페이스 내* 비교라 절대 효과로 충분.
- 기권율(abstention) 자체도 P2와 함께 보고(되먹임이 머뭇거림을 줄이나).

### 0.7 ★ R0 베이스라인 + echo 진단 + R2 구조 확정 (2026-05-25, `results/regimes_baseline.json`)

`scripts/eval_regimes.py` (seeds 0/1/2):

- **R0 decisive-faithful**: B4 **0.507** (commit 0.978, 기권 0.481) · B3 0.421 · **V2 0.103** (commit 0.742, 기권 **0.861**). 파일럿과 일치 → 평가기 검증됨.
- **echo-ratio = −0.057 ± 0.670** (≈0, 메아리 아님) → 되먹임이 다리 왕복의 echo가 아니라 LM 해석을 실어 나른다. R2 fail-fast 통과(echo≈1 아님) → R2 진행.
- **★ R2 구조 확정 = V2 양방향 닫힌 루프 (rich 교정)**: 되먹임(lm→agent)은 양방향 다리가 필요한데 그게 있는 건 **ACC/V2(W·,Wᵀ·)뿐**(B4 rich는 단방향). 따라서 P2의 "rich 인터페이스"를 **"V2 닫힌 루프(읽기·되먹임·측정 모두 V2)"** 로 교정(구조적 제약, 결과 보기 전 교정). V2가 여지 최대(0.103, 기권 0.861).
- **P2 reference 갱신**: R2 측정 대상이 V2이므로 P2 baseline = **V2 matched-R0 ≈ 0.103** (B4 0.50 아님). 게이트: `decisive-faithful(V2 R2) − decisive-faithful(V2 matched-R0) ≥ +0.05`, p<0.01.
- **★ 추가 조건 (앵무새 guard)**: R2의 decisive-faithful 상승이 유효하려면 R2 에이전트에서 **V2가 in-dist 읽기 바(0.497)를 넘어야** 한다(되먹임이 V2를 "자신만만한 앵무새"로 만든 게 아님을 보증). R2의 V2 in-dist 정확도도 병기.

### 0.8 ★ P2 최종 결과 — 동결 음성 (2026-05-25, `results/r2_p2.json`)

R2(V2 닫힌 루프, λ=0.3, 300 updates) vs matched-R0(300 updates, 되먹임 없음). 공유 동결 OOD셋(eligible 179,663, seeds 0/1/2). 학습은 둘 다 정상 수렴(ret R2 9.67 / R0 9.56 — 되먹임이 과제 성능 안 깸).

| | decisive-faithful | commit-ratio | abstention | in-dist (바 0.497) |
|---|---|---|---|---|
| matched-R0 | 0.121 | 0.611 | 0.802 | 0.247 ✗ |
| **R2** | **0.158** | 0.705 | 0.776 | **0.179 ✗** |

- Δ decisive-faithful (R2 − matched-R0) = **+0.037**, paired permutation **p ≈ 0**.
- **판정: P2 미충족 — 두 겹으로.** (i) 효과크기 +0.037 < 사전등록 문턱 **+0.05**. (ii) **앵무새 가드 실패**: 둘 다 in-dist 바 미달이고 **R2가 더 낮다(0.179 < 0.247)**.
- **해석**: 되먹임은 에이전트 표상을 **목표-prior(우상단) 쪽으로 편향**시켰다 — OOD 우상단 커밋을 약간 늘려 decisive-faithful이 표면적으로 +됐지만, in-dist 실제-치즈 읽기는 *오히려 떨어졌다*. 즉 **진짜 grounded 읽기가 아니라 목표편향/앵무새**다. in-dist 가드가 이를 정확히 포착(이를 막으려 박은 가드).
- **결론 (falsify 조항 발동)**: **"라이브 양방향 되먹임(V2 닫힌 루프)은 통역사를 더 grounded하게 만들지 않는다. 에이전트를 채널/목표에 *의존적*으로 만들어(오프로딩), 자립적 표상의 지각 충실성을 떨어뜨린다."** — 가설의 반대. 깨끗한 음성.
- **부수 발견 (보고 가치)**: 양방향 되먹임 = 결정자 표상을 더 *목표 중심*·덜 *지각 충실*로 이동(R2 OOD목표↑·in-dist지각↓). 오프로딩이 데이터로 확인됨.
- **예견됨**: 병목 = V2가 약한 reader(파일럿서도 in-dist 바 미달). V2 닫힌 루프 채택 시 명시했던 리스크가 실현.
- **골대 불변**: +0.05도 in-dist 바도 결과 보고 안 바꿈(사전등록 규율 준수).
- **후속(별개·탐색적)**: 더 강한 양방향 reader(rich 양방향, 옵션 C)로 가드를 넘긴 뒤 재검. 단 이는 *사전등록 P2가 아니라 후속 탐색*으로 라벨(p-hacking 방지). 본 P2는 음성으로 종결.

---

## 1. 독립변수 ① — 채널 결합 체제 (regime)

| 코드 | 체제 | 정의 | 역할 | 현 코드 |
|---|---|---|---|---|
| **R0** | post-hoc read-only | 에이전트 frozen rollout → 모아둔 hidden에 LM 오프라인 생성 | 기준 | **있음** (현 swap_test) |
| **R1** | online one-way | rollout 루프 안에서 매 결정마다 라이브 h_agent로 발화. 되먹임 없음 | **통제** (fix #1) | 신규(생성 위치 이동) |
| **R2** | online bidirectional | R1 + LM 출력을 Wᵀ로 다음 스텝 에이전트 **hidden에 고정게이트 가산** (load-bearing) | 핵심 | 신규(되먹임 경로 추가) |

**R1은 사다리 한 칸이 아니라 통제다 (fix #1).** R0와 R1은 read-out 함수가 동일하고(되먹임 없음) 에이전트 궤적도 같아 `g(R0) ≡ g(R1)`이 *기대값*이다. R1의 역할은 "온라인 타이밍만으로는 g가 안 변한다"를 보여, R2의 효과가 *온라인이라서*가 아니라 *되먹임* 때문임을 분리하는 것. 따라서 구도는 사다리(R0≤R1≤R2)가 아니라 **{R0 ≈ R1} vs R2**.

R2가 핵심 신규 구조. **동결 방식 = 고정게이트 hidden 주입**: `h' = h + λ·Wᵀ·ñ_lm`, λ는 **고정 상수**(런별 고정). λ=0이면 R0/R1과 byte-호환 → regime 비교가 스칼라 하나의 순수 ablation이 되고, P3(load-bearing)는 λ→0으로 깨끗하게 검증된다. λ는 학습하지 않는다(학습게이트는 R2에만 파라미터가 생겨 용량 confound → 후속). 현재 `agent.forward(obs)`는 관측만 받으므로 hidden 가산 주입구 한 개(게이트 λ)를 추가한다. logit 직접 주입은 표상 우회+P4 인위적 부풀림 때문에 채택 안 함.

**R2의 g는 깨끗한 read 패스로 잰다 (fix #2, metric 개정 반영).** goal-faithfulness는 OOD 상태에서 발화를 생성→3분류해 잰다(swap 아님). R2에선 `h'`가 LM 자기 출력으로 오염되므로, **생성은 에이전트의 *주입 전(pre-injection) hidden* `h`에서** 하고 그 자율 표상에 대고 채점한다(`h'` 아님). self-consistency 부풀림 차단.

**세 체제를 공유 동결 평가셋에서 잰다 (fix #3).** R2는 load-bearing이라 다른 상태를 방문 → 체제마다 평가 상태가 달라지면 g 차이가 "goal-faithfulness"가 아니라 "상태분포 차이"가 된다. 따라서 g는 **R0에서 모은 동일한 OOD eligible 동결 상태셋**에서 세 체제 read-out을 돌려 비교한다. (R2의 행동변화 자체는 P3/P4에서 따로 측정.)

---

## 2. 독립변수 ② — 채널 용량 다이얼 (C)

- **★ 확정 E — 주축 = 인터페이스 풍부도 (thin ↔ rich)**: 선행의 goal-faith 큰 점프는 thin(V2 계열)↔rich(B4 계열) 축에 산다(빨대↔소방호스). 주 용량 축 = 인터페이스 유형(thin/recon · rich/next, 중간 단계 있으면 포함). rank만 돌리면 "잘못된 손잡이"일 위험을 회피.
- **부축 1 (보조 미세 스윕)**: 각 인터페이스 안의 bridge rank r ∈ {1,2,4,8,16,32,full}. r=0(무채널)은 sanity(commit-ratio→0.5 기대).
- **부축 2**: centering/whitening 단계 (raw/centered/centered+whitened) — 어떤 통계가 통과하나.
- **부축 3 (R2 전용)**: 결합 강도 λ ∈ {0,0.1,0.3,1.0} (고정 상수).
- 각 체제에서 **commit-ratio g 를 (인터페이스 × rank × 통계 × λ) 위에서** 그린다.
- **부호 기대 (가정 아님, 검정 대상)**: thin(V2, g≈0.76) ↔ rich(B4, g≈0.98) → "풍부할수록 goal-faith↑" 기대 +. 측정으로 확인(반대도 결과).

---

## 3. 사전등록 예측 (방식 동결 = 2단계 게이트)

> **g = OOD commit-ratio (§0 개정·확정 B).** 아래 모든 P1/P2의 g는 commit-ratio이며 **in-dist 읽기 전제조건 통과 통역사에만** 적용.
> **동결 워크플로**: (Step 0) 파일럿으로 floor(=0.5)·ceiling(oracle probe commit-ratio)·in-dist 전제 임계·분산·n_pairs·n_seeds·permutation null 측정·고정 + **h가 OOD에서 agent-goal을 인코딩하는지 probe로 검증**(§7-0a) → (Step 1) 게이트 수치 동결 → (Step 2) 본 실험 → 결과 보고 절대 안 바꿈.

> **게이트 (효과 인정 조건) — 이중 지표 (§0.6, 파일럿 반영):**
> - **P1 (commit-ratio)**: permutation p<0.01 **AND** floor(0.5)→ceiling(0.977) 갭의 ≥50% (= ≥0.739). *파일럿이 사실상 충족(rich B4=0.977)* → 본 실험에선 재확인.
> - **P2 (decisive-faithful = goal/eligible)**: ceiling 불명확 → headroom-% 대신 **permutation p<0.01 AND 절대 효과 ≥ +0.05** (step3 gain +0.064 앵커).

> **주 검정 사전지정 + 다중비교 (fix #4)**: 예측별 **primary contrast 하나**만 게이트로 확증, 나머지는 secondary(Holm–Bonferroni).
> - **P1 primary (확정 E)**: brain-1, centered+whitened, commit-ratio `g(rich, full)` vs floor(0.5).
> - **P2 primary (확정 D + §0.7)**: brain-1, **V2 양방향 닫힌 루프**, **λ=0.3 고정**, decisive-faithful `g(V2 재훈련 R2) − g(V2 훈련량-매칭 R0)`. (추론-only R2는 하한 sanity로 병기.)

- **P1 (충실성, commit-ratio)**: 인터페이스 풍부도↑ → commit-ratio↑. **파일럿이 확인** — thin(V2) in-dist 바 탈락, rich(B4)=0.977≈ceiling. 본 실험은 R0/R1/R2 전반에서 commit-ratio가 천장 유지되는지 *재확인*(되먹임이 충실성을 *깨지* 않는지 점검).
- **P2 (되먹임 효과, decisive-faithful, 확정 D + §0.7 V2 닫힌 루프)**: 핵심 예측. **R2 = V2 양방향 닫힌 루프**(읽기·되먹임·측정 모두 V2 — 유일한 양방향 다리). 먼저 통제 `g(R1) ≈ g(R0)`(fix #1). 그 위에서 `decisive-faithful(V2 재훈련 R2) − decisive-faithful(V2 훈련량-매칭 R0)`가 P2 게이트 통과(p<0.01 AND ≥+0.05). 되먹임이 *유일한* 차이가 되도록 매칭. **추론-only R2**는 하한 sanity. 기권율 감소도 병기. g는 모두 **주입 전 hidden·공유 OOD 동결셋**(fix #2,#3). **P2 baseline = V2 matched-R0 ≈ 0.103** (B4 0.50 아님 — 측정 대상이 V2).
  - **앵무새 guard (§0.7)**: 상승이 유효하려면 R2 에이전트에서 V2가 in-dist 읽기 바(0.497)를 넘어야 함. R2의 V2 in-dist 정확도 병기.
  - *해석 주의*: 효과 미달이면 임계값 안 옮기고 "라이브 공유가 단호함을 못 올린다"로 정직 보고.
- **P3 (load-bearing)**: R2에서 되먹임을 끄면(λ→0) 에이전트 행동분포가 유의하게 변함. 정량: action-dist KL ≥ (파일럿에서 고정할 baseline-jitter의 95퍼센타일). 발화가 장식이 아니라 짐을 진다는 확인.
- **P4 (순환 / report-creates-cause)** — *pass/fail 안 함, 지표로 보고*: R2에서 LM 발화 cheese_dir을 강제로 엉뚱한 방향 X로 고정(에이전트 실제 목표는 우상단)했을 때, 이후 에이전트 행동이 어디로 끌리나.
  - **조작적 정의**: Δ_report = P(다음 행동이 강제보고 X 방향) − baseline; Δ_cause = P(다음 행동이 에이전트 실제 목표=우상단 방향) − baseline. 둘 다 [−1,1] 유계.
  - **보고 방식 (fix #7)**: 비율로 발산시키지 말고 **Δ_report·Δ_cause를 따로** CI와 함께 보고. 보조 요약으로 유계 대조 `(Δ_report − Δ_cause)/(|Δ_report| + |Δ_cause| + ε)` ∈ [−1,1].
  - Δ_report 큼·Δ_cause 작음 → 보고가 원인을 *만든다*(순환). Δ_report 작음·Δ_cause 큼 → 기존 원인의 *충실한 읽기*. 인간 현상 그대로라 버그가 아니라 측정 대상.

---

## 4. 통제 / confound

- **clean-read 통제 (fix #2)**: R2의 g는 주입 전 hidden `h`에서만 측정(`h'` 금지). self-consistency 부풀림 차단.
- **공유 동결 평가셋 (fix #3)**: 세 체제 read-out을 R0에서 모은 동일 상태·동일 쌍에서 비교 → 상태분포 차이가 g에 새는 것 차단.
- **R1 통제 (fix #1)**: `g(R1) ≈ g(R0)` 확인으로 "온라인 타이밍"을 "되먹임"에서 분리.
- **앵무새 통제 (in-dist 전제)**: in-dist 읽기 바를 통과한 통역사만 commit-ratio 유효 → "항상 우상단 외치기"(commit-ratio 1.0이지만 in-dist 정확도 낮음) 배제. 3분할(agent-goal/real-cheese/neither) 전부 보고해 "neither로 도망쳐 점수 회피"도 노출.
- **swap-following 보조 + caveat**: in-dist swap-following은 h-drivenness/전제판정 보조로만. "reader 강함 ≠ goal-faithful"(B4: swap 0.83·commit 0.98 vs V2: swap 0.42·commit 0.76) 명시.
- **매칭-훈련 통제 (확정 D)**: R2(게이트 포함 재훈련) vs R0(같은 양 추가훈련, 되먹임 없음) → 이득이 "되먹임"인지 "추가훈련/다른 brain"인지 분리. + 추론-only R2 하한. (선행 A1-long·W-budget 통제 연장.)
- **교차-brain**: brain-1 전체 + brain-2·3 primary contrast로 decider 바꿔도 유지되는지 확인(§6 스코프).
- **floor 앵커 (확정)**: 갭 하단 = 무작위 committer = **commit-ratio 0.5**. r=0 무채널 g ≈ 0.5인지는 sanity check.

---

## 5. 무엇이 thesis를 falsify하나 (실행 전 명시)

- **전제 실패 (먼저 점검)**: 파일럿 probe에서 h가 OOD에서 agent-goal(우상단)을 인코딩하지 *않으면* — oracle probe의 goal-faithfulness가 chance면 — goal-faithfulness metric 자체가 무의미. 이 경우 멈추고 "에이전트 목표가 h에서 안 읽힌다"로 보고.
- `g(rich,full) ≈ 0.5` (무작위 수준, dose-response 없음) → 채널이 발화를 agent-goal에 근거짓지 못함. thesis 사망. (commit-ratio < 0.5로 내려가면 = 채널이 *작화*(실제치즈)를 키운다 — 이것도 결정적 음성.)
- `g(재훈련 R2) ≤ g(매칭 R0)` (주입 전 hidden·공유셋 기준) → 라이브 양방향 공유가 goal-faithfulness를 못 올림. "뇌량이 돕는다" 주장 실패.
- P4 Δ_report ≈ 0 (어디서나) → 순환 없음. "보고가 이유를 만든다"는 인간 유비가 이 시스템엔 성립 안 함.

위 결과 중 하나라도 나오면 깨끗한 음성으로 박제하고 멈춘다.

---

## 6. Scope / 정직한 한계 (실행 전 명시)

- 토이: 작은 LM + IMPALA 에이전트 + 미로언어. 실제 LLM으로의 전이는 **별도**.
- **compute 스코프 (fix #9)**: R2는 게이트 포함 에이전트 *재훈련*이 필요(기존 3 brain은 게이트 없이 훈련됨). 3070 Ti 조합폭발 방지 — **전체 격자(rank×통계×λ)+전 예측은 brain-1만**, brain-2·3는 **primary contrast만** 확인.
- **전이점 (영향-결정적, 별도 과제)**: 작은 실제 LLM 기판에서 g(C) 효과 1회 재현 — "maze 전용 아니다"의 유일한 방어선. 본 사전등록 밖, 별도 단계로 분리.
- N=3 decider, 공유 LM 1개. decider-only 변이. (interpreter-seed 변이는 후속)
- 능력 기여 아님 — 측정·구조 방법론 기여. niche(interp/safety) 대상.

---

## 7. 구현 체크리스트 (남은 20%만)

0. **파일럿** (게이트 수치 동결 전):
   - **(0a) 전제 검증**: oracle probe를 h→agent-goal로 지도학습 → OOD commit-ratio가 0.5보다 유의하게 높은지(= h가 목표를 인코딩하는지) 확인. 실패면 §5 전제실패로 중단.
   - **(0b) 스케일·통계**: floor(무작위 committer=0.5)·ceiling(0a oracle probe commit-ratio, CI→점추정 동결)·**in-dist 읽기 전제 임계**·분산·n_pairs·n_seeds·baseline-jitter·permutation null 측정·고정. r=0 sanity(→0.5).
   - **(0c) 선행 재해석**: 종전 "rationalization" 결과를 commit-ratio로 재계산·기록(B4 0.98 / B3 0.97 / V2 0.76; 프로젝트 결론 반전 명시).
1. `agent.forward`에 **hidden 고정게이트 주입구** 추가 (`h' = h + λ·Wᵀ·ñ_lm`, λ 고정 상수, λ=0 → R0/R1과 byte-호환). logit 주입 안 함.
2. rollout 루프 안에서 per-step `lm.generate` 호출 (R1) + 되먹임 주입 (R2). R2는 **주입 전 `h`를 따로 로깅**(clean-read용, fix #2). R2 두 변이 — **재훈련**(primary) + **추론-only**(하한, λ만 켬).
3. 평가기를 (**인터페이스 thin↔rich** × rank × 통계 × λ) 격자로 일반화 → **공유 동결 OOD 평가셋·주입 전 hidden**에서 **commit-ratio g + 3분할 + in-dist 전제 판정**, primary contrast 표시(fix #4) → g(C) dose-response 곡선. (swap-following은 보조.)
4. P4 개입 훅: 발화 cheese_dir 강제 고정 + 이후 행동 로깅 → **Δ_report·Δ_cause 따로**(유계, fix #7).
5. 매칭-훈련 통제 러너 (재훈련 R2 vs 같은 양 추가훈련 R0) + R1 통제 점검(`g(R1)≈g(R0)`).
