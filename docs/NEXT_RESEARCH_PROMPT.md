# SPLIT-MAZE 후속 연구 킥오프 — "V2 이후, 충실한 통역사를 어떻게 만들 것인가"

> 이 파일은 *새 세션을 여는 프롬프트*다. 새 세션 첫 메시지로 이 내용을 그대로 주거나,
> 이 파일을 가리키면 된다. 목적: V2(분리-재구성)가 기각된 뒤, **V2가 아닌 다른 방식**으로
> 후속 연구를 설계·시작한다.

---

## [먼저 읽고 "한 줄 현재 위치"부터 보고]

1. `docs/SESSION_HANDOFF.md` — §1 현재 위치, §6 Phase 표
2. `PLAN.md` — §10.1 (Phase 4.1/4.2 + **CTRL-2x2** + POST-HOC-5/6/7), §5 측정 사전등록, §5.8 시나리오
3. `docs/RESULTS.html` — 워크숍 short-paper (세 발견 + 통제 2×2). **읽고 시작점 잡기**
4. `docs/PROCGEN_ENV.md`, `docs/LANGUAGE_SPEC.md` (필요 시)

이 4~5개 읽으면 컨텍스트 95% 복원. 다 읽고 **"한 줄 현재 위치"부터 보고**할 것.

---

## [지금까지의 결론 — 한 문단]

이질적 두 네트워크(procgen 미로 IMPALA-CNN RL 에이전트 = 결정자, 합성 미로언어 소형
트랜스포머 LM = 해석자)를 from scratch 공동학습하고 그 사이에 인공 뇌량(ACC)을 두었다.
**핵심 가설**(분리-재구성 V2 > next-token 어댑터 B4 충실도)은 **기각**됐다(Scenario C).
결정적 테스트 + 활성 스왑 + **통제 2×2(CTRL-2x2)**로 confound까지 해소한 결과:
**재구성 학습 신호 자체가 next-token보다 약하다**(인터페이스를 동일하게 맞춰도 얇은 쌍에서
B4Thin이 V2를 충실도 +0.31 / swap +0.40로 압도). 단 풍부한 negative + 재프레임을 얻었다.

---

## [핵심 전환 — 무엇이 죽고 무엇이 살았나]

**죽은 것**
- "분리-재구성(V2)이 충실도의 핵심 재료"라는 가설. 인터페이스 통제 후에도 재구성 ≪ next-token.
- V2Rich(rich×재구성)는 *degenerate*(loss↓≠성공 재발; full-hidden 타깃 ill-posed). 그대로는 폐기.

**산 것 (= 후속 연구의 토대)**
1. **B4(next-token + 분산 cross-attn)이 인과적으로 충실한 통역사다** — 활성 스왑 0.83~0.99.
   에이전트 표상을 강하게 인과 추적. *충실한 통역사는 만들 수 있다*(가설의 좁은 형태만 죽음).
2. **"충실 ≠ 합리화" 재프레임** — 목표 오일반화한 에이전트를 충실히 읽으면 오일반화 목표를
   보고한다(B4의 OOD "합리화" 50%는 둘러대기가 아니라 충실한 reading). 정렬·해석가능성의 미묘점.
3. **목표 오일반화가 표상 수준에서 읽힌다** — cheese_dir 충실도 in-dist 0.85 → OOD 0.07 전 빌드 붕괴.
   에이전트가 실제 치즈방향을 미표상, "우상단" prior 표상.
4. **측정 #3 활성 스왑이 충실도의 진짜 판별기** — 절대 충실도·합리화율의 confound를 인과로 가름.
5. **두 손잡이 다 B4 우세** — 좋은 학습 신호(next-token) *그리고* 분산 인터페이스 둘 다 기여.

---

## [새 연구 방향 후보 — 이번 세션에 AskUserQuestion으로 고를 것]

> 옵션 + 추천 + 이유 + 비유로 제시하고 사용자와 함께 1개(또는 조합) 확정 → PLAN §10.1 사전등록 박제.

**A. 인과를 *직접* 최적화하는 통역사 (Interchange Intervention Training / DAS) ★강력 추천**
- 발상: 재구성 같은 *proxy*를 버리고, **"h_agent의 일부를 다른 상태로 스왑하면 보고가 그쪽을
  따라가야 한다"**를 *학습 목표*로 삼는다. 우리가 *재는 것*(swap-following)을 *직접 최적화*.
- V2 실패의 정확한 교훈: 충실도를 *기대*하는 proxy(재구성)가 아니라 충실도를 *강제*하는 목표.
- 문헌 앵커: causal abstraction / interchange intervention training / distributed alignment search
  (Geiger et al.). 비유: "베껴 그려봐"가 아니라 "내가 머릿속 한 조각을 바꾸면 네 말이 바뀌어야 해"로 훈련.
- 비교: 같은 B4 인터페이스에서 next-token-only vs +interchange-objective. CTRL-2x2 골격 재사용.

**B. 지각 vs 의도 분리 readout ★강력 추천 / 핵심 기여 후보**
- 발상: 에이전트가 *실제 치즈 위치(지각)* 와 *추구 목표(우상단)* 를 둘 다 표상하는지, 충실한
  통역사가 어느 쪽을 보고하는지, **둘을 분리해 보고**하게 만들 수 있는지. (현재 데이터: OOD에서
  cheese_dir 붕괴 = 에이전트가 *행동 안 하는 지각은 버린다*는 신호 → recurrent/큰 에이전트면 다를 수도.)
- 임팩트: 정렬·기만 탐지에 직결("이 에이전트가 *보는 것* vs *원하는 것*"을 분리 진단).
- 비유: 운전자가 "표지판은 봤지만(지각) 그래도 늘 가던 길로 간다(의도)"를 통역사가 둘 다 말하게.

**C. 에이전트 표상 자체를 키우기**
- recurrent 에이전트(heading 천장이 *단일 프레임 한계*에 묶여 있었음 → 기억 주면 ↑), 또는 더 큰
  CNN이 *행동 안 해도* 지각 정보를 보존하는지. multi-layer 추출(#1 Deferred)도 여기.
- 주의(사전 예측): "큰 모델 1주"는 *목표 오일반화를 더 강화*할 뿐, cheese_dir OOD를 못 살림.
  크기가 아니라 *구조(메모리)* 가 지렛대.

**D. well-posed richer-reconstruction (V2Rich 수선)**
- rich×재구성 칸을 *공정히* 재려면 타깃 재설계: 에이전트-knowable 내용만 타깃, 또는 재구성 요약을
  lm.generate에 태우는 *견고한 디코드*. 우선순위 낮음(얇은 쌍이 이미 재구성 열세를 확정).

**E. 마무리 트랙 (논문 확정)**
- 얇은 쌍(V2 vs B4Thin) **multi-seed 3~5** paired-bootstrap 통계 + **#4 Procrustes**(W 위치무관).
- 현재 전부 **1 RL seed descriptive** → 통계 확정 시 워크숍 short-paper 투고 가능.

**[추천]** **A** 또는 **B**가 가장 새롭고 임팩트 큼. A = *방법론*(충실을 학습), B = *현상*(지각/의도
분리). 둘 다 SPLIT-9 → SPLIT-MNIST → SPLIT-MAZE 아크의 자연스러운 다음 장. E는 어느 쪽을 가든
병행할 통계 마무리.

---

## [프로젝트 문화 — 항상 지킬 것]

- 큰 결정은 **AskUserQuestion** (옵션 + 추천 + 이유 + 비유). 작은 단위로 만들고 각각 검증 후 진행.
- 모든 결정·러닝은 **PLAN.md §10.1 + docs/SESSION_HANDOFF.md**에 박제.
- 사전 등록 임계치(§5.6)는 결과 본 뒤 변경 금지 — 사유와 함께 기록만.
- 사용자 의심은 묵살 말고 **실험으로 검증**. 막히면 외부 문헌 참고. **`loss↓ ≠ 성공`** 항상 의심
  (trivial/degenerate 해 체크 — POST-HOC-6, V2Rich 두 번 데임).
- **WSL이 1차 검증자**(torch + procgen + GPU). sandbox는 단위테스트·코드 작성용(torch 디스크 제약 잦음).
  bash mount는 가끔 stale 스냅샷 — Read/Grep(Windows-truth)을 진실로, WSL이 확정.
- 지식 컷오프(2026-05) 이후 사실은 **웹 검색**으로 한 번씩 확인하고 진행.
- 모든 결과는 현재 **1 seed descriptive** — 통계 확정은 multi-seed(§5.7) 과제임을 명시.

---

## [이번 세션 첫 행동]

위 문서 읽고 **"한 줄 현재 위치"** 보고 → 새 방향 **A~E를 AskUserQuestion**으로 제시·확정 →
사전등록 박제(PLAN §10.1) → 작은 단위 구현(코드는 sandbox, 검증·실험은 WSL).

> 산출물 인벤토리(이미 있음): `checkpoints/phase3/{agent,B4,V2,V2_postfix2,B3}.pt`, `checkpoints/lm.pt`,
> `results/phase4_{builds,swap,ctrl2x2}.json`, `scripts/{eval_builds,swap_test,diagnose_v2,ceiling_v2,
> retrain_v2_acc,fit_2x2}.py`, `src/split_maze/{builds,acc,adapter,lm,...}.py`. 새 빌드는
> `builds.py`의 `Build` ABC + CTRL-2x2 하니스(`fit_2x2.py`)를 재사용해 작은 단위로 붙이면 된다.
