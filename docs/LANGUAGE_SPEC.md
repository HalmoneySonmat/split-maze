# 합성 미로언어 + describer oracle — 스펙 v0.1

> PLAN.md §3.3 박제(② 3슬롯 최소)의 구체 스펙. Phase 0 산출물.
> 이 문서가 정하는 것: 정확한 어휘, 문장 문법, describer oracle 함수,
> 표면 다양성 규칙, 중립 코퍼스 생성, 파싱 규칙.

**상태**: v0.1 박제 ✔ (2026-05-15, 사용자 확인). v0.3 패치 ✔ (2026-05-19,
PLAN §10.1 POST-HOC-3 — AGENT_REGION을 4 슬롯으로 분할, vocab 25→26).
*v0.2 (POST-HOC-2 단일 토큰화)는 학습 실패로 시도-접기*. `[Phase-time]`
표시는 구현 착수 시 확정.

---

## 1. 목적 (PLAN §3.3 요약)

3슬롯 최소 언어로, **에이전트의 실제 이동 방향(HEADING)**과 **치즈 방향
(CHEESE_DIR)**을 *독립 슬롯*으로 표현 — 둘이 어긋나는 문장(목표 오일반화
시그니처)을 문법이 허용해야 §5.1 결정적 테스트가 성립.

- describer oracle = **객관 사실 내레이터** (속마음 아님).
- LM 코퍼스 = **문법 샘플링 중립** (미로 불필요, 슬롯 상관 0).
- 편향(goal-misgen)은 오직 에이전트 안에만 산다.

---

## 2. 슬롯과 값

| 슬롯 | 의미 | 값 (개수) | 토큰화 |
|---|---|---|---|
| `AGENT_REGION` | 에이전트 위치 (3×3 격자) | row{top,middle,bottom} × col{left,center,right} = 9 | **단일 compound 토큰 9개** (v0.2 POST-HOC-2): `top-left`, `top-center`, ..., `bottom-right` |
| `HEADING` | 에이전트의 실제 이동 방향 | 8방위 + `still` = 9 | 단일 토큰 |
| `CHEESE_DIR` | 에이전트 기준 치즈 방향 | 8방위 = 8 | 단일 토큰 |

전체 상태 공간 = 9 × 9 × 8 = **648 triple**.

**v0.2 POST-HOC-2 변경**: AGENT_REGION을 *2 토큰 (`<row> <col>`)에서 단일
compound 토큰 (`<row>-<col>`)으로* 표현 변경. 의미 구조(슬롯, 값 개수)는
동일. 이유는 PLAN §10.1 POST-HOC-2.

8방위 토큰: `up, up-right, right, down-right, down, down-left, left, up-left`
(하이픈 복합어는 *단일 토큰*).

---

## 3. 어휘 (vocabulary)

| 범주 | 토큰 | 수 |
|---|---|---|
| 8방위 | `up up-right right down-right down down-left left up-left` | 8 |
| 정지 | `still` | 1 |
| 격자 행 (AGENT row 값) | `top middle bottom` | 3 |
| 격자 열 (AGENT col 값) | `center` (`left`/`right`는 8방위와 공유) | 1 |
| 슬롯 마커 | `agent column heading cheese` | 4 |
| 마커 동의어 (표면 다양성, §6) | `it going moving` | 3 |
| 연결어 (선택적, §6) | `and ,` | 2 |
| 특수 토큰 | `<BOS> <EOS> <PAD> <SUM>` | 4 |
| **합계** | | **26** (v0.3 POST-HOC-3; v0.1=25, v0.2=34 시도-접기) |

> v0.3 POST-HOC-3 변경: AGENT_REGION 슬롯을 *2 sub-slot* (`agent <row>` +
> `column <col>`)로 토큰화 — 각 sub-slot이 HEADING/CHEESE_DIR과 동일한
> `<marker> <단일 토큰>` 모양. 새 마커 `column` 1개 추가. 의미 구조는 그대로.

- `left`/`right`는 8방위와 격자 열에서 *공유* — 슬롯 마커가 문맥을 완전히
  disambiguate (자연어처럼 토큰 재사용, 작은 트랜스포머가 쉽게 처리).
- `<SUM>` = §3.4 손잡이 B의 요약 슬롯. 정확한 배치는 `[Phase-time]` (Phase 2
  LM 구현 시 — §3.4).

---

## 4. 문장 문법

**정규형 (canonical form, v0.3 POST-HOC-3)**:

```
agent <row>  column <col>  heading <dir|still>  cheese <dir>
```

예: `agent top column right heading up-right cheese down-left`
← 목표 오일반화 시그니처 (heading ≠ cheese 방향).

> 4 슬롯 셔플 가능 (4! = 24 순서). 각 슬롯이 `<marker> <단일 토큰>` 모양으로
> 통일 — Phase 2.2 학습 비대칭 fix (PLAN §10.1 POST-HOC-3).

- 각 슬롯은 *자기 마커*(`agent`/`heading`/`cheese`)로 시작 → 슬롯 순서가
  바뀌어도 파싱 가능 (§6 표면 다양성의 근거).
- 학습/평가 시 `<BOS> ... <EOS>`로 감쌈.
- 내용 토큰 수: 정규형 9개 (마커 3 + 값 6).

---

## 5. describer oracle — 결정적 함수

**입력**: 미로 상태 + 에이전트의 최근 궤적.
**출력**: 위 3슬롯을 채운 정규형 문장 (그 후 §6 표면 변형 적용).

```
describe(maze_state, trajectory) -> sentence:
    AGENT_REGION = quantize_to_3x3(agent_xy, maze_dims)
    HEADING      = quantize_8way( net_displacement(trajectory, last K steps) )
                   or 'still' if |net_displacement| < move_threshold
    CHEESE_DIR   = quantize_8way( cheese_xy - agent_xy )
    return render(AGENT_REGION, HEADING, CHEESE_DIR)
```

- **객관 사실만**: 관측 가능한 위치·이동·치즈 방향만. 에이전트의 *의도*
  ("구석에 가고 싶어 함")는 절대 쓰지 않음 — ACC가 복원할 대상.
- `HEADING`은 단일 프레임이 아니라 **최근 K스텝 순변위** 기준 (목표
  오일반화는 지속 행동). `[Phase-time]` K=4 제안, 튜닝 가능.
- `move_threshold`: 순변위가 이보다 작으면 `still`. `[Phase-time]`.
- **에지 케이스 — 에이전트가 치즈 위에 있음**: CHEESE_DIR 정의 불가.
  해당 상태는 describer oracle 문장에서 *제외* (드물고, 에피소드도 거기서
  종료). CHEESE_DIR에 `reached` 추가는 ③ 문법형(D-1)으로 미룸.
- `quantize_to_3x3`의 연속 좌표 → 격자 매핑은 미로 크기에 의존 →
  `[Phase-time]` (Phase 0 procgen 디리스킹이 미로 dims 확정).

---

## 6. 표면 다양성 (template memorization 방지 — PLAN §8.5)

*내용*(3슬롯 값)은 결정적이되, *표면 형태*는 다양하게 → LM이 1:1 템플릿을
통째 암기 못 하게.

1. **슬롯 순서 순열**: 3슬롯이 3! = 6가지 순서 중 무작위. (각 슬롯이
   자기 마커로 시작하므로 순서 무관하게 파싱 가능.)
2. **마커 동의어**: `heading` ↔ `going` ↔ `moving`; `agent` ↔ `it`.
   (`cheese`는 고정.)
3. **선택적 연결어**: 슬롯 사이에 `and` 또는 `,` 무작위 삽입/생략.

한 triple당 표면 변형 수 ≈ 6(순서) × 3(heading 동의어) × 2(agent 동의어)
× 연결어 변형 ≈ 수십 가지. `[Phase-time]` 동의어 집합 크기는 조정 가능.

---

## 7. 중립 코퍼스 생성 (LM 학습용 — PLAN §3.3)

```
generate_corpus(N):
    for _ in range(N):
        triple = uniform_sample(AGENT_REGION × HEADING × CHEESE_DIR)  # 균등!
        surface = random_surface_form(triple)                         # §6
        yield wrap_BOS_EOS(surface)
```

- **균등 샘플링이 핵심**: (HEADING, CHEESE_DIR) 상관이 0 → LM = 중립 언어
  기질. 어떤 조합도 말할 수 있고 특정 prior 없음.
- LM은 미로를 *전혀* 보지 않음 — 순수 문장 코퍼스.
- `[Phase-time]` 코퍼스 크기 N (제안: ~50k 문장) + train/held-out 분할.
- 648 triple 전부 train에 등장하도록 보장 (held-out은 *표면 형태* 기준
  분할, triple 기준 아님 — LM은 모든 의미를 봐야 함).

---

## 8. 파싱 규칙 (평가 #2 슬롯 일치율 채점용)

```
parse(sentence) -> {AGENT_REGION, HEADING, CHEESE_DIR} or partial:
    토큰 스캔:
      'agent'/'it'        다음 2토큰 → AGENT_REGION
      'heading'/'going'/'moving' 다음 1토큰 → HEADING
      'cheese'            다음 1토큰 → CHEESE_DIR
    마커 누락 / 값이 유효 토큰 아님 → 해당 슬롯 = 오답(None)
```

- 슬롯 순서·동의어·연결어에 강건.
- LM이 degenerate 출력(반복·누락)을 내면 해당 슬롯이 None → 충실도
  점수에 자연히 반영됨.
- 슬롯 일치율 = (정답과 일치한 슬롯 수) / 3.

---

## 9. `[Phase-time]` 미정 항목 정리

| 항목 | 값 | 상태 |
|---|---|---|
| HEADING 윈도우 K | 4 스텝 | Phase 2 |
| `move_threshold` | — | Phase 2 (에이전트 행동 스케일 보고) |
| 동의어 집합 크기 | §3 기준 | Phase 2 |
| **코퍼스 크기 N** | **50,000** | **박제 ✔ 2026-05-19 (PLAN P2-4)** |
| train/held-out 분할 | 90/10, 표면 형태 기준 | Phase 2 (lm 학습 시) |
| **`<SUM>` 손잡이 배치** | **시퀀스 끝에 명시 추가 (`<BOS> ... <SUM>`)** | **박제 ✔ 2026-05-19 (PLAN P2-1)** |
| **LM 크기** | **3층, d_model=256, n_head=4, FFN 1024** | **박제 ✔ 2026-05-19 (PLAN P2-2)** |
| **λ_ae (오토인코딩 가중치)** | **1.0 (L_nexttoken + 1.0·L_ae)** | **박제 ✔ 2026-05-19 (PLAN P2-3)** |
| 3×3 격자 ↔ 연속 좌표 매핑 | — | Phase 0 (procgen 미로 dims) |
| `reached` (치즈 도달) | 제외 → D-1 | ③ 문법형 확장 시 |

---

## 10. 구현 위치

- `src/split_maze/language.py` — 어휘, 문법, describer oracle, 코퍼스
  생성기, 파서.
- `tests/test_language.py` — 어휘 크기, 정규형 라운드트립, 파서 강건성,
  코퍼스 균등성(슬롯 상관 ≈ 0) 단위 테스트.
