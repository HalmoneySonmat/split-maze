# procgen 환경 디리스킹 — Phase 0 산출물

> Phase 0 task #65. 2026-05-15 기준 웹 조사 + 샌드박스 실측.
> PLAN §8.2 (procgen C++ 빌드 위험)에 대한 구체 결론.

## 1. 결론 요약

- **결정자 환경 = `JacobPfau/procgenAISC`** — Langosco et al. "Goal
  Misgeneralization in Deep RL"의 *바로 그* procgen 포크 (132 commits, 17 forks).
- **학습 환경 = `maze_aisc`** ("치즈가 항상 우상단 코너" — 소스에
  `maze_aisc.cpp` 확인). **OOD 평가 환경 = 평범한 `maze`** (치즈 임의 위치).
- 환경 수정이 *이미 구현*돼 있음 → 어제 검토했던 "치즈 검출기 + 레벨
  필터링"(구 옵션 2)은 **불필요**. procgenAISC가 그 일을 이미 해놓음.
- **WSL 빌드 확인 완료 (2026-05-15) — §5 참조.**

## 2. 샌드박스에서 확인된 것 (Ubuntu, Python 3.10.12, 2026-05-15)

| 항목 | 결과 |
|---|---|
| `pip install procgen` (base 0.10.7) | ✅ 성공. obs `(1,64,64,3)` uint8, action `Discrete(15)`, 20스텝 롤아웃 정상. |
| base procgen `maze` | ✅ 동작. 단 평범한 maze만 — goal-misgen 환경 없음. |
| `pip install procgen-tools` (PyPI) | ❌ 실패 — 옛 `gym` 의존성 빌드 불가 (PyPI판 rot). |
| `JacobPfau/procgenAISC` git clone | ✅ 성공. `procgen/src/games/maze_aisc.cpp` 존재 확인. |
| `eczy/procgen2` (Farama-Foundation/Procgen-Staging 계열) | 살아있는 유지보수 포크. 단 goal-misgen 환경 없음 (평범한 maze만). |
| procgenAISC conda 빌드 | ⛔ 샌드박스 검증 불가 — 네트워크 프록시가 conda/micromamba 다운로드(GitHub 릴리스·CDN) 403 차단. PyPI·git clone은 통과. → 사용자 WSL에서 검증. |

## 3. procgenAISC 빌드 레시피

`environment.yml` 핀: `python=3.7.3, cmake=3.14.0, qt=5.12.5` + pip
`gym3==0.3.3, numpy==1.17.2, gym==0.15.3, filelock==3.0.10`.
→ **conda 기반 C++ 빌드** (Qt 5.12.5 필요). `build.py`가 Qt cmake 경로를
conda env에서 찾음. C++는 `.so` 공유 라이브러리로 컴파일되어 Python wrapper가
로드 (gym3.libenv C 인터페이스).

## 4. 사용자 WSL에서 실행할 Phase 0 빌드 검증

```bash
# WSL Ubuntu. miniconda/miniforge 필요 (없으면 먼저 설치).
git clone https://github.com/JacobPfau/procgenAISC.git
cd procgenAISC

# --- 시도 A (권장): Python 3.10 통일 env ---
# base procgen이 cp310 wheel을 내므로 같은 빌드 시스템인 procgenAISC도
# 3.10에서 빌드될 가능성 높음. 성공하면 SPLIT-MAZE 전체가 한 env.
# 주의 1: 2025년부터 conda는 Anaconda 기본 채널(repo.anaconda.com) ToS 동의를
# 요구. 우리는 conda-forge만 필요 → --override-channels 로 기본 채널 우회.
# 주의 2: conda-forge의 python은 pip을 자동으로 안 딸려옴 → pip 명시 + python -m pip.
#         (안 그러면 pip이 시스템 Python으로 새어 PEP 668 externally-managed 오류)
conda create -n splitmaze python=3.10 pip cmake qt=5.12 -c conda-forge --override-channels -y
conda activate splitmaze
python -m pip install -e .
# 주의 3: procgenAISC setup.py는 런타임 의존성을 자동 설치 안 함 → 별도 설치.
#         environment.yml의 옛 핀(gym==0.15.3, numpy==1.17.2)은 py3.10에서
#         빌드 불가 → 최신판으로. gym3가 자기 의존성(glfw·moderngl 등)을 끌어옴.
python -m pip install gym3
python -c "from procgen import ProcgenGym3Env; e=ProcgenGym3Env(num=1, env_name='maze_aisc'); print('maze_aisc OK', e.observe()[1]['rgb'].shape)"
python -c "from procgen import ProcgenGym3Env; ProcgenGym3Env(num=1, env_name='maze'); print('maze OK')"

# --- 시도 B (A 실패 시): 문서 핀 그대로 (py3.7 별도 env) ---
# 이 경우 procgen은 py3.7 env, RL 에이전트(torch 2.x)는 별도 py3.10+ env로
# 분리됨 → 롤아웃 데이터를 파일/소켓으로 주고받는 구조 필요.
conda env update --name procgen_aisc --file environment.yml
conda activate procgen_aisc
pip install -e .
python -c "from procgen import ProcgenGym3Env; ProcgenGym3Env(num=1, env_name='maze_aisc')"
```

빌드가 `building procgen...done` 후 `maze_aisc` 환경 생성에 성공하면
Phase 0 procgen 항목 통과.

## 5. WSL 빌드 결과 — ✅ 성공 (2026-05-15)

사용자 WSL(Ubuntu)에서 시도 A 경로로 빌드 확인:
- `conda create -n splitmaze python=3.10 pip cmake qt=5.12 -c conda-forge --override-channels`
- `python -m pip install -e .` → `building procgen...done`
- `python -m pip install gym3`
- `maze_aisc` 환경: `maze_aisc OK (1, 64, 64, 3)` ✅
- `maze` 환경: `maze OK` ✅

→ **PLAN §8.2 최고 위험(procgen C++ 빌드) 통과.** 시도 A(Python 3.10 통일
env) 성공 → SPLIT-MAZE 전체가 한 conda env(`splitmaze`)에서 돌아감.
시도 B(py3.7 분리)·Docker fallback 불필요.

도중 만난 gotcha 3개 (모두 §4 주석에 박제): ① conda ToS → `--override-channels`,
② conda-forge python에 pip 없음 → `pip` 명시 + `python -m pip`, ③ procgenAISC
setup.py가 런타임 의존성 자동설치 안 함 → `gym3` 별도 설치. gym3의 "Gym
unmaintained" 출력은 무해한 경고.

## 6. 환경 옵션 메모 (Phase 1에서 쓸 것)

procgen 공통 옵션 (procgen2 README 기준): `num_levels`, `start_level`,
`distribution_mode` (`easy/hard/...`), `use_sequential_levels`,
`center_agent`, `use_backgrounds`, `use_monochrome_assets` 등.
`get_state()`/`set_state()`로 상태 저장·복원 가능 (활성 스왑 측정 #3에
유용할 수 있음 — Phase 4).

## 7. 확정 차원 (Phase 0 #67 — 2026-05-15)

| 항목 | 값 | 근거 |
|---|---|---|
| 관측 (obs) | (B, 3, 64, 64) uint8 | procgenAISC 빌드 확인 (rgb `(1,64,64,3)` → transpose) |
| 행동 공간 | `Discrete(15)` | procgen 표준 (샌드박스 확인) |
| IMPALA-CNN depths | `[16, 32, 32]` | train-procgen-pytorch (goal-misgen 연구 표준 repo) |
| **d_a (에이전트 추출 = IMPALA embedding)** | **256** | train-procgen-pytorch IMPALA-CNN embedding size 256. §3.4 "마지막 dense 단일 지점" 확정. |
| 소형 LM d_lm | sweep {128, 192, 256}, 기본 256 | §3.4 (Phase 2 sweep). 기본 256이면 ACC W가 정사각 256×256. |
| 소형 LM 층수 | sweep {2, 3, 4}, 기본 3 | §3.4 |
| ACC W | d_lm × d_a (≤ 256×256 ≈ 65k params) | §4.2 tied W + Wᵀ |

에이전트 구조 = train-procgen-pytorch의 IMPALA-CNN (PPO, no recurrent) —
goal-misgen 문헌이 쓰는 바로 그 구조. Phase 1에서 from scratch 재학습:
`input (3,64,64) → [16,32,32] IMPALA 블록 → flatten → Linear→256 → ReLU
= h_agent(256)`, 그 뒤 policy head(→15) + value head(→1).

## 8. 미로 스프라이트 색 (Phase 0 #69)

`procgenAISC/procgen/data/assets/`의 실제 PNG에서 추출 (`src/split_maze/env.py`
에 박제):

| 스프라이트 | 색 (RGB) | 픽셀 수 (원본 PNG) |
|---|---|---|
| **CHEESE** (`misc_assets/cheese.png`, 27×27) | (253,155,37) 오렌지 / (254,231,98) 노랑 / (181,112,82) 갈색 | 4 unique RGB, 322 opaque px |
| **MOUSE** (`kenney/Enemies/mouse_move.png`, 128×128) | (187,203,204) 청회색 / (243,185,203) 핑크 / (100,122,123) 진회색 | 71 unique RGB, 4147 opaque px |

64×64 다운샘플 후 L1 거리 ≤ 20 매칭. 바닥(~200,145,95)과의 L1 거리는 최소
75 이상 → tolerance 20은 충분히 안전. 샌드박스에서 `maze` 환경 + seed 0/5/13
모두 두 스프라이트 모두 검출 (cheese 6~9px, mouse 3~5px).

## 9. Phase 0 #69 사용자 WSL 검증 명령

```bash
conda activate splitmaze
cd /mnt/d/brain/split_maze
python -m pip install pytest numpy   # tests용 (procgen은 splitmaze에 이미)
PYTHONPATH=src python -m pytest tests/ -q   # 41 tests: language + env + 실제 procgen 통합
PYTHONPATH=src python scripts/check_env.py --env_name maze_aisc --steps 50 --seed 0
```

`✓ PHASE 0 #69 PASS` 뜨면 Phase 0 전체 완료 — Phase 1 진입 가능.
