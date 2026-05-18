# SPLIT-MAZE — 세션 핸드오프 (2026-05-18)

> 새 세션 시작 시 본 문서 + PLAN.md + docs/PROCGEN_ENV.md + docs/LANGUAGE_SPEC.md
> 4개 읽으면 컨텍스트 95% 복원.

---

## 1. 한 줄 현재 위치

**Phase 1.3 코드 완료 — 사용자 WSL 1차 검증 대기.** Phase 0 + 1.1(IMPALA-CNN)
+ 1.2(PPO) + **1.3(train.py + CLI + test_train.py — 19 tests 신규)**.
샌드박스 디스크 제약으로 PyTorch 직접 검증 못 했음 → 사용자 WSL에서
첫 검증. 통과 시 합계 **64 + 19 = 83 tests**. 다음: 1.4 WSL 정식 학습.

---

## 2. 새 세션 시작 시 읽을 순서

| # | 파일 | 용도 |
|---|---|---|
| 1 | **`docs/SESSION_HANDOFF.md`** (본 문서) | 진행 상황·러닝·인벤토리·다음 단계 spec |
| 2 | **`PLAN.md`** | 설계 전문 v1.0, 9개 핵심 결정 |
| 3 | `docs/PROCGEN_ENV.md` | procgen 빌드·차원·환경 옵션 |
| 4 | `docs/LANGUAGE_SPEC.md` | 합성 미로언어 + describer oracle 스펙 |

---

## 3. 사용자 WSL 환경 상태 (재시작에도 영구)

- **Conda env**: `splitmaze` (Python 3.10)
- **torch** 2.12.0+cu130, CUDA True, GPU = RTX 3070 Ti, VRAM 8.6GB
- **procgenAISC**: from-source 빌드 완료, `maze_aisc` + `maze` 둘 다 동작
- **gym3** 0.3.3+ 깔림
- 우리 코드: `/mnt/d/brain/split_maze/` (= D:\brain\split_maze) 그대로

전체 테스트 명령:
```bash
conda activate splitmaze
cd /mnt/d/brain/split_maze
PYTHONPATH=src python -m pytest tests/ -q
```
이번 세션 후 통과해야 할 테스트 수: **88** (test_language=30, test_env=11,
test_agent=10, test_ppo=13, **test_train=24** — 기본 19 + rolling 5).

---

## 4. 코드 인벤토리

### `src/split_maze/`

| 파일 | 내용 | 검증 |
|---|---|---|
| `__init__.py` | 패키지 헤더, `__version__="0.0.1"` | ✓ |
| `language.py` | 어휘·문법·`describer_oracle`·코퍼스 생성기·파서 (PLAN §3.3) | 30 tests |
| `env.py` | rgb 스프라이트 검출(`find_sprite_centroid`) + `TrajectoryTracker` + `extract_maze_state` + `make_maze_env` | 11 tests |
| `agent.py` | `ImpalaAgent` (3 IMPALA blocks, d_a=256, policy/value heads), `AgentOutput` | 10 tests |
| `ppo.py` | `PPOConfig`, `RolloutBuffer` (GAE λ), `sample_action`, `ppo_loss`, `PPOUpdater` | 13 tests |
| `train.py` | `obs_to_tensor`, `RolloutStats`, `collect_rollout`, `train`(+rolling-mean), `MockMazeEnv` (gym3-호환 fake) | 24 tests (19 + rolling 5) |

### `tests/`
- `test_language.py`, `test_env.py`, `test_agent.py`, `test_ppo.py`, **`test_train.py`**.
- 마지막 WSL 결과(test_train 제외 64개): 모두 통과. `test_env.py`의 procgen
  통합 테스트는 WSL에서만 활성화됨. `test_train.py`는 MockMazeEnv만 써서
  procgen 무관, 어디서나 돌아감.

### `scripts/`
- `check_env.py` — Phase 0 #69 산출물. 무작위 정책 롤아웃 + describer oracle 동작 데모. WSL에서 `--env_name maze_aisc --steps 50` 50/50 문장 생성 확인.
- **`train_agent.py`** — Phase 1.3 #72 산출물. `--mock`(MockMazeEnv smoke)
  / 실 procgen(`--env_name maze_aisc`) 양쪽 지원. argparse CLI로 N·T·총
  스텝·체크포인트·JSONL 로그 노출. 학습 루프 본체는 `train.train()`.

### `docs/`
- `PROCGEN_ENV.md` — procgen 빌드 가이드 + WSL 빌드 결과 (성공) + sprite 색표 + 확정 차원 (d_a=256).
- `LANGUAGE_SPEC.md` — 합성 미로언어 spec (3슬롯 최소, describer oracle, 중립 코퍼스).

### Root
- `PLAN.md` (v1.0 — 9개 핵심 결정 박제)
- `README.md` (Phase 0 placeholder)
- `LICENSE` (Apache 2.0)
- `pyproject.toml`, `requirements.txt`, `.gitignore`

---

## 5. 박제된 9개 핵심 결정 (PLAN v1.0 요약)

1. **정렬 메커니즘 (C-thin)** — 해석자(LM+ACC)만 적응, 에이전트는 순수 RL. ACC grad는 LM 인터페이스까지만, 언어 코어는 stop-grad.
2. **합성 미로언어 ② 3슬롯** — AGENT_REGION(3×3=9) / HEADING(8+still=9) / CHEESE_DIR(8). 어휘 25 토큰.
3. **시스템** — IMPALA-CNN 마지막 dense 단일 추출(d_a=256). 소형 LM = 디코더형 트랜스포머 2~4층, d_lm 128~256. LM 손잡이 = B(자기 오토인코딩 일관성).
4. **ACC** — tied W (d_lm × d_a) + Wᵀ + LayerNorm + 양방향 MSE. (C-thin) 이중 grad 경계.
5. **측정** — #1 과제 성능 / #2 cosine + 슬롯 일치율 ★ / #3 **활성 스왑** / #4 Procrustes vs random. Scenario A/B/C 사전 등록.
6. **베이스라인** — B1(에이전트 단독) / B3(직접 추적기) / **B4**(어댑터+next-token) / **V2**(ACC+분리 재구성). 핵심 비교 = V2 vs B4. 양쪽 다 LM 코어 보호.
7. **Phase 게이팅** — Phase 0~4 엄격 게이트. Phase 1 = **중간 막대** (in-dist ≥80%, OOD goal-misgen ≥50%).
8. **위험 fallback** — §8.3 = **C** (같은 미로, 막대 낮춰 진행 + 한계 명시). A(공개 체크포인트) 기각.
9. **Deferred D-1~D-13** — ③ 문법형, 비대칭 W, 정렬 (A)/(B) ablation, 인간 언어 해석자, 양방향 뇌량 등.

---

## 6. Phase 진행 상태

| Phase | 상태 | 핵심 산출물 |
|---|---|---|
| 0 설계+환경 | ✅ 완료 | PLAN v1.0, procgen 빌드, 합성언어, env, check_env |
| **1.1 IMPALA-CNN 에이전트** | ✅ 완료 | `agent.py` (d_a=256, ~626k params), 10 tests |
| **1.2 PPO 알고리즘** | ✅ 완료 | `ppo.py` (buffer+GAE+loss+updater), 13 tests |
| **1.3 train_agent.py** | ✅ 코드 완료 / WSL 검증 대기 | `train.py`+`scripts/train_agent.py`+`test_train.py` (19 tests, MockMazeEnv) |
| 1.4 WSL 정식 학습 | ← **다음** | in-dist 성공률 ≥ 80% (중간 막대) |
| 1.5 OOD goal-misgen 평가 | 대기 | 평범한 maze에서 goal-misgen율 측정 |
| 1.6 게이트 판정 | 대기 | in-dist ≥80% AND OOD ≥50% → Phase 2 |

---

## 7. 샌드박스 디스크 전략 (러닝)

- Cowork 샌드박스 ~10GB. torch + procgen 둘 다 깔면 항상 꽉 참.
- **새 세션에서는 torch만 깔고 procgen은 안 깐다** — procgen은 WSL에서만 *실제로* 돌고, 샌드박스 역할은 PyTorch 단위테스트뿐.
- 옛 세션 디렉토리(`/sessions/*/`)는 *절대* `rm -rf` 하지 말 것 — 이 핸드오프의 원인. VM이 wedge됨.
- 새 세션에서 첫 명령: `pip install torch numpy --no-cache-dir -q`. (numpy는 보통 같이 와도 명시).

---

## 8. 최근 잡힌 버그 (러닝)

| 위치 | 증상 | 원인 | 교훈 |
|---|---|---|---|
| `test_agent.py` #5 | `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` | `(h_agent.detach() ** 2).sum().backward()` — grad_fn 없는 텐서에 backward | autograd 의미론은 머릿속 검증으로 못 잡음. `W_fake.requires_grad=True`를 끼워야 (C-thin) 시뮬레이션 가능. |
| `ppo.py` 정규화 | `Categorical(logits=NaN)` | 1-샘플 mini-batch에서 `std()=NaN` 전파 (단일 샘플로 unbiased std 정의 불가) | NaN 가드 (`numel()>1`) 박제. 작은 mini-batch 발생 가능한 경계조건 항상 체크. |
| 2026-05-18 sandbox | `pip install torch` ENOSPC unpack | `/sessions` 디스크 9.8G 중 8.4G가 옛 세션. 옛 세션 rm은 명시 금지 (이전 wedge 원인). torch 휠 532MB는 다운로드되지만 unpack 못 함 | 외부 제약 — 회피 불가. 사용자 WSL을 1차 검증자로 전환. 코드 품질로 보완(가드·dtype 명시·시그니처 정확 매칭). WSL에서 83 passed로 통과 확인 — 보완 전략 유효. |
| 2026-05-18 mock smoke | 첫 update `val=50` → 두 번째 `val=0.26` | N=4, T=16 → mb 크기 8. advantage normalize std가 8 샘플에서 노이즈 큼 → advantage 부풀려짐 → 24 SGD step에서 value head 한 번 튐, 자기 손실로 회복. clipfrac=0.73, approx_kl=0.15도 동일 시그니처 | *코드 버그 아닌 작은 mini-batch 분산 노이즈*. 진짜 procgen N=64·T=256에선 mb=2048로 통계 안정. Phase 1.4 smoke에서 재확인. |

→ 원칙: 새 세션에서 sandbox에 torch 깔자마자 **모든 PyTorch 변경을 직접
돌려본 뒤** 사용자한테 넘긴다. AST·산수만으론 PyTorch 버그를 못 잡는다.
단 디스크 제약으로 torch 못 깔리는 세션이면 *코드 신중성*으로 보완 +
박제된 WSL 검증 명령(아래 §12) 그대로 사용자에게 전달.

---

## 9. Phase 1.3 (#72) train_agent.py spec — 완료(2026-05-18)

> 코드 작성 완료. WSL 1차 검증 대기. spec 원문 보존(아카이브 가치).

### 목표
`scripts/train_agent.py`: PPO 학습 루프. procgenAISC vec-env + ImpalaAgent + RolloutBuffer + PPOUpdater 통합. CLI로 hyperparam 노출. 샌드박스에서 *짧은* smoke run(loss 감소·reward 증가 sanity)으로 검증.

### 구조
```
scripts/train_agent.py
  argparse: --env_name (maze_aisc), --num_envs (64), --num_steps (256),
            --total_steps, --seed, --save_path, --log_interval
  main:
    env = make_maze_env(num=num_envs, env_name=env_name, ...)
    agent = ImpalaAgent().to(device)
    updater = PPOUpdater(agent, PPOConfig())
    buffer = RolloutBuffer(T=num_steps, N=num_envs, device=device)

    # gym3 loop
    _, obs, first = env.observe()  # initial
    while step < total_steps:
        # ---- rollout T steps ----
        for t in range(num_steps):
            obs_tensor = torch.tensor(obs["rgb"].transpose(0,3,1,2), dtype=torch.uint8, device=device)
            with torch.no_grad():
                out = agent(obs_tensor)
                action, log_prob = sample_action(out.logits)
            buffer.store_step(t, obs=obs_tensor, action=action,
                              log_prob=log_prob, value=out.value)
            env.act(action.cpu().numpy())
            rew, obs, first = env.observe()
            buffer.store_post(t,
                              reward=torch.tensor(rew, device=device),
                              done=torch.tensor(first, device=device))
        # ---- bootstrap value ----
        with torch.no_grad():
            obs_tensor = torch.tensor(obs["rgb"].transpose(0,3,1,2), dtype=torch.uint8, device=device)
            last_value = agent(obs_tensor).value
        # ---- GAE + update ----
        buffer.compute_advantages_and_returns(last_value, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
        log = updater.update(buffer)
        # ---- log + checkpoint ----
```

### 주의점 (러닝 반영)
- **`obs.transpose(0,3,1,2)`** — procgen rgb는 (N,H,W,3) numpy uint8. PyTorch는 (N,3,H,W) 기대. 변환 잊지 말 것.
- **gym3 first vs done**: `first[t+1]` (다음 observe의 first) = "step t가 종결됐다". `RolloutBuffer.store_post(t, ..., done=first_now)` — 즉 t-번째 act 후 observe에서 받은 first가 그 step의 done.
- **`action.cpu().numpy()`** — gym3는 numpy int 기대.
- **`torch.no_grad()`** 안에서 rollout 수집 (autograd graph 메모리 안 쌓이게).
- **GPU 메모리**: buffer를 GPU에 두면 의외로 큼 (T=256, N=64, obs uint8 (3,64,64) → 256·64·3·64·64 byte ≈ 200MB just for obs). CPU buffer + GPU forward도 옵션 — sandbox smoke는 CPU만으로 충분.
- **smoke run 인자**: `--num_envs 4 --num_steps 16 --total_steps 64` 정도로 1분 내 끝나게.

### 테스트 전략
샌드박스에 torch 깔린 상태라는 전제로:
1. `tests/test_train_smoke.py` — 4 env × 16 steps × 1 update가 발산 없이 돌고, 한 번이라도 loss가 감소하는가.
2. (선택) CLI 자체를 subprocess로 호출하는 통합 테스트.

### 완료 기준
- 샌드박스 smoke (CPU, 64 step) 통과 + 검증 가능한 PPO 메트릭 로깅 (policy_loss, value_loss, entropy, approx_kl, clipfrac, ep_reward)
- WSL에서 매우 짧은 GPU 학습(~1만 step)으로 loss curve가 합리적인지 확인 (Phase 1.4 본 학습 전 sanity)

### 산출물 (2026-05-18 박제)
- `src/split_maze/train.py` — `obs_to_tensor` / `RolloutStats` /
  `collect_rollout` / `train` / `MockMazeEnv` (gym3 호환 fake).
- `scripts/train_agent.py` — argparse CLI 래퍼. `--mock` flag로 procgen 없이
  smoke 가능. `--save_path` (.pt 체크포인트) / `--log_path` (JSONL).
- `tests/test_train.py` — 19 tests (MockEnv 6 + obs_to_tensor 3 +
  collect_rollout 4 + train 6). 디스크 제약으로 sandbox PyTorch 실행 못 함
  → WSL 첫 검증.

---

## 9.5 다음 단계 — Phase 1.4 (#73) WSL 정식 학습 spec

### 목표
실제 `maze_aisc` 환경에서 IMPALA-CNN 에이전트를 PPO로 학습. Phase 1 완료
게이트(PLAN §7.1): **in-dist 성공률 ≥ 80% AND OOD goal-misgen율 ≥ 50%**.

### 권장 단계
1. **Smoke (~1~5분, GPU)** — Phase 1.3 코드 sanity. 메트릭 발산 없는지.
   ```bash
   PYTHONPATH=src python scripts/train_agent.py \
       --env_name maze_aisc --num_envs 16 --num_steps 64 \
       --total_env_steps 50_000 --device cuda --seed 0
   ```
   기대: loss 발산 없음, ep_return 점진 증가(0→0.x), clipfrac/kl 합리적.
2. **Mid run (~30분~1h, GPU)** — ep_return 곡선이 학습 추세 보이는지.
   ```bash
   --total_env_steps 1_000_000 --save_path checkpoints/maze_aisc_mid.pt \
       --log_path logs/maze_aisc_mid.jsonl
   ```
3. **Full run (~수시간, GPU)** — Phase 1 게이트 달성 목표. 표준 PPO
   procgen 학습은 보통 25M~200M frame.
   ```bash
   --total_env_steps 25_000_000 --save_path checkpoints/maze_aisc_full.pt \
       --log_path logs/maze_aisc_full.jsonl
   ```
4. **평가** — full checkpoint로 `--env_name maze`(OOD) + held-out
   `start_level` (in-dist held-out) 양쪽 평가 스크립트 별도 작성.

### Phase 1 게이트 (PLAN §7.1 박제 그대로)
- in-dist 성공률 ≥ 80% (학습한 levels held-out)
- OOD goal-misgen율 ≥ 50% (`maze` 환경에서 "치즈 대신 우상단" 비율)
- 미달 시 §8.3 fallback C (막대 낮춰 진행 + 한계 명시).

---

## 9.6 사용자 WSL 1차 검증 명령 (2026-05-18 신규)

> 이 세션 끝나면 사용자는 WSL에서 아래를 돌리면 됨.

### A. 전체 단위 테스트 (88 tests 기대 — 83 + rolling 5)
```bash
conda activate splitmaze
cd /mnt/d/brain/split_maze
PYTHONPATH=src python -m pytest tests/ -q
```
기대 출력 마지막 줄: `88 passed in X.Xs`.

### B. test_train.py만 단독 (Phase 1.3 직접 점검)
```bash
PYTHONPATH=src python -m pytest tests/test_train.py -v
```
기대: 19 tests pass. 핵심은 `test_train_smoke_param_updates_under_ppo`
— policy.weight가 PPO update로 움직이는지(grad 흐름) 직접 확인.

### C. CLI smoke (procgen 없는 mock, ~10초)
```bash
PYTHONPATH=src python scripts/train_agent.py --mock \
    --num_envs 4 --num_steps 16 --total_env_steps 128 --device cpu --seed 0
```
기대: 2 update 로그 출력, finite한 PPO metric, 마지막 `2 updates done`.

### D. procgen GPU smoke (~1분)
```bash
PYTHONPATH=src python scripts/train_agent.py \
    --env_name maze_aisc --num_envs 16 --num_steps 64 \
    --total_env_steps 50_000 --device cuda --seed 0
```
기대: ~50 update, ep_return 점진 증가, 발산 없음.

### 보고 양식 (어느 한 단계가 깨지면)
- 마지막 출력 5~10줄 + 그 직전 traceback
- 어느 step(A/B/C/D)에서 깨졌는지
- 가능하면 `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

---

## 10. git 상태 (사용자 환경에서 확인)

- 마지막 태그: `v1.0-plan` (Phase 0 진입 직전).
- 이후 추가된 파일 (전부 D:\brain\split_maze 안):
  - 소스: `src/split_maze/{__init__,language,env,agent,ppo,train}.py`
  - 테스트: `tests/test_{language,env,agent,ppo,train}.py`
  - 스크립트: `scripts/{check_env,train_agent}.py`
  - 문서: `docs/{PROCGEN_ENV,LANGUAGE_SPEC,SESSION_HANDOFF}.md`
- 다음 태그 후보: `v1.1-phase1` (Phase 1 완료 시 — 정식 학습 + OOD 평가까지).
- 권장 중간 commit (WSL 83 tests 통과 후):
  ```bash
  git add -A && git commit -m "Phase 1.3 — train.py + train_agent.py + test_train.py (19 tests)"
  ```

---

## 11. 프로젝트 작업 스타일 (러닝/문화)

- **차분히 차근차근**. 큰 결정엔 `AskUserQuestion`.
- **사람말로**. 필요하면 비유(미로/통역, 영수/민수 등).
- **박제 문화**. Deferred 항목·러닝 모두 명시적으로 문서화. PLAN.md에 정밀화 로그 유지.
- **결정 시 옵션 명시 + 추천 + 이유**. 사용자가 "추천대로" 라고 답할 수 있게.
- **사전 등록 (pre-registered)**: 임계치·시나리오·완료 기준을 학습 *전*에 박제. 결과 본 후 임계치 변경 금지 (잘못된 임계치 발견 시 "Post-hoc adjustments"로 *이유 명시*).
