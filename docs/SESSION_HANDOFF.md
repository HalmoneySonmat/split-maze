# SPLIT-MAZE — 세션 핸드오프 (2026-05-15)

> Cowork 샌드박스 VM이 wedge되어 새 세션이 필요한 시점의 상태 박제.
> 새 세션 시작 시 본 문서 + PLAN.md + docs/PROCGEN_ENV.md + docs/LANGUAGE_SPEC.md
> 4개 읽으면 컨텍스트 95% 복원.

---

## 1. 한 줄 현재 위치

**Phase 1.3 진입 직전.** Phase 0 + Phase 1.1(IMPALA-CNN) + Phase 1.2(PPO)
완료, 사용자 WSL에서 **41 + 10 + 13 = 64 tests pass**. 다음:
`scripts/train_agent.py` (PPO 학습 루프).

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
지금 통과해야 할 테스트 수: **64** (test_language=30, test_env=11, test_agent=10, test_ppo=13).

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

### `tests/`
- `test_language.py`, `test_env.py`, `test_agent.py`, `test_ppo.py`
- 마지막 WSL 결과: 모두 통과. `test_env.py`의 procgen 통합 테스트는 WSL에서만 활성화됨.

### `scripts/`
- `check_env.py` — Phase 0 #69 산출물. 무작위 정책 롤아웃 + describer oracle 동작 데모. WSL에서 `--env_name maze_aisc --steps 50` 50/50 문장 생성 확인.

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
| **1.3 train_agent.py** | ← **다음** | PPO 학습 루프 (procgen vec-env + 에이전트 + 버퍼) |
| 1.4 WSL 정식 학습 | 대기 | in-dist 성공률 ≥ 80% (중간 막대) |
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

→ 새 세션에서 sandbox에 torch 깔자마자 **모든 PyTorch 변경을 직접 돌려본 뒤** 사용자한테 넘긴다. AST·산수만으론 PyTorch 버그를 못 잡는다.

---

## 9. 다음 단계 — Phase 1.3 (#72) train_agent.py spec

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

---

## 10. git 상태 (사용자 환경에서 확인)

- 마지막 태그: `v1.0-plan` (Phase 0 진입 직전).
- 이후 추가된 파일 (전부 D:\brain\split_maze 안):
  - 소스: `src/split_maze/{__init__,language,env,agent,ppo}.py`
  - 테스트: `tests/test_{language,env,agent,ppo}.py`
  - 스크립트: `scripts/check_env.py`
  - 문서: `docs/{PROCGEN_ENV,LANGUAGE_SPEC,SESSION_HANDOFF}.md`
- 다음 태그 후보: `v1.1-phase1` (Phase 1 완료 시).
- 새 세션 시작 전 권장: `git add -A && git commit -m "Phase 0+1.1+1.2 — agent + PPO 핸드오프 시점"`

---

## 11. 프로젝트 작업 스타일 (러닝/문화)

- **차분히 차근차근**. 큰 결정엔 `AskUserQuestion`.
- **사람말로**. 필요하면 비유(미로/통역, 영수/민수 등).
- **박제 문화**. Deferred 항목·러닝 모두 명시적으로 문서화. PLAN.md에 정밀화 로그 유지.
- **결정 시 옵션 명시 + 추천 + 이유**. 사용자가 "추천대로" 라고 답할 수 있게.
- **사전 등록 (pre-registered)**: 임계치·시나리오·완료 기준을 학습 *전*에 박제. 결과 본 후 임계치 변경 금지 (잘못된 임계치 발견 시 "Post-hoc adjustments"로 *이유 명시*).
