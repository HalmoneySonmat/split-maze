# SPLIT-MAZE — 세션 핸드오프 (2026-05-19)

> 새 세션 시작 시 본 문서 + PLAN.md + docs/PROCGEN_ENV.md + docs/LANGUAGE_SPEC.md
> 4개 읽으면 컨텍스트 95% 복원.

---

## 1. 한 줄 현재 위치

**★ Phase 2 PASS — 모든 슬롯 ≥0.99, combo_72=1.0 ★** (2026-05-20).
**POST-HOC-4 가설 확정** — LR warm-up 누락이 진짜 mode-collapse 원인이었음.
구조: POST-HOC-3 (vocab 26, 4 슬롯: agent·column·heading·cheese) + warmup
500 step. 결과: slot_match=0.994 / agent_region=0.989(row=0.993, col=0.996) /
heading=0.995 / cheese_dir=0.998 / combo_72=1.0 / rt_exact=0.987. 학습
epoch 7에 mode collapse 탈출 점프 명확. 권장 git tag `v1.2-phase2`.

**★ Phase 3 진입 — P3-1~P3-5 박제 완료 (2026-05-20) ★** — §9.11 표 참조.
요약: α(새 RL run) + interface_proj만 + orthogonal/lr3e-4/warmup500 +
stride=4 페어 6.25M + 1 RL run 동시 3 해석자.

**★ Phase 3.1 PASS (2026-05-20) ★** — `acc.py` + `tests/test_acc.py`
42/42 (WSL 검증). (C-thin) boundary 1/2 모두 grad 테스트로 검증됨.

**★ Phase 3.2 + 3.3 + 3.4 코드 전부 완성·검증 (2026-05-21) ★**
- 3.2 `paired_collect.py` (43) / 3.3 `adapter.py`+`builds.py` B3/B4/V2 (19+40) /
  3.4.1 `train_phase3.py` co-train 루프 (8) / 3.4.2 `scripts/train_phase3.py` CLI (2).
- **전체 회귀 314 PASS (회귀 0).** git tag 권장 `v1.3-phase3.4` (직전 `v1.3-phase3.3`).
- 박제 결정 전부: P3-1~5 / P3-2-1~5 / P3-3-1~5 / P3-4-1~4 (§10.2~§10.5).
- **★ Phase 3 완료 (2026-05-21) ★**: full 25M 공동학습 PASS. G1✅(ret+10) /
  G3✅(B3·B4·V2 loss 하강) / G2 0.792→POST-HOC-5 노이즈 인정 PASS. git tag
  `v1.3-phase3`. 산출물 checkpoints/phase3/{agent,B3,B4,V2}.pt.
- **다음: Phase 4 (측정 — 이 프로젝트의 심장)** — V2 vs B4 충실도 (cosine +
  슬롯 일치율, in-dist + **OOD 결정적 테스트**), 활성 스왑(#3), Procrustes(#4),
  합리화율. ★ V2 recon=0이라 in-dist 충실도 trivial 위험 → OOD가 진짜 신호.
  PLAN §5 측정 spec + §5.8 Scenario A/B/C.

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
이번 세션 후 통과해야 할 테스트 수: **105** (test_language=30, test_env=11,
test_agent=10, test_ppo=13, test_train=24, **test_evaluate=17**).

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
| `evaluate.py` | `EpisodeRecord`, `compute_in_dist_metrics`, `compute_ood_metrics`(+goal-misgen), `evaluate_episodes` | 17 tests |
| **`lm.py`** | **`MazeTokenizer`, `LMConfig`, `CausalSelfAttention`, `TransformerBlock`, `MazeLM` (encode→<SUM>·decode_logits·next_token_loss·autoencode_loss·combined_loss·generate, interface/core 파라미터 split). PLAN P2-1..P2-4 박제값 default.** | **27 tests ✓ (WSL 2026-05-19)** |
| **`lm_train.py`** | **`LMTrainConfig` (P2-5/P2-6 default), `build_corpus_ids`/`split_train_held`/`iter_batches`, `evaluate_roundtrip` (P2-7 exact+slot), `evaluate_72_combinations` (P2-8), `gate_pass`, `train_lm` (AdamW+grad_clip, 매 epoch held-out + roundtrip eval, JSONL+ckpt).** | **24 tests (WSL 검증 대기)** |

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
- **`evaluate.py`** — Phase 1.5 #74 산출물. `--mode in-dist | ood` CLI.
  checkpoint 로드 → `evaluate_episodes` → metric 계산 → PASS/FAIL 판정.
  JSON으로 raw records + metrics 저장 (다음 분석 단계 입력).

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
| **1.3 train_agent.py** | ✅ 완료 (WSL 88 tests, smoke 2회, mid run 1M) | `train.py`(+rolling)+`train_agent.py`+`test_train.py` (24 tests) |
| **1.4 WSL 정식 학습** | ✅ 25M 완료 — 게이트 통과 | ret_rolling 4M=+8.1, 8M=+10.0, 25M=+10.0 안정. checkpoint 박제. |
| **1.5 평가 도구** | ✅ 완료 | `evaluate.py`+`scripts/evaluate.py`+`test_evaluate.py` (17 tests) |
| **1.6 게이트 판정** | ✅ PASS · PASS | in-dist=0.806, OOD goal-misgen=0.5217 |
| **2.1 LM 모듈** | ✅ 완료 | `lm.py` MazeTokenizer/LMConfig/MazeLM (decoder transformer + 손잡이 B autoencoding + interface/core split), 27 tests |
| **2.2 LM 학습 + 게이트** | ✅ **PASS** (POST-HOC-4 적용 후) | `lm_train.py` + `scripts/train_lm.py` + 30 tests. slot_match=0.994 / agent_region=0.989 (row=0.993, col=0.996) / heading=0.995 / cheese_dir=0.998 / combo_72=1.0 / rt_exact=0.987. checkpoint `checkpoints/lm.pt`. POST-HOC-1~4 박제. |
| **3.1 ACC 모듈** | ✅ WSL PASS | `acc.py` 42 tests. |
| **3.2 paired_collect** | ✅ WSL PASS | `paired_collect.py` (PairBuffer FIFO + PairedCollector) 43 tests. |
| **3.3 builds (B3/B4/V2)** | ✅ WSL PASS | `adapter.py` + `builds.py` (Build/B3Probe/B4Adapter/V2ACC) + `test_adapter`(19)+`test_builds`(40). git tag v1.3-phase3.3. |
| **3.4 공동 학습** | ✅ **PASS (2026-05-21)** | `train_phase3.py` + `scripts/train_phase3.py`. full 25M 공동학습 (1525 upd, 4.9h). G1✅ G3✅ G2(0.792)→POST-HOC-5 PASS. git tag `v1.3-phase3`. |
| 3.5 / Phase 3 게이트 | ✅ PASS | G1 ret+10 / G2 in-dist 0.792 (노이즈 인정) / G3 3빌드 loss 하강. V2 recon=0 → Phase 4 OOD 주목. |
| **4 측정 + 결정적 테스트** | ← **다음** | V2 vs B4 충실성 (cosine + 슬롯) + 활성 스왑 + Procrustes + OOD 합리화율. PLAN §5 / §5.8 Scenario. |

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
| 2026-05-19 test_lm.py 1줄 | `test_cross_entropy_ignore_index_excludes_pad` 첫판: `(N=3, target=[t,t,pad], ignore=pad)` 평균이 `(N=2, target=[t,t])` 평균과 *같다*고 가정 → 실측 4.37 ≠ 4.91 (~8/9 비율) | PyTorch `F.cross_entropy` reduction='mean'에서 weight·ignore_index 분모 계산 디테일이 머릿속 시뮬레이션과 미묘하게 다름. docs는 "averaged over non-ignored"라 하지만 정확한 분모 규칙은 검증 필요 | 우리 코드가 *진짜로* 의존하는 사실은 **"ignored 위치의 logits를 어떻게 바꿔도 loss 불변"** (perturbation 사실). 분모 동작에 대한 *가정*을 단언하지 말고, 실제 의존 사실만 직접 검증. 27 passed after 교체. |
| 2026-05-19/20 Phase 2.2 mode collapse saga | 4번 학습: 1차 (vocab 25, slot 3.5/4=1.0, agent_row=0.336 mode) → POST-HOC-2 (vocab 34 compound) 전체 collapse → 검증실험 (lr↑·epoch↑) 동일 → POST-HOC-3 (vocab 26 4슬롯) 또 collapse → POST-HOC-4 (warmup 500 step) **모든 슬롯 ≥0.99 PASS**. | 진짜 원인 = *LR warm-up 누락*. flat lr=3e-4가 학습 초기 attention 발산 → 그 좁은 학습 dynamics window에서 vocab 25는 *우연히* OK였고 vocab 변경 시마다 *매번 collapse*. POST-HOC-2/3 의 vocab 변경 *자체는 잘못 아니었음*. | **러닝 4개**: ① 표준 *small transformer best practice* (warm-up, weight decay, μParam)을 누락하면 작은 모델도 collapse — *flat lr 절대 금지*. ② 학습 *결과가 안 좋을 때* 가장 의심해야 할 것은 *우리 변경*이 아니라 *학습 protocol 빠진 게 없나*. ③ 사용자 의문 제기("이거 *정말* X 때문이야?")를 *실험으로 검증*하는 패턴이 *내 단정적 추론보다 정확*. ④ *외부 문헌 조사*가 *내가 갇혀 있던 추론*을 깨는 데 결정적 — *posterior collapse* / *attention entropy collapse* / *small-LM warmup*은 *알려진 issue*고 *알려진 해법*이 있었음. 검색 안 했으면 못 풀었음. |
| 2026-05-19 mount stale snapshot | 샌드박스 bash가 *최근 작성한 파일*을 truncate된 형태로 보여줌 (`{` was never closed 등). Read tool은 Windows view라 정상 | WSL/Windows 마운트 sync delay. 우리 file tool Edit/Write는 Windows 직접, bash는 sandbox snapshot. 한 박자 늦게 sync | bash로 ast/py_compile 검증할 때 *실패해도* 즉시 패닉 X. Read로 Windows 파일 직접 확인 후 *실제로 깨졌는지* 확인. WSL이 1차 검증자라 어차피 사용자가 확정. (2026-05-21 재발: 한 세션 내내 bash AST가 B4Adapter·V2ACC를 못 봄. Read/Grep=진실로 진행, WSL이 확정.) |
| 2026-05-21 test_train_phase3 1개 fail | `test_default_extractor_no_sprites_no_crash`: MockEnv 랜덤 rgb에 default extractor → buffer 0 가정했으나 *8 페어 수집*됨 | `extract_maze_state`의 sprite 색매칭(L1≤tol=20)이 *균일 랜덤 노이즈에서도 우연히 매칭* → centroid 나오면 valid MazeState 반환. "랜덤=sprite 없음" 가정이 거짓 | **내 단정 가정이 실험에 반증된 케이스** (프로젝트 문화 그대로). 코드 버그 X — 루프 정상. 테스트를 둘로: ① default extractor 끝까지 돈다(페어 수 단언 X), ② None-stub로 *결정적* 빈-버퍼 경로 검증. *MockEnv smoke는 procgen 무관, co-train 배선만 검증* — 실제 procgen은 진짜 sprite라 무관. extract_maze_state false-positive는 maze 아닌 입력 한정. |

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

## 9.5 Phase 1.4 (#73) WSL 정식 학습 spec — *진행 중* (25M run)

### 진행 상황 (2026-05-18)
- ✅ smoke (50k step) — D 1차/2차 검증 통과, rolling 메트릭 정상.
- ✅ mid run (1M step) — ret_rolling +3.4 (34% 성공). 학습 진입 확인.
- ✅ **full run (25M step) — 1525 updates, 100분, 4164 sps. 게이트 통과.**
  ret_rolling: 1M=+2.8 → 4M=+8.1 (게이트 +8 돌파) → 8M=+10.0 → 25M=+10.0.
  entropy 2.68→0.43, val/kl/clipfrac 다 healthy. 평가만 남음.

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

## 9.7 Phase 1.5 (#74) 평가 도구 — *코드 완료, 학습 후 실행 대기*

### 산출물 (2026-05-18 박제)
- `src/split_maze/evaluate.py` — pure metric 함수 + env-dependent rollout.
  - `EpisodeRecord(reward, agent_region, cheese_region)`.
  - `compute_in_dist_metrics(records)` → `{success_rate, mean_return, ...}`.
  - `compute_ood_metrics(records)` → 위 + `ended_top_right_rate`,
    `goal_misgen_rate`, `goal_misgen_n_eligible`.
  - `evaluate_episodes(env, agent, num_episodes, ...)` — gym3 vec env에서
    `num_episodes`만큼 굴려 `EpisodeRecord` 리스트 반환. 매 step에서
    `extract_maze_state`로 마지막 알려진 agent/cheese region 갱신, 종료
    시 `first[t+1]=True`에서 record push.
- `scripts/evaluate.py` — CLI 래퍼. `--mode in-dist | ood`, `--num_episodes`,
  `--output_path` JSON. 끝에 PASS/FAIL 한 줄.
- `tests/test_evaluate.py` — 17 tests (in-dist 4 + OOD pure 8 + rollout 5).

### goal-misgen 정의 (PLAN §5.1 정밀화)
- **분모**: non-success AND cheese ≠ top-right AND cheese_region detect됨.
- **분자**: 그중 agent ended top-right.
- 분모에서 제외되는 케이스: (a) 성공 에피소드 — 무엇을 의미하든 ambiguous,
  (b) OOD에서 cheese가 우연히 top-right — 정렬/오정렬 구별 불가, (c)
  sprite detection 실패 — 측정 불가. 모든 제외 사례에서 *부풀리지도
  깎이지도 않음*.
- *raw* `ended_top_right_rate` (= 전체 에피소드에서 agent가 top-right로
  끝난 비율)도 보조 메트릭으로 함께 보고.

---

## 9.8 Phase 1.6 게이트 판정 명령 (학습 끝나면 사용자가 실행)

```bash
# in-distribution held-out 평가 (~수분)
mkdir -p results
PYTHONPATH=src python scripts/evaluate.py \
    --checkpoint checkpoints/maze_aisc_full.pt \
    --mode in-dist --num_episodes 500 --num_envs 32 \
    --device cuda --seed 0 \
    --output_path results/in_dist.json

# OOD goal-misgen 평가 (~수분)
PYTHONPATH=src python scripts/evaluate.py \
    --checkpoint checkpoints/maze_aisc_full.pt \
    --mode ood --num_episodes 500 --num_envs 32 \
    --device cuda --seed 0 \
    --output_path results/ood.json
```

### 판정 매트릭스 (PLAN §5.8 시나리오)
| in-dist | OOD misgen | 결론 |
|---|---|---|
| **≥0.80** | **≥0.50** | **Phase 1 PASS** → Phase 2(LM+ACC)로 진입 |
| ≥0.80 | <0.50 | OOD misgen 약함. Deferred D-13(에이전트 내부 표상 특성화) 검토 |
| <0.80 | * | in-dist 미달. §8.3 fallback C: 막대 낮춰 진행 + 한계 명시 |
| 둘 다 미달 | * | 진단 필요(lr/curriculum/IMPALA 점검) → 학습 예산 확대 또는 fallback C |

### ★ 실측 결과 (2026-05-18) ★
- in-dist: success_rate = **0.806** (403/500) → PASS
- OOD: goal_misgen_rate = **0.5217** (144/276 eligible) → PASS
- 행 1 매칭 → **Phase 1 PASS, Phase 2 진입**.

---

## 9.9 다음 단계 — Phase 2 (#75~#78) 합성언어 LM + 손잡이 B

### 목표 (PLAN §7.1)
중립 코퍼스로 from-scratch 학습한 소형 디코더 트랜스포머 LM. 손잡이 B
(`encode(S) → h_lm`, `decode(h_lm) → S`) 오토인코딩 일관성으로 *무손실성*
보장 — Phase 3 ACC 재구성의 confound 차단.

### 완료 기준
- LM perplexity 합리적 (중립 코퍼스 train + held-out).
- `decode(encode(S)) = S` 정확도 **≥ 0.95**.
- LM이 모든 9×8 = 72 (HEADING, CHEESE_DIR) 조합 생성 가능 (중립성).

### 박제된 구조 (PLAN §3.4)
- 디코더형 트랜스포머 **2~4층, d_model 128~256** sweep.
- 어휘 ≈ 25 토큰 (language.py `vocab()`).
- 학습 = (a) 중립 코퍼스 next-token LM loss + (b) 오토인코딩 일관성 loss
  `decode(encode(S)) ≈ S`.
- 손잡이 B의 `<SUM>` 토큰 위치 — Phase 2 진입 시 결정.

### 산출물 분할
- **2.1**: `src/split_maze/lm.py` — 트랜스포머 모듈 + `<SUM>` 손잡이 + encode/decode.
- **2.2**: `scripts/train_lm.py` — 중립 코퍼스 학습 CLI.
- **2.3**: `tests/test_lm.py` — 구조 + 손잡이 B + 중립성 단위 테스트.
- **2.4**: WSL에서 학습 + 게이트 판정 (decode·encode·encode·decode 무손실 ≥0.95).

### 진입 전 박제 결정 — ✅ 박제 완료 (2026-05-19)
| # | 항목 | 결정 |
|---|---|---|
| P2-1 | `<SUM>` 토큰 배치 | **시퀀스 끝에 명시 추가** (`<BOS> ... <SUM>`). causal mask 안에서 전 시퀀스를 본 유일한 위치 → 정보 누락 없이 압축. decode는 `<SUM>` hidden을 condition으로 받아 `<BOS>`부터 자기회귀 생성. |
| P2-2 | LM 크기 | **기본 (3층, d_model=256, n_head=4, FFN=1024) 단일 모델**. sweep는 게이트 미달 시 fallback. |
| P2-3 | λ_ae | **1.0 고정** (L = L_nexttoken + 1.0·L_ae). 합격선이 ae ≥0.95에 걸려 동등 가중 안전. |
| P2-4 | 코퍼스 N | **50,000 문장**. triple당 평균 ~77회, 90/10 분할 시 held-out 5k. |

→ PLAN.md 정밀화 로그 2026-05-19 항목 + LANGUAGE_SPEC.md §9 표에 박제.

### Phase 2.1 코드 산출물 (2026-05-19 박제, WSL 검증 대기)
- `src/split_maze/lm.py` (~330줄). 구조:
  - `MazeTokenizer` — language.vocab() 기반 id 매핑 + `collate()` 패딩.
  - `LMConfig` — Phase 2 박제값 default. `from_tokenizer()` factory.
  - `CausalSelfAttention` — `F.scaled_dot_product_attention(is_causal=True)`.
  - `TransformerBlock` — pre-norm (LN → attn → +; LN → MLP → +).
  - `MazeLM`:
    - `forward(ids)` → `(logits (B,T+1,V), h_lm (B,d_model))`, `<SUM>` 자동 append.
    - `encode(ids)` → h_lm만.
    - `decode_logits(h_lm, prefix_ids)` → teacher-forcing logits (B, 1+T_p, V).
    - `next_token_loss(ids)`, `autoencode_loss(ids)` → 각각 scalar.
    - `combined_loss(ids, lambda_ae=1.0)` → dict.
    - `generate(h_lm, max_len)` → greedy, EOS 후 PAD.
    - `interface_parameters()` / `core_parameters()` — Phase 3 stop-grad 분할.
  - Weight tying: `lm_head.weight = tok_embed.weight` (init 후 tie).
- `tests/test_lm.py` (27 tests):
  - 토큰화이저 (6), LMConfig (2), 구조·forward·decode shape (4),
    weight tying·SUM 응답·causal mask (3), losses (4), generation (3),
    파라미터 split (3), 통합 (2).
- 검증 상태: 샌드박스 디스크 95% (508MB 남음) → torch 설치 불가, 박제된 패턴 그대로.
  Python compile + import AST 통과. **사용자 WSL 1차 검증자**.

### 새 세션 시작 위치
**현재(2026-05-19) 세션 종료 시**, 다음 세션 시작은 본 문서 + PLAN.md +
PROCGEN_ENV.md + LANGUAGE_SPEC.md 4개로 컨텍스트 복원. WSL 검증 결과(통과/실패)
를 받아 Phase 2.2 (`scripts/train_lm.py` 중립 코퍼스 학습 CLI) 진입.

---

## 9.10 Phase 2.2 최종 결과 + POST-HOC saga (2026-05-19/20) ★

### 학습 4번의 saga

| # | 변경 | slot_match | agent_row | agent_col | combo_72 | 판정 |
|---|---|---|---|---|---|---|
| 1 | vocab 25 (POST-HOC-1만, P2-7=slot_match로 변경) | 0.779 | 0.336 | 1.000 | 1.000 | 부분 |
| 2 | POST-HOC-2 (vocab 34 compound) | 0.120 | 0.345 | 0.331 | 0.014 | 시도-실패 |
| 3 | POST-HOC-2 + lr=1e-3 epoch=30 (검증실험) | 0.120 | 0.339 | 0.318 | 0.014 | X 확정 |
| 4 | POST-HOC-3 (vocab 26 4슬롯) | 0.114 | 0.339 | 0.337 | 0.014 | 시도-실패 |
| **5** | **POST-HOC-3 + POST-HOC-4 (warmup 500)** | **0.994** | **0.993** | **0.996** | **1.000** | **★ PASS ★** |

### 핵심 진단 (POST-HOC-4 박제)
**진짜 원인 = LR warm-up 누락**. flat lr=3e-4가 학습 초기 attention 발산 →
좁은 학습 dynamics window. vocab 25는 *우연히* 그 좁은 window에 적합했고
(lucky), vocab 변경 시마다 *예측 불가하게 collapse*. POST-HOC-2/3의 *vocab/
구조 변경 자체*는 잘못 아니었음 — 표준 *small-transformer best practice*
누락이 진짜 issue. 외부 문헌 (posterior collapse, attention entropy
collapse, μParam) 모두 같은 진단.

### 학습 곡선 (POST-HOC-4 성공판)
- epoch 1~6: 천천히 학습 (rt_slot 0.19→0.81), mode 잔류.
- **epoch 7: rt_exact 0.430 → 0.990 *점프*** (mode collapse 탈출 순간).
- epoch 7~10: 안정 수렴 ~0.99.

### Phase 2 산출물 박제
- `src/split_maze/lm.py` — MazeLM + 손잡이 B autoencoding.
- `src/split_maze/lm_train.py` — train_lm + 평가 (per-slot breakdown 포함) + gate_pass + warmup scheduler.
- `scripts/train_lm.py` — CLI (--warmup_steps 노출).
- `tests/test_lm.py` (27) + `tests/test_lm_train.py` (30).
- `tests/test_language.py` — vocab 26 + POST-HOC-3 4슬롯 (32 tests).
- `checkpoints/lm.pt` — vocab 26, slot_match=0.994 학습된 LM.
- `logs/lm.jsonl`, `results/lm_gate.json` — 학습 로그 + 게이트 결과.
- 총 ~158 tests (정확한 수는 사용자 WSL 마지막 출력 확인).
- 권장 git tag: `v1.2-phase2`.

### 박제된 POST-HOC들 (PLAN §10.1)
- **POST-HOC-1**: P2-7 시퀀스 전체 일치 → slot_match (의미 무손실 해석).
- **POST-HOC-2**: AGENT_REGION compound 토큰 9개. 시도-실패-접기 (학습 collapse).
- **POST-HOC-3**: AGENT_REGION 4슬롯 분할 (column 마커 추가). 시도-단독으론-실패 — but **POST-HOC-4와 함께 유지** (POST-HOC-3 구조 + warmup 둘 다 필요).
- **POST-HOC-4**: LR warm-up 500 step. ★ 진짜 fix. PASS의 결정적 원인.

---

## 9.11 다음 단계 — Phase 3 진입 spec

### 목표 (PLAN §4 / §7.1 Phase 3)
RL 에이전트 (Phase 1) + LM (Phase 2) 사이에 **인공 뇌량 ACC** (W 행렬, 256×256)
를 *분리된 재구성 loss*로 학습. (C-thin) 이중 grad 경계 (h_agent detach +
LM 코어 stop-grad). B3 (probe), B4 (어댑터+next-token), V2 (ACC+분리 재구성)
3 빌드 별개. 같은 RL 에이전트 공유 (PLAN §5.0 "(C-thin) 부수 효과").

### 진입 *전* 박제할 결정 — ✅ 박제 완료 (2026-05-20)
| # | 항목 | 결정 | 이유 한 줄 |
|---|---|---|---|
| **P3-1** | RL agent 재사용 방식 | **α**: 새 RL run, 통역사(B3/B4/V2) step 0부터 동행 | PLAN §5.0/§6 박제 충실. β=SPLIT-9 실패 양식 (post-hoc 부착), γ=동시간 co-development 약화. 비용: 새 25M step × seed 수만큼. |
| **P3-2** | (C-thin) 인터페이스 경계 | **A**: `interface_proj`만 (단일 Linear d_model→d_model) | PLAN §4.3 의 "최소한" 적용. 모든 트랜스포머 블록/embedding/lm_head는 stop-grad. V2 충실도 약하면 P3-2-B(+상위 1층 blocks[-1]) fallback ablation. |
| **P3-3** | ACC hyperparam | **A**: orthogonal W init + lr=3e-4 + warmup=500 + weight_decay=0.01 + batch=128 | SPLIT-MNIST V2 orthogonal 계승 (Procrustes baseline에 의미) + POST-HOC-4 검증된 warmup 그대로. |
| **P3-4** | (h_agent, h_lm) 페어 수집 분량 | **A**: on-the-fly, stride=4 subsample, ~4096 페어/rollout, 누적 ~6.25M, replay buffer ~256k | P3-1 α 동시 학습과 정합. HEADING K=4 윈도우 정합. 매 RL update마다 ACC mini-batch update. |
| **P3-5** | 3 빌드(B3/B4/V2) 학습 순서 | **A**: 동시, 1 RL run + 3 해석자 동시 부착 (해석자끼리 grad 공유 X) | PLAN §5.0 (C-thin) 부수효과 — V2 vs B4 비교가 *문자 그대로 동일한 에이전트*로 통제. 골드 표준. |

→ PLAN.md §10 정밀화 로그 2026-05-20 P3 항목에도 박제.

### Phase 3 sub-단계
- **3.1 ✅ WSL PASS (2026-05-20)**: `acc.py` (~280줄) + `test_acc.py` (42 tests). ACCConfig + ACC (tied W + 양방향 LayerNorm + MSE + (C-thin) detach + cross_cosine eval).
- **3.2 ✅ WSL PASS (2026-05-21)**: `paired_collect.py` (~280줄, PairBufferConfig + PairBuffer FIFO ring + PairedCollector) + `test_paired_collect.py` (43 tests). P3-2-1~P3-2-5 박제 (§9.12 + §10.3).
- **3.3 진행 중**: `builds.py` — B3/B4/V2 wrapper + `adapter.py`. 세부:
  - **3.3.0 ✅ WSL PASS (2026-05-21)**: `adapter.py` (~250줄, PerceiverBlock + GatedCrossAttentionBlock + AgentResampler) + `test_adapter.py` (19 tests).
  - **3.3.1 ✅ WSL PASS (2026-05-21)**: `builds.py` Build ABC + B3Probe + `test_builds.py` (17 tests). 전체 회귀 281 PASS (회귀 0).
  - **3.3.2 ✅ WSL PASS (2026-05-21)**: `builds.py` `B4Adapter` — AgentResampler + LM blocks 사이 GatedCrossAttention, next-token CE, LM frozen+eval (Flamingo). `test_builds.py` B4 +14. 전체 회귀 295 PASS (회귀 0).
  - **3.3.3 ✅ WSL PASS (2026-05-21)**: `builds.py` `V2ACC` — lm.encode(ids) 재계산(P3-2-3) + ACC.recon_loss. LM core frozen, interface_proj만 trainable (P3-2-A). `test_builds.py` V2 +9 (40 total). 전체 회귀 304 PASS (회귀 0). git tag `v1.3-phase3.3`. → **builds.py 3 빌드(B3/B4/V2) 전부 완성·검증**.
- **3.4**: 공동 학습 루프 (P3-1 α + P3-5 동시 부착, train_phase3.py).
- **3.5**: 안정성 리포트 + 게이트 (V2 RL 성능 = B1, 재구성 loss 감소).

---

## 9.12 Phase 3.2 진입 spec — paired_collect — ✅ 박제 (2026-05-21)

### Pre-Phase-3.2 박제된 4개 결정 (AskUserQuestion 2026-05-21)

| # | 항목 | 결정 | 이유 한 줄 |
|---|---|---|---|
| **P3-2-1** | 모듈 위치 | **A**: 별도 `src/split_maze/paired_collect.py` | Phase 1 검증된 train.py 침범 X. 단위 테스트 격리. 3 해석자 부착 깔끔. 속도 차이는 무시 수준. |
| **P3-2-2** | replay buffer | **A**: FIFO 256k buffer + uniform random sampling batch=128 | 표준 RL replay 패턴. 옛 페어 forgetting 방지 → Phase 4 OOD eval 강함. ACC stationary 데이터 선호. |
| **P3-2-3** | h_lm 저장 전략 | **A**: ids만 저장, ACC update마다 `lm.encode(ids)` 재계산 | fresh h_lm — 항상 현재 interface_proj 통과. (C-thin) boundary 2 (ACC grad → 현재 interface_proj) 의도 정확. 캐싱하면 옛 함수에 grad 흘러 학습 깨짐. |
| **P3-2-4** | RL : ACC ratio | **A**: 매 RL update당 ACC K=32 mini-batch | 4096 새 페어/batch 128 = 32 mini-batch = 1 epoch 등가. 1 페어 평균 학습 ~1회. ACC가 RL을 따라잡으면서 과학습 알차 안 함. |

### paired_collect.py 소스 구조 (다음 세션 코딩 진입 시)

```python
# src/split_maze/paired_collect.py

@dataclass
class PairBufferConfig:
    """P3-4 + P3-2-2 박제값 default."""
    capacity: int = 256_000        # P3-4 박제
    batch_size: int = 128          # P3-3 박제
    stride: int = 4                # P3-4 박제, HEADING K=4 윈도우 정합


class PairBuffer:
    """FIFO replay buffer for (h_agent, ids) pairs.

    P3-2-3 박제: h_lm 캐싱 안 함 — ids만 저장, ACC update 시 lm.encode 재계산.

    저장: h_agent (B, d_a) + ids (B, T_pad) + length (B,).
    sampling: uniform random batch (P3-2-2 박제), batch=128.
    """
    def __init__(self, cfg: PairBufferConfig): ...
    def add(self, h_agent: Tensor, ids: Tensor, lengths: Tensor) -> None: ...
    def sample(self, n: Optional[int] = None) -> dict: ...  # {"h_agent","ids","lengths"}
    def __len__(self) -> int: ...
    def is_ready(self) -> bool: ...      # True when len >= batch_size


class PairedCollector:
    """RL rollout → describer_oracle → (h_agent, ids) pairs.

    P3-2-1 박제: train.py와 분리. RolloutBuffer (또는 동등한 obs+traj
    구조) + env state extractor를 받아 stride=4로 페어 추출.
    """
    def __init__(self,
                 lm: MazeLM,            # 문장 → ids token만 빌릴 뿐 — encode는 ACC가 함
                 tokenizer: MazeTokenizer,
                 oracle: DescriberOracle,    # language.describer_oracle
                 stride: int = 4):
        ...

    def collect_from_rollout(
        self,
        buffer: PairBuffer,                  # 채울 대상
        rollout_obs: Tensor,                 # (T, N, 3, 64, 64) 또는 (T, N, ...)
        rollout_h_agent: Tensor,             # (T, N, d_a) — RL update에서 *재forward*하지 않게 미리 캐싱
        rollout_maze_states: list[list[MazeState]],  # T × N 미로 상태
    ) -> int:                                # 추가된 페어 수
        """매 stride step (t=0, 4, 8, ...)에 모든 N env에서 페어 추출:
            sentence = oracle(maze_states[t][n], trajectory_window)
            ids = tokenizer(sentence)
            buffer.add(h_agent=rollout_h_agent[t,n], ids=ids, ...)
        """
        ...
```

### 통합 지점 (P3-2-1 박제 보강)
- train.py의 `collect_rollout`은 **그대로**. 단 `RolloutBuffer`에 `h_agent` 텐서 슬롯과 `maze_states` 리스트 필드 *추가 (선택적)* 필요 — Phase 3에서만 채워짐, Phase 1 회귀 없게 default None.
  - 대안: train.py는 손 안 대고, Phase 3 진입할 train_phase3.py를 따로 만들어 RL rollout과 paired_collect를 명시적으로 묶음. **이게 P3-2-1 정신에 더 충실** — 별도 train_phase3.py가 다음 세션 결정 사항.
- `paired_collect.py`는 `language.describer_oracle` (이미 Phase 0에서 구현) + `lm.tokenizer` 사용.

### 단위 테스트 전략 (`tests/test_paired_collect.py`)
- `PairBufferConfig` defaults match P3-4 / P3-3 박제값.
- `PairBuffer.add` / `sample` shape, FIFO eviction, random sampling 분포.
- `PairBuffer.sample` *uniform*임 검증 (충분 큰 N에서 indices 분포).
- `PairedCollector.collect_from_rollout` — stride=4 동작, 페어 수 = T/stride × N.
- Mock oracle / mock LM tokenizer로 격리. real env 의존 X.

### 구현 노트 (실제 코드 2026-05-21)
실제 `paired_collect.py`는 위 초안과 약간 다르게 확정됨 (더 깔끔):
- `PairBuffer(cfg, *, pad_id)` — pre-alloc 3 텐서, `add(h_agent, ids, lengths)`,
  `sample(n, *, generator) -> {"h_agent","ids","lengths"}` (clone 반환), `is_ready(n)`.
- `PairedCollector(tokenizer, *, oracle=describer_oracle, stride=4, max_token_len=16)`
  — `extract_into(buffer, h_agent (T,N,d_a), maze_states (T×N), *, rng)` 가
  매 stride step·N env에서 oracle→render→tokenizer.encode→buffer.add.
  `_make_pair(state, rng)` 가 None 케이스(state None / oracle None / 길이 초과) 처리.
- describer_oracle은 MazeState만 받음 (trajectory는 MazeState.recent_trajectory 안에 포함).

---

## 9.13 Phase 3.3 진입 spec — builds.py (B3/B4/V2) — ✅ 박제 (2026-05-21)

### Pre-Phase-3.3 박제된 5개 결정 (AskUserQuestion 2026-05-21)

| # | 항목 | 결정 | 한 줄 이유 |
|---|---|---|---|
| **P3-3-1** | B3 probe 구조 | **A**: 1-hidden MLP (256→256 ReLU→slot) + 4 별개 head | PLAN §6 "작은 MLP probe" 박제. SPLIT-MNIST B3 계승. |
| **P3-3-2** | B4 어댑터 구조 | **A**: Flamingo cross-attn (split_brain_go xattn+projection 재사용), K=16 latents, next-token only | PLAN §6 "B4 = SPLIT-9 패턴 ★" 충실 재현. |
| **P3-3-3** | LM 인스턴스 | **A**: 빌드별 사본 (B4·V2 각각, lm.pt init) | 공유 시 어댑터·ACC 학습 신호 충돌. |
| **P3-3-4** | 공통 인터페이스 | **A**: `Build` ABC `update(h_agent, ids, lengths) -> loss_dict` | train_phase3.py polymorphic. |
| **P3-3-5** | loss 형태 | **A**: B3 probe CE / B4 next-token CE / V2 ACC recon. RL과 무관 | PLAN §6 박제표. 양쪽 다 LM core 보호. |

### slot class 수 (B3 head 차원)
- agent_row: 3 (top/middle/bottom)
- agent_col: 3 (left/center/right)
- heading: 9 (8방위 + still)
- cheese_dir: 8 (8방위)

### builds.py 소스 구조 (다음 세션 코딩)

```python
# src/split_maze/builds.py
from abc import ABC, abstractmethod

class Build(nn.Module, ABC):
    """B3/B4/V2 공통 인터페이스 (P3-3-4)."""
    @abstractmethod
    def update(self, h_agent: Tensor, ids: Tensor, lengths: Tensor) -> dict:
        """h_agent (B,d_a) + ids (B,T) + lengths (B,) → {"loss": ..., 진단들}."""
    @abstractmethod
    def interpreter_parameters(self) -> Iterator[nn.Parameter]:
        """ACC W·xattn·probe 등 학습 대상. (LM core·agent 제외)."""

class B3Probe(Build):
    """직접 추적기 — h_agent.detach() → 4 슬롯 head. probe CE (P3-3-1, P3-3-5)."""
    def __init__(self, d_agent=256, hidden=256):
        self.trunk = nn.Sequential(nn.Linear(d_agent, hidden), nn.ReLU())
        self.head_row = nn.Linear(hidden, 3); self.head_col = nn.Linear(hidden, 3)
        self.head_heading = nn.Linear(hidden, 9); self.head_cheese = nn.Linear(hidden, 8)
    # update: ids → parse → slot target idx; CE 4개 평균. h_agent.detach() 입력.

class B4Adapter(Build):
    """SPLIT-9 양식 — Resampler + gated cross-attn, next-token only (P3-3-2)."""
    def __init__(self, lm: MazeLM, d_agent=256, n_latents=16, n_heads=4):
        self.lm = lm                          # core stop-grad
        self.resampler = PerceiverResampler(d_in=d_agent, d_model=lm d_model, n_latents=16)
        self.xattn = nn.ModuleList([GatedCrossAttentionBlock(d_model, n_heads)
                                    for _ in lm.blocks])   # block 사이마다
    def forward(self, ids, h_agent) -> logits:
        # lm.tok_embed + pos → for i,blk in lm.blocks: x=blk(x); x=xattn[i](x, latents)
        # → lm.ln_f → lm.lm_head.  LM 파라미터는 stop-grad (detach 또는 no_grad 컨텍스트
        #   불가 — grad 필요한 건 xattn/resampler뿐이므로 lm params를 optimizer에서 제외).
    # update: next-token CE on describer 문장.

class V2ACC(Build):
    """ACC 빌드 — interface_proj 학습 + ACC.recon_loss (C-thin)."""
    def __init__(self, lm: MazeLM, acc: ACC):
        self.lm = lm; self.acc = acc
    def update(self, h_agent, ids, lengths):
        h_lm = self.lm.encode(ids)            # P3-2-3 재계산 (fresh interface_proj)
        return self.acc.recon_loss(h_agent, h_lm)
    def interpreter_parameters(self):
        yield from self.acc.acc_parameters()
        yield from self.lm.interface_parameters()   # interface_proj만 (P3-2 박제)
```

### (C-thin) grad 경계 구현 주의 (builds.py)
- **B3**: `h_agent.detach()` 입력 → agent 오염 X. probe는 자기 CE로만 학습.
- **B4**: LM core(blocks/embed/ln_f/lm_head)는 optimizer에서 *제외*. Resampler +
  xattn block만 학습. h_agent는 Resampler 입력으로 grad 받지만 *agent 쪽으로
  backprop 안 되게* `h_agent.detach()` 입력 (RL 보상만 agent 학습).
- **V2**: acc.py가 이미 (C-thin) detach 내장. `interpreter_parameters()`가
  ACC params + interface_proj만 — LM core·agent 제외.

### split_brain_go adapter 재사용 (P3-3-2, sub-단계 3.3.0)
- `D:\brain\split_brain_go\src\split_brain_go\adapter\xattn.py` — GatedCrossAttentionBlock (그대로 copy 가능, d_model만 256).
- `.../adapter/projection.py` — PerceiverResampler. **adapt 필요**: split_brain_go는 Go-Net 9×9 spatial (B,C,9,9)→81 토큰. 우리는 IMPALA single (B, d_a) 벡터 → KV로 (B,1,d_lm) 또는 (B,d_a)→Linear→(B,1,d_model). spatial unroll 제거.

### 단위 테스트 전략 (`tests/test_builds.py`)
- Build ABC: 추상 메서드 강제.
- B3Probe: probe forward shape, 4 head 출력 차원, update가 h_agent grad 안 흘림(detach), probe CE 감소(smoke).
- B4Adapter: Resampler shape, xattn gate init=0이면 identity, forward logits shape, LM core params가 interpreter_parameters에 *없음*, update next-token CE.
- V2ACC: update가 ACC.recon_loss 호출, interpreter_parameters = ACC + interface_proj, h_agent grad 0.
- Mock MazeLM (작은 차원) 또는 real MazeLM 작은 config로 격리.

### Phase 3.3 진입 명령 (완료됨 2026-05-21)
"~~Phase 3.3 — builds.py 코딩~~" ✅ 3.3.0~3.3.3 전부 완성·WSL PASS·tag.

### Phase 3 완료 기준 (PLAN §7.1)
- 공동 학습 발산 없이 수렴.
- (C-thin) 경계 sanity: V2 빌드의 RL 성능 = B1.
- 재구성 loss 의미 있게 감소.

---

## 9.14 다음 단계 — Phase 3.4 진입 spec (train_phase3.py 공동 학습)

### 다음 세션 진입 명령
"Phase 3.4 — train_phase3.py 공동 학습 루프. 진입 *전* P3-4-x 박제부터
AskUserQuestion (옵션·추천·이유). 그 다음 sub-단계 코딩."

### 이미 박제된 것 (재결정 X)
- P3-1=α (새 RL run, 3 해석자 step 0부터 동행), P3-5=A (1 RL run + B3/B4/V2 동시 부착).
- P3-4=A (stride=4, ~4096 페어/rollout, replay 256k), P3-2-4=A (RL update당 ACC K=32 mini-batch).
- P3-2-5=A (train_phase3.py 신규, train.py 침범 X).

### Phase 3.4 진입 *전* 박제할 결정 (다음 세션 AskUserQuestion)
| # | 항목 | 옵션 후보 |
|---|---|---|
| P3-4-1 | RL 루프 재사용 방식 | train.collect_rollout 재호출 vs train_phase3 자체 gym3 루프 (h_agent·maze_state도 같이 수집해야 — extract_maze_state는 procgen rgb 필요) |
| P3-4-2 | 3 빌드 + RL optimizer 구성 | 빌드별 독립 optimizer(3개) + agent PPO optimizer 별개 vs 통합 |
| P3-4-3 | maze_state 추출 시점 | rollout 중 매 stride step env.extract_maze_state(rgb) (procgen rgb 필요) vs ground-truth state API |
| P3-4-4 | 학습 스케줄/총 step | RL 25M (Phase 1 동일) + 매 update마다 3 빌드 각 K=32. seed 수 |
| P3-4-5 | 로깅/체크포인트 | 빌드별 loss·gate(tanh)·recon JSONL + 빌드별 checkpoint |

### Phase 3.4 sub-단계
- **3.4.1 ✅ WSL PASS (2026-05-21)**: `src/split_maze/train_phase3.py` (~290줄) — `Phase3Config` + `collect_rollout_with_pairs` (augmented: h_agent + maze_state per step, 정렬=에이전트가 본 상태) + `train_phase3` co-train 루프 (PPO + paired_collect + 빌드별 K=32 update, warmup, 빌드별 독립 optimizer). `state_extractor` 주입 가능. `tests/test_train_phase3.py` 8 tests. 전체 회귀 312 PASS (회귀 0). (test 1개 fail→정정: extract_maze_state가 노이즈에 false-positive — §8 buglog.)
- **3.4.2 ✅ WSL PASS (2026-05-21)**: `scripts/train_phase3.py` CLI (--mock / procgen, --lm_checkpoint 로드→B4/V2 각 사본, --builds 선택, 빌드별 JSONL+ckpt). CLIHelpers 2 tests + CPU smoke 실작동 (B3 5.28→3.89, 3빌드 동시 학습 확인). 전체 회귀 **314 PASS** (회귀 0). → **Phase 3.4 코드(라이브러리+CLI) 전부 완성·검증**. git tag 권장 `v1.3-phase3.4`.
- **3.4.3 ✅ 완료 (2026-05-21) — Phase 3 PASS**: full 25M 공동학습 (1525 upd, ~4.9h). **G1 PASS** (ret +10 안정), **G3 PASS** (B3 3.67→0.18 / B4 1.33→0.79 / V2 3.86→0.000), **G2 0.792 → POST-HOC-5로 노이즈 인정 PASS** (CI[0.756,0.828]∋0.80, ≈Phase1 0.806, C-thin 구조적 미오염). 산출물 `checkpoints/phase3/*.pt` + `logs/phase3.jsonl` + `results/phase3_in_dist.json`. git tag `v1.3-phase3`. ★ V2 recon=0 → Phase 4 in-dist 충실도 trivial 위험 (OOD가 진짜 신호).

### ⚠ 주의 (다음 세션)
- maze_state 추출은 procgen rgb 의존 → train_phase3.py의 paired_collect 통합은
  **MockMazeEnv smoke로 먼저 검증** 후 WSL procgen. (paired_collect 자체는
  oracle/tokenizer만 — env 무관. train_phase3가 rgb→extract_maze_state→MazeState
  → PairedCollector.extract_into 연결.)
- bash mount 이 세션(2026-05-21) 내내 stale. 다음 세션도 Read/Grep(Windows)이
  진실, WSL이 최종 판정 전제.

---

## 9.15 Phase 3.4.3 — WSL 정식 공동학습 + 게이트 (사용자 실행)

### Phase 3 완료 게이트 — 사전 등록 ✔ (2026-05-21)
| # | 기준 | 임계치 | 측정 |
|---|---|---|---|
| G1 | 수렴/발산 없음 | NaN 0 + entropy 안 붕괴 AND agent ret_rolling ≥ +8 | logs/phase3.jsonl |
| G2 | (C-thin) sanity = B1 | agent in-dist 성공률 ≥ 0.80 | evaluate.py |
| G3 | 인터프리터 학습 | B3/B4/V2 각 loss 최종25%평균 < 첫25%평균 | logs/phase3.jsonl |
*결과 본 뒤 변경 금지. goal-misgen(OOD)은 Phase 4로 분리.*

### 학습 사다리 (Phase 1.4 양식)
```bash
conda activate splitmaze
cd /mnt/d/brain/split_maze
mkdir -p checkpoints/phase3 logs results

# 1) GPU smoke (~1~5분) — 발산 없는지, 3빌드 loss 찍히는지
PYTHONPATH=src python scripts/train_phase3.py \
    --env_name maze_aisc --num_envs 16 --num_steps 64 \
    --total_env_steps 50_000 --device cuda --seed 0 \
    --lm_checkpoint checkpoints/lm.pt --builds B3,B4,V2 \
    --log_path logs/phase3_smoke.jsonl
# 기대: pol/val healthy, pairs>0 (실제 sprite 검출), B3/B4/V2 loss 출력, NaN 없음.
# ★ 실측 smoke (2026-05-21, 50k, 48 upd, 77s, ~638 sps) — PASS:
#   - pairs=256/update 일정 (실 maze 프레임 sprite 검출 정상).
#   - 인터프리터 loss 전부 하강: B3 3.50→0.24, V2 3.89→0.09, B4 1.33→0.85.
#   - val 스파이크 점점 축소 (72→36→19→7→2.7…), NaN 없음 — 발산 아님.
#   - ret_rolling +10→+0.7 하강은 *divergence 아님*: 초기 완료 에피소드 1~2개
#     (운 좋은 +10)가 window 채워지며 랜덤정책 평균(~+0.7)으로 회귀한 small-sample
#     artifact. 50k는 미로풀이 학습엔 너무 짧음 (Phase 1도 50k smoke 낮음, 1M부터 상승).
#   - ★ 다음 확인: (C-thin)이라 에이전트=순수 PPO → mid(1M)에서 ret이 Phase 1처럼
#     +3 부근까지 올라가야 정상. 안 오르면 조사 (co-train 간섭 의심).

# 2) mid (~30분~1h) — co-train 추세
PYTHONPATH=src python scripts/train_phase3.py \
    --env_name maze_aisc --num_envs 64 --num_steps 256 \
    --total_env_steps 1_000_000 --device cuda --seed 0 \
    --lm_checkpoint checkpoints/lm.pt --builds B3,B4,V2 \
    --save_dir checkpoints/phase3_mid --log_path logs/phase3_mid.jsonl
# 기대: ret_rolling 상승 추세, recon/probe/next-token loss 하강.
# ★ 실측 mid (2026-05-21, 1M, 61 upd, 708s, ~1410 sps) — PASS, on track:
#   - ret_rolling: upd1~21 ~+2-3 → upd22~49 저점 ~+0.2 (U자) → upd50~60 상승
#     +1.2→+4.4, 끝 +4.0 상승 중. Phase 1 mid(1M)=+3.4 동급+ → (C-thin) 검증
#     (에이전트=순수 PPO, Phase 1처럼 학습). G1(ret≥+8) 진로 양호.
#   - U자 저점은 정상 PPO exploration→exploitation (value head 보정기 정책 흔들림
#     → val loss 83→0.3 수렴 후 정책 개선 폭발). 발산 아님.
#   - 인터프리터: V2 3.86→0.11(강), B3 3.67→0.53, B4 1.33→0.92(plateau, frozen
#     LM+어댑터 제한 용량 — 설계상 약한 해석자). G3 셋 다 하강.
#   - 체크포인트 checkpoints/phase3_mid/{agent,B3,B4,V2}.pt 저장 확인.

# 3) full 25M (~수시간) — Phase 3 게이트 목표
PYTHONPATH=src python scripts/train_phase3.py \
    --env_name maze_aisc --num_envs 64 --num_steps 256 \
    --total_env_steps 25_000_000 --device cuda --seed 0 \
    --lm_checkpoint checkpoints/lm.pt --builds B3,B4,V2 \
    --save_dir checkpoints/phase3 --log_path logs/phase3.jsonl
# 기대: ret_rolling ≥ +8 (G1).

# 4) G2 판정 — 공동학습 agent in-dist (Phase 1 evaluate 재사용)
PYTHONPATH=src python scripts/evaluate.py \
    --checkpoint checkpoints/phase3/agent.pt \
    --mode in-dist --num_episodes 500 --num_envs 32 \
    --device cuda --seed 0 --output_path results/phase3_in_dist.json
# 기대: success_rate ≥ 0.80 (G2).
# ✔ 포맷 호환 확인됨: train_phase3는 {"agent": state_dict}로 저장,
#   evaluate.py는 ckpt["agent"] 로드 — 그대로 작동 (train_agent.py와 동일 포맷).
```

### G3 판정 (logs/phase3.jsonl 분석)
- 각 빌드 `<name>/loss` 시계열에서 첫 25% 구간 평균 vs 마지막 25% 구간 평균 비교. 셋 다 감소면 G3 PASS. (간단 분석 스크립트는 필요 시 다음 세션에.)

### 보고 양식
- 깨지면: 어느 단계(smoke/mid/full/eval) + 마지막 출력 ~10줄 + traceback.
- 통과면: ret_rolling 최종값, in-dist success_rate, 3빌드 loss 첫/끝 평균.

### evaluate.py 체크포인트 포맷 — ✔ 호환 확인
evaluate.py(scripts/evaluate.py:91-95)는 `ckpt["agent"]` 로드. train_phase3
CLI는 `{"agent": agent.state_dict()}` 저장 — 동일 포맷, 보강 불필요.

---

## 9.6 사용자 WSL 1차 검증 명령 (2026-05-18 신규)

> 이 세션 끝나면 사용자는 WSL에서 아래를 돌리면 됨.

### A. 전체 단위 테스트 (~196 tests 기대 — Phase 2의 ~158 + acc ~38)
```bash
conda activate splitmaze
cd /mnt/d/brain/split_maze
PYTHONPATH=src python -m pytest tests/ -q
```
기대 출력 마지막 줄: `~196 passed in X.Xs` (실제 acc 테스트 수는 collection 결과 확인).

### A3. ★ 신규 acc 테스트만 단독 (Phase 3.1 1차 검증)
```bash
PYTHONPATH=src python -m pytest tests/test_acc.py -v
```
기대: 모든 테스트 pass. 핵심 검증 (특히 (C-thin) detach 정책):
- `TestCThinGradBoundary::test_h_agent_receives_no_grad` — **boundary 1** 핵심: backward 후 h_agent.grad == None.
- `TestCThinGradBoundary::test_h_lm_receives_grad` — boundary 2: h_lm.grad 0 아님.
- `TestCThinGradBoundary::test_h_lm_grad_includes_both_directions` — A2L 타깃 + L2A 예측 양쪽에서 h_lm 받는지.
- `TestCThinGradBoundary::test_h_agent_grad_zero_even_when_l2a_only` — L2A loss만 backward해도 h_agent 미오염.
- `TestWInit::test_orthogonal_init_yields_orthonormal_rows_or_cols` — W @ W.T ≈ I (P3-3-A 박제).
- `TestIntegrationSmoke::test_one_optim_step_decreases_loss` — AdamW로 50 step 회귀 sanity.

### A3-주의. PLAN §4.2 수식 vs 코드 해석 점검 ★

PLAN §4.2 박제식: `ñ_agent = LayerNorm(h_agent); ... ñ_agent.detach()`.
PyTorch 의미 그대로 따르면 `ñ_agent.detach()`가 **ln_agent.weight/bias까지 차단**해서 ACC LayerNorm이 학습 안 됨 (의도와 모순).
acc.py는 *(C-thin) 의도*를 더 정확히 반영하려고 `n_agent = ln_agent(h_agent.detach())`로 구현 — h_agent 그래디언트는 차단되지만 ln_agent 파라미터는 ACC-side로 학습된다.
WSL 검증(42/42 PASS, 2026-05-20)에서 해당 해석 받아들여짐. PLAN §4.2 본문 각주는 다음 세션에 추가 (선택).

### A4. ★ 신규 paired_collect 테스트만 단독 (Phase 3.2 1차 검증)
```bash
PYTHONPATH=src python -m pytest tests/test_paired_collect.py -v
```
기대: 모든 테스트 pass. 핵심:
- `TestPairBufferAdd::test_add_fifo_wraparound_data` — FIFO ring 동작 (P3-2-2 박제).
- `TestPairBufferAdd::test_add_huge_batch_keeps_last_capacity` — B≥capacity edge.
- `TestPairBufferAdd::test_add_detaches_h_agent` — (C-thin) boundary 1 정합 (RL grad 누수 방지).
- `TestPairBufferSample::test_sample_returns_clone` — buffer overwrite 후 snapshot 안정성.
- `TestPairBufferSample::test_sample_is_uniformish` — uniform random sampling 분포 확인.
- `TestPairedCollectorMakePair::test_oracle_none_returns_none` — agent-on-cheese 처리.
- `TestPairedCollectorMakePair::test_length_overflow_returns_none` — token 초과 skip.
- `TestPairedCollectorExtract::test_stride_count_all_valid` — stride=4, T=8, N=2 → 4 페어.
- `TestIntegrationSmoke::test_extract_then_sample_round_trip` — full pipeline.
- `TestIntegrationSmoke::test_default_p32_hyperparams_match_session_handoff` — 박제값 drift 방지.

### A5. ★ 전체 회귀 (acc 42 + paired_collect 43 추가, 2026-05-20 시점 251 PASS)
```bash
PYTHONPATH=src python -m pytest tests/ -q
```
Phase 1·2 기존 회귀 없는지 확인.

### A6. ★ 신규 adapter + builds 테스트 (Phase 3.3.0/3.3.1 1차 검증)
```bash
PYTHONPATH=src python -m pytest tests/test_adapter.py tests/test_builds.py -v
```
기대: 모든 테스트 pass. 핵심:
- `TestGatedCrossAttentionBlock::test_gate_init_is_identity` — Flamingo gate=0 → output==hidden (LM Phase 2 동작 보존).
- `TestAgentResampler::test_n_kv_too_small_raises` — n_kv≥2 강제 (latent 차별화).
- `TestAgentResampler::test_h_agent_flows_through` — 다른 h_agent → 다른 adapter token.
- `TestB3ProbeTargets::test_round_trip_known_slots` — Slots→render→encode→ids→복원 인덱스 일치.
- `TestB3ProbeTargets::test_round_trip_various_surface_forms` — 표면 변형 무관 슬롯 복원.
- `TestB3ProbeUpdate::test_h_agent_receives_no_grad` — **(C-thin) boundary 1** (probe가 agent 미오염).
- `TestB3ProbeSmoke::test_optim_reduces_loss` — probe CE 50 step 감소.
- `TestCESafety::test_all_ignore_returns_zero_with_graph` — 전체 ignore CE NaN 가드.

### A7. ★ 전체 회귀 (3.3.0/3.3.1 추가 후) — 2026-05-21 281 PASS 확인
```bash
PYTHONPATH=src python -m pytest tests/ -q
```

### A8. ★ B4Adapter 테스트 (Phase 3.3.2 1차 검증)
```bash
PYTHONPATH=src python -m pytest tests/test_builds.py -v -k "B4"
PYTHONPATH=src python -m pytest tests/ -q   # 전체 회귀 (281 + B4 13 ≈ 294)
```
기대: 모든 테스트 pass. 핵심:
- `TestB4AdapterConstruction::test_lm_params_frozen` — LM core requires_grad=False.
- `TestB4AdapterConstruction::test_train_keeps_lm_in_eval` — train() 후에도 LM eval (Flamingo).
- `TestB4AdapterForward::test_gate_zero_makes_adapter_inert` — gate=0 → h_agent 무관 동일 logits (식별성).
- `TestB4AdapterForward::test_open_gates_make_h_agent_matter` — gate 열면 h_agent 반영.
- `TestB4AdapterUpdate::test_lm_core_receives_no_grad` — **(C-thin) boundary 2** (LM frozen).
- `TestB4AdapterUpdate::test_h_agent_receives_no_grad` — boundary 1 (gate 열고 검증).
- `TestB4AdapterUpdate::test_optim_reduces_loss` — next-token CE 30 step 감소.

> ★ 2026-05-21 mount stale 재발: 샌드박스 bash AST가 B4Adapter·V2ACC 추가 후에도 *stale snapshot* (B3Probe만 보임, sleep 3s 후에도 동일, 세션 내내 동결). Read/Grep tool(Windows view)은 B4Adapter line 219 + V2ACC line 345 + __all__ 5빌드 정상 확인. py_compile은 통과. → §8 박제 패턴 그대로, WSL이 최종 판정. 이 세션 mount sync 유난히 느림.

### A9. ★ V2ACC 테스트 (Phase 3.3.3 1차 검증) — builds.py 3 빌드 완성
```bash
PYTHONPATH=src python -m pytest tests/test_builds.py -v -k "V2ACC"
PYTHONPATH=src python -m pytest tests/ -q   # 전체 회귀 (295 + V2 9 = 304)
```

### A10. ★ train_phase3 골격 테스트 (Phase 3.4.1 1차 검증)
```bash
PYTHONPATH=src python -m pytest tests/test_train_phase3.py -v
PYTHONPATH=src python -m pytest tests/ -q   # 전체 회귀 (304 + train_phase3 ~7)
```
기대: 모든 테스트 pass. 핵심:
- `TestCollectRolloutWithPairs::test_shapes` — h_agent (T,N,d_a) + maze_states T×N 정렬.
- `TestTrainPhase3::test_runs_and_returns_logs` — PPO + 페어수집 + 로그.
- `TestTrainPhase3::test_interpreter_updates_fire` — buffer ready 후 B3/B4/V2 각 K=32 update 로그.
- `TestTrainPhase3::test_default_extractor_no_sprites_no_crash` — MockEnv 랜덤 rgb → sprite 없음 → 페어 0 → 루프 무사.

> 참고: train_phase3 smoke는 `state_extractor` 주입(stub)으로 페어 흐름 검증.
> 실제 procgen에선 default_state_extractor(=env.extract_maze_state)가 sprite 검출.
> 이 MockEnv smoke는 procgen 무관 — co-train *루프 배선*만 검증.

### A11. ★ train_phase3 CLI 테스트 (Phase 3.4.2 1차 검증)
```bash
PYTHONPATH=src python -m pytest tests/test_train_phase3.py -v -k "CLIHelpers"
# CPU smoke — 실제 CLI 전체 배선 (procgen 무관, fresh LM):
PYTHONPATH=src python scripts/train_phase3.py --mock \
    --num_envs 4 --num_steps 8 --total_env_steps 64 \
    --builds B3,B4,V2 --interp_batch 8 --stride 2 \
    --acc_updates_per_rl 2 --device cpu --seed 0
PYTHONPATH=src python -m pytest tests/ -q   # 전체 회귀
```
기대: CLIHelpers 2 tests pass + smoke가 update 로그 몇 줄 출력 후 정상 종료
("[train_phase3] N updates in ...s"). lm_checkpoint 없으면 fresh LM 경고는 정상.

> ★ 2026-05-21 mount stale 또 재발: bash py_compile이 test_train_phase3.py를
> line 200 `'(' was never closed`로 봤으나 Read tool(Windows)은 line 204 정상
> 종료 확인 — §8 buglog 그대로. 이 세션 mount 완전 동결. WSL이 최종 판정.
기대: 모든 테스트 pass. 핵심:
- `TestV2ACCConstruction::test_only_interface_proj_trainable_in_lm` — LM core frozen, interface_proj만 trainable (P3-2-A).
- `TestV2ACCConstruction::test_interpreter_params_are_acc_plus_interface` — interpreter = ACC + interface_proj (core 제외).
- `TestV2ACCUpdate::test_h_agent_receives_no_grad` — (C-thin) boundary 1.
- `TestV2ACCUpdate::test_lm_core_receives_no_grad_but_interface_does` — **(C-thin) boundary 2** — interface_proj만 학습, core grad None.
- `TestV2ACCUpdate::test_acc_W_receives_grad` — ACC W 학습.
- `TestV2ACCUpdate::test_optim_reduces_loss` — recon loss 30 step 감소.

### A2. ★ 신규 lm_train 테스트만 단독 (Phase 2.2 1차 검증)
```bash
PYTHONPATH=src python -m pytest tests/test_lm_train.py -v
```
기대: 24 tests pass. 핵심:
- `test_train_lm_loss_decreases_across_epochs` — 학습 루프가 실제로 손실 감소시키나.
- `test_evaluate_roundtrip_exact_match_with_mocked_echo` — 평가 산수 올바른가.
- `test_evaluate_72_combinations_returns_72_total` — 72조합 게이트 카운팅.
- `test_gate_pass_*` — 임계치 동작.
- `test_train_lm_saves_checkpoint` — 체크포인트 저장 + 로드 가능.

### C. ★ Phase 2.2 짧은 CPU smoke (~10초)
```bash
PYTHONPATH=src python scripts/train_lm.py \
    --corpus_size 200 --epochs 2 --batch 16 --device cpu --seed 0
```
기대: 2 epoch 학습 로그 + Phase 2 gate 출력 (FAIL 정상 — 데이터 작아).

### D. ★ Phase 2.2 정식 학습 + 게이트 판정
```bash
mkdir -p checkpoints logs results
PYTHONPATH=src python scripts/train_lm.py \
    --corpus_size 50000 --epochs 10 --batch 64 \
    --device cuda --seed 0 \
    --save_path checkpoints/lm.pt \
    --log_path  logs/lm.jsonl \
    --gate_path results/lm_gate.json
```
기대: 10 epoch (수분~10분, CUDA) → 마지막에 `Phase 2 verdict: PASS` 출력.
PASS 조건 (사전 등록): `roundtrip_exact ≥ 0.95 AND combo_72_pass = 1.0`.

깨지면 보고: 어느 step(A/A2/C/D)에서, 마지막 출력 ~10줄 + traceback,
`torch.__version__`.

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
- **2026-05-18 ★ Phase 1 완료 — 권장 tag**:
  ```bash
  git add -A && git commit -m "Phase 1 — agent + PPO + evaluate, 105 tests, gates PASS (in-dist=0.806, OOD misgen=0.5217)"
  git tag v1.1-phase1
  ```
- **2026-05-20 ★ Phase 2 완료 — 권장 tag**:
  ```bash
  git add -A && git commit -m "Phase 2 — LM + ACC handle B + POST-HOC-4 warmup, ~158 tests, slot_match=0.994 combo_72=1.0"
  git tag v1.2-phase2
  ```

---

## 11. 프로젝트 작업 스타일 (러닝/문화)

- **차분히 차근차근**. 큰 결정엔 `AskUserQuestion`.
- **사람말로**. 필요하면 비유(미로/통역, 영수/민수 등).
- **박제 문화**. Deferred 항목·러닝 모두 명시적으로 문서화. PLAN.md에 정밀화 로그 유지.
- **결정 시 옵션 명시 + 추천 + 이유**. 사용자가 "추천대로" 라고 답할 수 있게.
- **사전 등록 (pre-registered)**: 임계치·시나리오·완료 기준을 학습 *전*에 박제. 결과 본 후 임계치 변경 금지 (잘못된 임계치 발견 시 "Post-hoc adjustments"로 *이유 명시*).
- **★ 작은 단위로, 검증부터 (2026-05-21 사용자 강조)**: 큰 산출물(builds.py)은 sub-단계(3.3.0→3.3.1→3.3.2→3.3.3)로 쪼개 *각 단위마다 WSL 검증 PASS 후 다음으로*. 한 번에 몰아 짜고 한꺼번에 검증 X — 버그 추적 어렵고 검토 호흡 짧아짐. 검증된 체크포인트마다 git commit 권장.
