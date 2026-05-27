# SPLIT-MAZE

🇰🇷 한국어 · [🇬🇧 English](README.en.md) · [🇨🇳 中文](README.zh.md) · [🇯🇵 日本語](README.ja.md)

*procgen 미로 RL 에이전트와 from-scratch 미로언어 LM을 인공 뇌량으로 잇고, 목표 오일반화한 에이전트의 실제 내부 목표를 LM이 합리화가 아니라 충실하게 말하는지 — 둘 다 from scratch로 — 검증한 프로젝트.*

라인업의 세 번째다: SPLIT-9(사후 어댑터, 음성) → SPLIT-MNIST(공동학습 분리-재구성 V2, 동질 toy 양성) → 본 프로젝트. V2 패턴을 *이질적인 진짜 환경*(미로 RL 에이전트 + 미로언어 LM, 양쪽 from scratch)에서, *목표 오일반화*(치즈가 늘 우상단이던 미로에서 자란 에이전트가 OOD에선 치즈를 두고도 우상단으로 감)를 충실 vs 합리화 판별기로 재검증했다.
세 단계로 갔다. **Phase 4** — 핵심 가설(분리-재구성 V2 > next-token 어댑터 B4 충실도)은 기각(Scenario C). 단 통제 2×2로 confound를 해소하고 풍부한 음성 3개를 얻었다(목표 오일반화가 표상 수준에서 읽힘 · "충실 ≠ 합리화" 재프레임 · ACC 단일벡터 병목). **Phase 5** — 기록 기반 뇌량(CCM)이 두 뇌가 마중하면 학습 번역기 천장까지 자란다(공동적응 +0.064±0.020, 5-seed, 결정자 뇌 3개로 일반화). **Phase 6** — 라이브 양방향 되먹임으로 해석자를 더 grounded하게 만들려 했으나 음성(Δ +0.037 < 문턱 +0.05 + 앵무새 가드 실패; 되먹임 = 목표편향/오프로딩). 사전등록 가드가 spurious 효과를 잡아낸 깨끗한 음성이다.

**헤드라인:** Phase 4 V2 기각(per-slot 0.375 / 스왑 0.42 vs B4 0.66 / 0.83) · Phase 5 다리 성장 +0.064±0.020 (5-seed · 3-뇌 일반화) · Phase 6 R2 음성 (Δ +0.037 < +0.05, in-dist 가드 실패).

**[상세 보기 → 3개 Phase 개요 + 전체 결과·그래프 (한/영)](https://halmoneysonmat.github.io/split-maze/)**

---

### 연구 라인업

[SPLIT-9](https://github.com/HalmoneySonmat/split-9) → [SPLIT-MNIST](https://github.com/HalmoneySonmat/split-mnist) → **SPLIT-MAZE**

재현·스택은 상세 페이지에. 테스트: `pytest tests/ -q`.

---

이 프로젝트를 만드는 데 Claude를 사용했다.
