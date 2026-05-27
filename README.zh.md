# SPLIT-MAZE

[🇰🇷 한국어](README.md) · [🇬🇧 English](README.en.md) · 🇨🇳 中文 · [🇯🇵 日本語](README.ja.md)

*把 procgen 迷宫的 RL 智能体与 from-scratch 的迷宫语言 LM 用人工胼胝体相连,检验目标过泛化的智能体的真实内部目标,LM 能否「忠实地、而非编造地」说出 —— 两侧都 from scratch。*

脉络中的第三个: SPLIT-9(事后适配器,阴性)→ SPLIT-MNIST(共同训练的分离重建 V2,同质 toy 阳性)→ 本项目。在*异质的真实环境*(迷宫 RL 智能体 + 迷宫语言 LM,两侧 from scratch)里复检该 V2 模式,并以*目标过泛化*(在「奶酪总在右上」的迷宫里长大的智能体,在 OOD 下无视真实奶酪仍走向右上)作为忠实 vs 编造的判别器。
分三个阶段。**Phase 4** —— 核心假设(分离重建 V2 > next-token 适配器 B4 的忠实性)被否(Scenario C);但用受控 2×2 解决了 confound,得到三个有料的阴性(目标过泛化在表征层可读 · 「忠实 ≠ 合理化」再框定 · ACC 的单向量瓶颈)。**Phase 5** —— 基于记录的胼胝体(CCM)在两脑相向而行时会长到训练翻译器的天花板(共适应 +0.064±0.020,5 seed,泛化到 3 个决策者大脑)。**Phase 6** —— 想用实时双向反馈让解释器更 grounded,结果阴性(Δ +0.037 < 阈值 +0.05 + 未过鹦鹉守门;反馈只是把智能体偏置/卸载向目标先验)。预注册守门捕获 spurious 效应的干净阴性。

**头条:** Phase 4 V2 被否(per-slot 0.375 / swap 0.42 vs B4 0.66 / 0.83)· Phase 5 桥生长 +0.064±0.020(5 seed · 3 大脑)· Phase 6 R2 阴性(Δ +0.037 < +0.05,in-dist 守门失败)。

**[查看详情 → 三阶段概览 + 全部结果·图表(韩/英)](https://halmoneysonmat.github.io/split-maze/)**

---

### 研究脉络

[SPLIT-9](https://github.com/HalmoneySonmat/split-9) → [SPLIT-MNIST](https://github.com/HalmoneySonmat/split-mnist) → **SPLIT-MAZE**

复现与技术栈见详情页。测试: `pytest tests/ -q`。

---

本项目的搭建使用了 Claude。
