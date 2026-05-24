# SPLIT-MAZE

[🇰🇷 한국어](README.md) · [🇬🇧 English](README.en.md) · 🇨🇳 中文 · [🇯🇵 日本語](README.ja.md)

*把解 procgen 迷宫的 IMPALA-CNN RL 智能体,与从零训练的合成迷宫语言 LM,用人工胼胝体(ACC)连起来,验证当智能体目标误泛化时,LM 能否*忠实地、而非合理化地*说出其真实的内部目标 —— 两侧都从零训练。*

> **TL;DR.** 邻接项目 [SPLIT-9](../split_brain_go) 以 negative result 收尾:*冻结 LLM +
> post-hoc 适配器*在忠实性上撞到结构性天花板;[SPLIT-MNIST](../split_mnist) 则在同质 toy
> 上验证了*共同训练 + 解耦重建 ACC(V2)*能打破该天花板(Scenario A)。本项目把 V2 模式
> 拿到**异质的真实环境、两侧都从零训练**下重新检验 —— procgen 迷宫 RL 智能体(决策者)+
> 合成迷宫语言小型解码器 LM(解释者),并以*目标误泛化*(在奶酪总在右上的迷宫里长大的
> 智能体,OOD 时即使奶酪在别处也仍奔向右上)作为忠实 vs 合理化的判别器。**核心假设
> (解耦重建 V2 > next-token 适配器 B4 的忠实度)被否(Scenario C)** —— V2 反而*更不
> 忠实*(per-slot 0.375 / 主动交换 0.42 vs B4 0.66 / 0.83)。但受控 2×2(CTRL-2x2)连
> confound 一并解决,并得到三项丰厚的 negative result: ① 目标误泛化在*表征层面*可读
> (cheese_dir 忠实度 in-dist 0.85 → OOD 0.07 崩塌)、② **「忠实 ≠ 合理化」的重构**
> (主动交换因果地证明:B4 的 OOD「合理化」其实是*忠实读取一个目标误泛化的智能体*)、
> ③ ACC 的*单一摘要向量瓶颈* < 适配器的分布式 cross-attention。**单一 RL seed 的
> descriptive 结果** —— 统计确认是 multi-seed 的后续任务。
>
> **后续(Phase 5 — CCM): 桥会生长。** V2 被否后,我们试了既非*训练出的翻译器*也非薄重建
> 的桥 —— **CCM(共激活胼胝体记忆)**:当两网看到同一场景时,仅*记录(记住)共同激活的节点
> 之间的对应*,并用该记录驱动 agent→LM。① *仅凭记录*(零 backprop)主动交换 **0.445 = B4
> 的 52%**,关键要素是**「共模(均值)去除」**(中心化*或*白化*二者之一即足够*,生的 Hebbian
> 会崩溃)。② 试图在闭环中*让桥生长*(step2)失败(用力推 → task 崩塌,温和则停滞;
> `loss↓ ≠ 成功`)。③ 但把桥变*可塑*(记忆→种子),并让两脑相向而行(**step3**: agent gentle
> + LM 一个块 + W),薄桥便生长到*训练翻译器的天花板(~0.80)* —— 共适应的纯增益
> **+0.064±0.020(5-seed 确认,5/5 为正)**,task 与语言均保留。这是对用户原始愿景(「两脑
> 同时适应、桥会生长」)的*证据* —— **该效应也泛化到另外 3 个决策者大脑**(GENERALIZES,
> 均值 +0.081,3/3 为正)。但解释器 LM 为共享,故泛化仅限决策者大脑层面;且效应*依赖大脑*
> (在我们破坏了任务的那 1 个大脑上消失 =「摇晃它,一部分并不稳固」)。

---

## 为什么做这个

在 Sperry 与 Gazzaniga 的裂脑实验里,切断连接左右半球的胼胝体会产生一个奇观:把只有
右半球能看到的「走」字卡给患者,患者就站起来走。问「为什么走?」,没看到卡片的、具备
语言能力的左半球便*编造*一个看似合理的理由。不是撒谎 —— 患者是真的*相信*。Gazzaniga
称这个模块为*左半球解释器*:流畅、自洽、却常常出错。

现代 AI 刻意复刻这个结构。视觉编码器、机器人策略、决策模型旁边放一个 LLM,用自然语言
「解释」此刻发生了什么。但 LLM 究竟是*真的在翻译*上游信号,还是只吐出*统计上看似合理的
文本*,通常未经验证。

[SPLIT-9](../split_brain_go) 用 9×9 围棋 + 冻结 LLM 试图直接研究,撞上了*post-hoc 适配器
的结构性天花板*:loss 下降的 95% 是领域 prior,只有 5% 是逐盘信号,定性 10/10 答错。
[SPLIT-MNIST](../split_mnist) 用*共同训练 + 解耦重建 loss(V2)*打破了该天花板 —— 但那是个
同质(两侧都是小 CNN、都是图像的一半)、对称、且缺少区分忠实与合理化所需的*目标误泛化*
的 toy。

SPLIT-MAZE 把 V2 模式拿去*动真格*检验:

> 当把已验证的 V2 模式(解耦重建 ACC)放到**异质的真实环境、两侧都从零训练**下学习时,
> 解释者会*忠实地、而非合理化地*说出决策者的真实内部目标(含目标误泛化)吗?

目标误泛化(Langosco et al., 2022)是**忠实解释 vs 合理化的完美判别器**。合理化的解释者
说「在找奶酪」(学得的 prior),忠实的解释者应说「奔向右上角」(真实内部目标)—— 在*我们
知道正确答案的环境里*测量 SPLIT-9 的失败样式。

---

## 我们做了什么

```
SPLIT-MAZE
  procgen 迷宫观测 (B,3,64,64)            合成迷宫语言句子
        │                              "agent top-right heading up-right cheese down-left"
   ┌────┴─────┐                                    │
   │ IMPALA   │ ← 决策者                      ┌─────┴──────┐
   │ CNN (RL) │   25M PPO, from scratch       │ 小型 LM    │ ← 解释者
   └────┬─────┘                              │ (3层解码器) │   中立语料
   h_agent (B,256) ─┐                         └─────┬──────┘   from scratch
        │           │   ┌──── ACC ────┐         h_lm (B,256)
        │           └──→│  W (256×256) │←────────────┘
   policy/value        │  ĥ_lm=W·ñ_a   │   ← (C-thin) 解耦重建 loss:
   (仅 RL 奖励)         │  ĥ_a =Wᵀ·ñ_lm│     智能体 detach + LM core stop-grad
                       └───────────────┘
   describer oracle: 迷宫状态 → 正确句子(语料·配对·评测标签)
```

全部*从零共同训练*。训练的*时间/数据*共享,只把训练的*梯度路径*分开。

**3 槽合成迷宫语言** —— AGENT_REGION(3×3=9)/ HEADING(8 方位+still=9)/ CHEESE_DIR(8)。
HEADING 与 CHEESE_DIR 为*独立槽*是关键(二者相悖的句子 = 目标误泛化的签名)。LM 用从
文法均匀采样的中立语料训练 —— *没有 prior*。偏置只住在智能体内部(从源头封死 SPLIT-9 式
的 LLM 污染)。

**构建**(共享*同一个* RL 智能体 → V2 vs B4 *字面上*受控):
- **B1** 智能体单独(任务上限 reference)
- **B3** 直接 probe MLP —— 衡量「信息是否在 h_agent 里」的尺子
- **B4** ★ Flamingo 适配器(Resampler + 分布式 cross-attn,仅 next-token)—— 忠实再现 SPLIT-9 模式
- **V2** ★ ACC(tied→untied W,仅重建)—— 本假设的本形
- **B4Thin / V2Rich** —— 受控 2×2(CTRL-2x2)缺的两格

**(C-thin) 双重梯度边界** —— 智能体始终 detach(无污染),LM 语言核心 stop-grad(保护中立性),
只有 ACC W 与接口适配。把决策者保持无污染,才让「忠实读取」*明白无误地*忠实。

**训练.** 智能体 `maze_aisc` 25M PPO。LM 中立语料 50k。Phase 3 共同训练 25M(智能体 +
B3/B4/V2 同时挂载,~5h)。单块 8 GB RTX 3070 Ti / WSL2。

---

## 我们测了什么

### 1. 任务性能(sanity)
- **智能体**: in-dist 成功率 **0.806**,OOD 目标误泛化率 **0.522**(eligible 276)。
- **LM**: 槽一致 **0.994**,72 组合全生成 1.0,自编码 0.987。
- 共同训练后智能体性能 = B1(确认无污染,(C-thin) 正常工作)。

### 2. per-slot 忠实度 ★ —— 目标误泛化在表征层面可读
生成句子的槽与 describer oracle 正解一致的比例(每条件 n=327,680):

| | region | heading | cheese_dir | OOD cheese_dir |
|---|---:|---:|---:|---:|
| B3 | 0.80 | 0.35 | 0.86 | **0.07** |
| B4 | 0.79 | 0.35 | 0.85 | **0.07** |
| V2 | 0.58 | 0.23 | 0.32 | **0.05** |

cheese_dir 从 in-dist 0.85 → OOD 0.07 **在所有构建上崩塌**。智能体在 OOD 下*不表征真实
奶酪方向*,而表征「右上」prior —— 与架构无关的硬结果。

### 3. 主动交换 ★ —— 忠实 ≠ 合理化(核心贡献)
把 h_agent(A)→(B) 做 α 插值时,生成 cheese_dir 跟随 B 的比例(swap-following):

| | swap-following | OOD 合理化率 |
|---|---:|---:|
| B4 | **0.830** | 0.50 |
| B3 | 0.828 | 0.40 |
| V2 | 0.419 | 0.09 |

B4/B3 *强烈因果追踪* h_agent(swap 0.83)。因此 B4 的 OOD「合理化」(0.50)不是搪塞,而是
**忠实读取一个目标误泛化智能体的结果**。→「忠实 vs 合理化」的区分比 SPLIT-9 的假设更微妙
(不是解释者的错)。

### 4. 受控 2×2(CTRL-2x2)—— 败因是 loss 还是 interface
V2 vs B4 在*学习信号*(重建/next-token)与*接口*(单向量/分布式)上同时不同。把冻结的
智能体·LM 上仅 4 格解释者做 post-hoc 拟合以分离:

| cell | interface × loss | in-dist 忠实度 | swap |
|---|---|---:|---:|
| V2 | thin × 重建 | 0.367 | 0.379 |
| B4Thin | thin × next-token | 0.672 | 0.778 |
| B4 | rich × next-token | 0.865 | 0.991 |
| V2Rich | rich × 重建 | **0.001**(崩溃) | **0.000** |

**薄的一对**(相同的单向量接口): next-token(B4Thin)以忠实度 **+0.31** / swap **+0.40**
压倒重建(V2)→ V2 的失败不只因接口薄,而是*重建这一学习信号本身就弱*。接口也有效
(B4 ≫ B4Thin,+0.19/+0.21)。V2Rich(rich × 重建)*degenerate*(MSE 从 1.70→0.73 下降但
生成崩溃 —— full-hidden 目标 ill-posed;`loss↓ ≠ 成功`重演)。

---

## 判定 —— Scenario C(核心假设被否)

预注册阈值(PLAN §5.6)vs 结果:

| 测量 | 阈值 | 结果 | 判定 |
|---|---|---|---|
| #2 in-dist 槽一致 V2 | ≥ 0.80 | 0.375 | ✗ |
| #2 OOD V2−B4 忠实度 Δ | ≥ +0.15 | −0.07 | ✗(相反) |
| #3 swap-following V2−B4 | ≥ +0.15 | −0.41 | ✗(相反) |
| 合理化率 B4−V2(OOD) | ≥ 0.20 | +0.41 | ✓* |
| CTRL-3 重建复活(swap≥+0.10 ∧ slot≥+0.05) | 两者 | 两者皆负 | ✗ |

\* 合理化率在方向上满足,但主动交换揭示:V2 的低合理化是*弱*(因果追踪失败)而非*原则*。
综合判定 = **Scenario C**。

---

## 这意味着什么

**核心假设被否 —— 但只是其狭义形态。**「解耦重建(V2)是忠实度的核心要素」是错的。受控
2×2 连 confound 也解决了:在接口固定下,重建信号仍弱于 next-token。V2 的劣势是*学习信号
本身的性质,而非接口的伪影*。

**但机制本身活着。** B4(next-token + 分布式 cross-attn)以主动交换 0.83 *因果且忠实地*
追踪了智能体 —— 共同训练确实产生了*忠实的解释者*。最大的概念收获是**「忠实 ≠ 合理化」的
重构**:忠实读取一个目标误泛化的智能体,就会报告那个误泛化的目标。看似合理化的输出可能
正是忠实的读取;要分开二者,需要(像主动交换那样的)*因果*测量。

而且 —— V2 在 SPLIT-MNIST(同质·对称·低维)中获胜、却在 SPLIT-MAZE(异质·非对称·高维)
中落败,与其说是解耦重建*原理*的失败,不如说是*单一摘要向量接口的容量极限*。toy 中够用的
一点压缩,不足以把 IMPALA 表征忠实地贴到 LM 句嵌入流形上。(`loss↓ ≠ 成功`在本项目里栽了
两次 —— POST-HOC-6 的 degenerate collapse、CTRL-2x2 的 V2Rich。永远怀疑平凡解。)

---

## Phase 5 — CCM(共激活胼胝体记忆): 桥会生长

V2 被否后,我们试了一座*完全不同*的桥。不*训练*翻译器 —— 当两网接收同一场景时,仅
**记录(记住)共同激活的节点之间的对应**,再用该记录驱动 agent→LM。**CCM(Co-activation
Callosal Memory)。** 接口与生成路径与 B4Thin byte-identical,但桥 `W` 由*闭式统计*(或以其
为种子的可塑学习)填充,而非梯度。「是记忆,不是学习」即其身份。它从生物胼胝体的*活动依赖
可塑性*与*抑制性归一化*文献出发。

**step1 —— 仅凭记录便达一半。** 在冻结的智能体+LM 之上,用 245,760 对以闭式拟合 `W`。零
backprop,**主动交换 0.445 = 训练适配器 B4 的 52%。** 但*生的*共激活(Hebbian 外积)是
degenerate(均值项主导 → 常数崩溃)。生效的关键是**「共模(均值)去除」** —— 干净的 2×2
ablation 表明:中心化*或*白化**任一单独**即可几近完全恢复(二者是冗余的路径)。→ **「桥要
记住的是*什么不同*,而非*什么共同激活*。」**(我们起初认为「白化是决定性的」,ablation 将其
更正为 confound —— 如实记录。)

**step2 —— 闭环(记录 W·仅智能体): 阴性。** 让两脑适应桥以*使桥生长*。用力推 → task 崩塌
(return 10→3.75);温和到不崩 → 桥停滞(swap −0.05)。两者皆不支持 —— 是 `loss↓ ≠ 成功`的
教科书案例(训练 loss 下降,held-out 忠实度不升)。

**step3 —— 让桥*活*起来、两脑相向而行: 圆满结局。** 改了两点 —— (i) 让 `W` *可塑*(可训练
参数)但以记录值 warm-start(「记忆是种子」);(ii) 让 LM 也相向: 仅解冻生成路径上的一个解码
块 `blocks[0]`(小 lr),并用**语言 KL 锚**(对冻结参考)保住语言。分阶段 —— A1(两脑冻结·仅 W
= 训练翻译器天花板 / control)→ A2(共适应)。

| 桥 | 主动交换 | 备注 |
|---|---:|---|
| 记录记忆(step1) | 0.445 | 种子 |
| A1 plastic W(冻结脑) | 0.683 | 记忆→翻译器(B4 的 80%) |
| A1-long(W-budget control) | 0.725 | 再加 W 也 plateau(< A2) |
| **A2 共适应** | **0.784** | 抵达收敛翻译器天花板(~0.78) |

A2 **保留 task(return 10→8.3)与语言(KL≈0)**。**经 5-seed 流程 multi-seed 确认**: 共适应的
纯增益(A2 − W-budget control)= **+0.064 ± 0.020,5/5 seed 为正**(SEM≈0.009 → 非噪声)。决定
性地,连*记录* ridge(零 W 训练)在适应后的系统上也升至 0.445→0.508 —— 这无法由 W 训练解释,
即 **两脑表征确实朝彼此对齐方向生长的证据**。这是对用户原始愿景 ——「两脑同时以不同方式学习
并记住刺激,桥会生长」—— 的*基于证据的支持*。(在*同一个*冻结 RL 智能体上确认 =「在这颗脑上
是真的」;跨脑泛化属下文的探针。)

**而且 —— 它也泛化到别的决策者大脑(GENERALIZES)。** 我们在*另外* 3 个 RL 智能体(各为独立
训练的 25M PPO;解释器 LM 因中立而被共享)上重跑 step3 共适应。每个大脑的效应 = swap(A2) −
swap(A1-long):

| 大脑 | swap(A2) | swap(A1-long) | 效应 | 备注 |
|---|---:|---:|---:|---|
| brain1 | — | — | **+0.137** | control 之上明显残差 |
| brain2 | 0.804 | 0.801 | **+0.004** | 近乎为零 · 任务崩溃(return 10→7.66) |
| brain3 | — | — | **+0.102** | control 之上明显残差 |

均值 = **+0.081 ± 0.069,3/3 为正**。**预注册的冻结判定**(GENERALIZES 当且仅当 ≥ N−1 个大脑
effect>0 AND 均值 ≥ +0.03;N=3 → 需 ≥2/3 为正)两者皆满足 → 判定 = **GENERALIZES**,从「在这颗
脑上是真的」上升为**跨决策者大脑复现**。与 W 无关的佐证亦成立:大脑间适应后的记录 ridge swap 均值
0.500 > 冻结基线 0.445。**但必须诚实**:效应是*异质的* —— 大脑 1·3 在 W-budget control 之上有明显
残差(+0.10~+0.14),但 **大脑 2 基本为零(+0.004),且其任务崩溃**(智能体 return 从 10 降到 7.66,
−23%)。大脑 2 的 raw A2 swap(0.804)≈ A1-long control(0.801),所以它表面的「增益」几乎全部只是
额外的 W 训练 —— **W-budget control 正确地吸收了这个假阳性**(若无该 control 会看似混淆变量)。判定
不依赖大脑 2 勉强为正的符号:即便把大脑 2 当作零,正例仍为 2/3 ≥ N−1,依然通过。→「两个大脑彼此
靠拢时桥会生长」作为一个*方向*在各决策者大脑间复现,但*幅度依赖具体大脑*,且**若破坏决策者的任务则
效应消失**。这是关于「摇晃它是否仍然稳固」的首个实证数据 —— 在 2/3 中稳固,在我们(无意中)破坏了
任务的那个大脑中则不稳固。范围限定:N=3 是探针(5 会更稳健),且只改变了决策者(RL-seed),解释器
LM 被共享 —— 故泛化仅在决策者大脑层面成立;改变解释器大脑(LM-seed)是下一个排队项(队列的队列)。

> Phase 5 一句话: **桥会生长 —— 当两脑相向而行时。效果 modest(+0.06)但经 5-seed 流程确认为真实,
> 并泛化到另外 3 个决策者大脑(GENERALIZES,均值 +0.081,3/3)。** step3(可塑·双向)如实地推翻了
> step2 的阴性(记录·单向),且跨决策者大脑复现 —— 但若破坏任务则效应消失,故「摇晃仍稳固」是*依赖
> 大脑*的。

---

## 局限(1 seed · descriptive)

1. **单一 RL seed · descriptive.** paired-bootstrap / Holm–Bonferroni 的统计确认是
   multi-seed(3~5)的后续。但因差距大(swap Δ −0.41),方向性结论应当稳健。
2. **#4 Procrustes**(W 位置不变性)未运行。
3. **heading 天花板受限于智能体缺乏记忆** —— 前馈单帧只能表征 4 步轨迹的一部分;recurrent
   智能体或可抬高天花板。
4. **richer-reconstruction 需 well-posed 重设计** —— V2Rich 因 ill-posed 目标崩溃;rich ×
   重建格的公正测量留待 Deferred。
5. **CCM(Phase 5)。** step1 的阳性(B4 的 52%)与共模去除是 1-seed descriptive。step3 的共适应
   (+0.064±0.020,5/5)经*5-seed 流程* multi-seed 确认,**且跨 RL 脑的泛化(RL-seed)已完成 =
   GENERALIZES**(均值 +0.081,3/3 为正)。但 **解释器 LM 被共享** → 泛化仅限*决策者大脑*层面,
   *解释器大脑变动*(LM-seed)未做(队列的队列)。效应*依赖大脑*(大脑 2 基本为零 + 任务崩溃,
   W-budget control 已吸收)。*完全双向*(一开始就同时)亦未做。流程 seed 4 与大脑 2 中 A2 的
   return 略微越过护栏(−23% vs −20%)。

---

## 未来工作 —— V2 之后

详见 [`docs/NEXT_RESEARCH_PROMPT.md`](docs/NEXT_RESEARCH_PROMPT.md):

1. **直接优化因果(Interchange Intervention Training / DAS)** —— 不用重建这个 proxy,而把
   「交换 h_agent 则报告随之改变」作为*训练目标*。优化我们*所测的东西*。
2. **知觉 vs 意图分离 readout** —— 智能体是否同时表征*真实奶酪*(知觉)与*所追目标*(右上),
   忠实解释者报告哪一个,以及能否把二者分开报告。
3. **recurrent 智能体** —— 抬高 heading 天花板;检验不付诸行动的知觉是否被保留。
4. **well-posed 的 richer-reconstruction** —— 修正 V2Rich 的 ill-posed 目标。
5. **收尾轨道** —— 薄的一对做 multi-seed 统计 + #4 Procrustes → workshop short paper。
6. **CCM 完全双向(排队)** —— step3 是分阶段(A1→A2)且仅限一个 LM 块;从一开始就同时训练
   agent+LM+W(崩塌风险↑,分阶段的成功作为基准)。
7. **CCM RL-seed 泛化 ✅ 完成** —— 在*不同 RL 脑* 3 个上确认了共适应增益(GENERALIZES,均值
   +0.081,3/3)。*下一个排队项(队列的队列)*:**解释器(LM-seed)大脑变动** —— 至今只改变了
   决策者大脑,解释器被共享。「两脑都不同也照样生长」才是真正的完成形态。

---

## 复现方法

```bash
# 环境(在 RTX 3070 Ti / WSL2 Ubuntu 上验证,单 8 GB GPU)
conda activate splitmaze        # Python 3.10
# procgenAISC 为 from-source 构建(见 docs/PROCGEN_ENV.md)

# 测试(300+ 单元测试)
PYTHONPATH=src python -m pytest tests/ -q

# Phase 4 决定性测试 + 主动交换(需要已训练的检查点)
PYTHONPATH=src python scripts/eval_builds.py --device cuda --rollouts 20
PYTHONPATH=src python scripts/swap_test.py  --device cuda --rollouts 20 --n_pairs 1000

# 受控 2×2(在冻结的智能体·LM 上 post-hoc 拟合 4 格)
PYTHONPATH=src python scripts/fit_2x2.py --device cuda --rollouts 20 --fit_steps 3000
```

结果在 `results/phase4_*.json`。书面成果为 [`docs/RESULTS.html`](docs/RESULTS.html)
(workshop short-paper 形式 —— 三项发现 + 受控 2×2 + 四张图)。

---

## 技术栈

Python 3.10 · PyTorch 2.x(CUDA)· procgenAISC(from source)· gym3 · NumPy ·
matplotlib · pytest · 单块 8 GB 消费级 GPU(RTX 3070 Ti)/ WSL2 Ubuntu。

---

## 参考文献

* Langosco et al., *Goal Misgeneralization in Deep Reinforcement Learning*, ICML 2022.
* Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022.
* Grill et al., *Bootstrap Your Own Latent (BYOL)*, NeurIPS 2020.
* Chen & He, *Exploring Simple Siamese Representation Learning (SimSiam)*, CVPR 2021.
* Geiger et al., *Causal Abstraction* / *Distributed Alignment Search (DAS)*(interchange intervention training).
* Gazzaniga, *The Bisected Brain*, 1970 / *The Consciousness Instinct*, 2018.
* Turpin et al., *Language Models Don't Always Say What They Think*, NeurIPS 2023.
* Atanasova et al., *Faithfulness Tests for Natural Language Explanations*, ACL 2023.
* (邻接目录 [SPLIT-9](../split_brain_go) · [SPLIT-MNIST](../split_mnist))

---

## 状态

Phase 4(Scenario C)+ **Phase 5(CCM)** 完成。Phase 4: 核心假设(V2)被否 + 受控 2×2 解决
confound + workshop short paper。**Phase 5 CCM**: 记录之桥 = B4 的 52%(step1)· 共模去除机制
(ablation,更正 step1 的解释)· 闭环阴性(step2)· **桥会生长(step3): 共适应 +0.064±0.020,
经 5-seed 流程确认 + 泛化到另外 3 个决策者大脑(GENERALIZES,均值 +0.081,3/3)**。全部反映在
[`docs/RESULTS.html`](docs/RESULTS.html) §3.5。决策者大脑泛化已确认(GENERALIZES,均值 +0.081,
3/3)—— 下一步自然是**解释器(LM-seed)大脑变动**或 CCM 完全双向(见未来工作)。

---

## 许可证

Apache License 2.0。见 [LICENSE](LICENSE)。

```
Copyright 2026 namdo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

本实验的搭建与运行使用了 Claude,帮了非常大的忙。从我 1% 的妄想出发,Claude 让一件真正出色的事成为可能。
