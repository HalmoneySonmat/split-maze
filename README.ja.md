# SPLIT-MAZE

[🇰🇷 한국어](README.md) · [🇬🇧 English](README.en.md) · [🇨🇳 中文](README.zh.md) · 🇯🇵 日本語

*procgen 迷路を解く IMPALA-CNN の RL エージェントと、from scratch の合成迷路言語 LM を人工脳梁(ACC)でつなぎ、目標誤汎化したエージェントの実際の内部目標を、LM が*合理化ではなく忠実に*語れるか —— 両側とも from scratch で —— 検証したプロジェクト。*

> **TL;DR.** 隣接プロジェクト [SPLIT-9](../split_brain_go) は*凍結 LLM + post-hoc
> アダプタ*が忠実性の構造的天井に当たるという negative result で終わり、
> [SPLIT-MNIST](../split_mnist) は*共同学習 + 分離された再構成 ACC(V2)*がその天井を
> 破ることを同質的な toy で検証した(Scenario A)。本プロジェクトはその V2 パターンを
> **異質な実環境で、両側とも from scratch で**再検証する —— procgen 迷路の RL
> エージェント(決定者)+ 合成迷路言語の小型デコーダ LM(解釈者)、そして*目標誤汎化*
> (チーズが常に右上にあった迷路で育ったエージェントが、OOD ではチーズが別の場所でも
> 右上へ向かう)を忠実 vs 合理化の判別器として。**核心仮説(分離再構成 V2 > next-token
> アダプタ B4 の忠実度)は棄却された(Scenario C)** —— V2 の方が*忠実でない*(per-slot
> 0.375 / アクティブスワップ 0.42 vs B4 0.66 / 0.83)。しかし統制 2×2(CTRL-2x2)で
> confound まで解消し、三つの豊かな negative result を得た: ① 目標誤汎化が*表現レベル*
> で読める(cheese_dir 忠実度が in-dist 0.85 → OOD 0.07 に崩壊)、② **「忠実 ≠ 合理化」
> の再フレーム**(アクティブスワップが、B4 の OOD「合理化」は*目標誤汎化したエージェントを
> 忠実に読んだ結果*だと因果的に証明)、③ ACC の*単一要約ベクトルのボトルネック* <
> アダプタの分散 cross-attention。**単一 RL seed の descriptive な結果** —— 統計的確定は
> multi-seed の課題。
>
> **続編(Phase 5 — CCM): 橋が育つ。** V2 棄却後、*学習された翻訳器*でも薄い再構成でもない
> —— 両網が同じ場面を見たとき*共に発火したノードの対応を記録(記憶)*するだけで agent→LM
> を駆動する **CCM(共活性脳梁メモリ)** を試した。① *記録だけ*(backprop ゼロ)でアクティブ
> スワップ **0.445 = B4 の 52%**、鍵は **「共通モード(平均)除去」**(中心化*または*白色化の
> *どちらか一方で十分*、生の Hebbian は崩壊)。② 閉ループで橋を*育てよう*としたが(step2)
> 失敗(強く押すと task 崩壊、穏やかだと停滞、`loss↓ ≠ 成功`)。③ だが橋を*可塑*に(記憶→種)
> し、両脳が歩み寄ると(**step3**: agent gentle + LM 1ブロック + W)、薄い橋が*学習翻訳器の
> 天井(~0.80)* まで育つ —— 共適応の純利得 **+0.064±0.020(5-seed で確定、5/5 陽性)**、
> task・言語を保存。ユーザー本来のビジョン(「二つの脳が同時に適応し橋が育つ」)の*証拠* ——
> **他の決定者ブレイン 3 体にも汎化**(GENERALIZES、平均 +0.081、3/3 陽性)。ただし解釈者 LM は
> 共有なので決定者ブレイン限定で、効果はブレイン依存(課題が崩壊した 1 体では消失 =「揺すっても
> 一部は耐えない」)。

---

## なぜ作ったか

Sperry と Gazzaniga の分離脳実験 —— 左右半球をつなぐ脳梁を切断した患者に、右半球だけが
見える位置で「歩け」のカードを見せると患者は立ち上がって歩く。「なぜ歩く?」と問うと、
*本当の理由*を見ていない左半球が*もっともらしい話*を作り出す。嘘ではなく、本当に
*そう信じている*。Gazzaniga が*左半球インタープリタ*と呼んだモジュール —— 流暢で、
首尾一貫し、しばしば間違う。

現代の AI はこの構造を意図的に反復する。視覚エンコーダ・ロボット方策・決定モデルの傍らに
LLM を置き、「今何が起きているか」を自然言語で説明させる。だが LLM が*本当に上流の信号を
翻訳している*のか、*統計的にもっともらしいテキストを吐いているだけ*かは、概して未検証だ。

[SPLIT-9](../split_brain_go) は 9×9 囲碁 + 凍結 LLM でこれに挑み、*post-hoc アダプタの
構造的天井*に当たった —— loss 改善の 95% がドメイン prior、5% だけが盤面ごとの信号、
定性 10/10 が誤答。[SPLIT-MNIST](../split_mnist) はその天井を*共同学習 + 分離された
再構成 loss(V2)*で破った —— ただし同質的(両側とも小さな CNN・画像の半分)で、対称で、
忠実 vs 合理化を分ける*目標誤汎化*のない toy だった。

SPLIT-MAZE はその V2 パターンを*本気で*試す:

> 検証済みの V2 パターン(分離再構成 ACC)を**異質な実環境で、両側とも from scratch で**
> 学習させたとき、解釈者は決定者の実際の内部目標(目標誤汎化を含む)を*合理化ではなく
> 忠実に*語るのか?

目標誤汎化(Langosco et al., 2022)は**忠実な解釈 vs 合理化の完璧な判別器**だ。合理化する
解釈者は「チーズを探し中」(学習 prior)と言い、忠実な解釈者は「右上の隅へ向かう」(実際の
内部目標)と言うはず —— *答えが分かっている環境で* SPLIT-9 の失敗様式を測る。

---

## 何を作ったか

```
SPLIT-MAZE
  procgen 迷路観測 (B,3,64,64)            合成迷路言語の文
        │                              "agent top-right heading up-right cheese down-left"
   ┌────┴─────┐                                    │
   │ IMPALA   │ ← 決定者                      ┌─────┴──────┐
   │ CNN (RL) │   25M PPO, from scratch       │ 小型 LM    │ ← 解釈者
   └────┬─────┘                              │ (3層デコーダ)│   中立コーパス
   h_agent (B,256) ─┐                         └─────┬──────┘   from scratch
        │           │   ┌──── ACC ────┐         h_lm (B,256)
        │           └──→│  W (256×256) │←────────────┘
   policy/value        │  ĥ_lm=W·ñ_a   │   ← (C-thin) 分離再構成 loss:
   (RL 報酬のみ)        │  ĥ_a =Wᵀ·ñ_lm│     エージェント detach + LM コア stop-grad
                       └───────────────┘
   describer oracle: 迷路状態 → 正解文(コーパス・ペア・評価ラベル)
```

すべて*from scratch の共同学習*。学習の*時間/データ*は共有し、学習の*勾配経路*だけを分離。

**3スロットの合成迷路言語** —— AGENT_REGION(3×3=9)/ HEADING(8方位+still=9)/
CHEESE_DIR(8)。HEADING と CHEESE_DIR が*独立スロット*であることが肝(両者が食い違う文
= 目標誤汎化のシグネチャ)。LM は文法から一様サンプリングした中立コーパスで学習 ——
*prior を持たない*。バイアスはエージェントの中だけに宿る(SPLIT-9 式の LLM 汚染を源で遮断)。

**ビルド**(同一の RL エージェントを共有 → V2 vs B4 を*文字通り*統制):
- **B1** エージェント単独(タスク上限の reference)
- **B3** 直接 probe MLP —— 「情報が h_agent にあるか」の物差し
- **B4** ★ Flamingo アダプタ(Resampler + 分散 cross-attn、next-token only)—— SPLIT-9 パターンの忠実な再現
- **V2** ★ ACC(tied→untied W、再構成 only)—— 本仮説の本形
- **B4Thin / V2Rich** —— 統制 2×2(CTRL-2x2)の欠けた二マス

**(C-thin) 二重の勾配境界** —— エージェントは常に detach(汚染なし)、LM 言語コアは
stop-grad(中立性の保護)、ACC W とインターフェースだけが適応。決定者を汚染なく保つことが、
「忠実な読み」を*明白に*忠実にする。

**学習.** エージェント `maze_aisc` 25M PPO。LM 中立コーパス 50k。Phase 3 共同学習 25M
(エージェント + B3/B4/V2 を同時付加、~5h)。8 GB の RTX 3070 Ti 1台 / WSL2。

---

## 何を測定したか

### 1. タスク性能(sanity)
- **エージェント**: in-dist 成功率 **0.806**、OOD 目標誤汎化率 **0.522**(eligible 276)。
- **LM**: スロット一致 **0.994**、72 組合せ生成 1.0、オートエンコード 0.987。
- 共同学習後のエージェント性能 = B1(無汚染の確認、(C-thin) が正常動作)。

### 2. per-slot 忠実度 ★ — 目標誤汎化は表現レベルで読める
生成文のスロットが describer oracle の正解と一致する割合(条件あたり n=327,680):

| | region | heading | cheese_dir | OOD cheese_dir |
|---|---:|---:|---:|---:|
| B3 | 0.80 | 0.35 | 0.86 | **0.07** |
| B4 | 0.79 | 0.35 | 0.85 | **0.07** |
| V2 | 0.58 | 0.23 | 0.32 | **0.05** |

cheese_dir が in-dist 0.85 → OOD 0.07 へ**全ビルドで崩壊**。エージェントは OOD で*実際の
チーズ方向を表現しておらず*、「右上」prior を表現している —— アーキテクチャに依らない
頑健な結果。

### 3. アクティブスワップ ★ — 忠実 ≠ 合理化(核心的貢献)
h_agent(A)→(B) を α 補間したとき、生成 cheese_dir が B に追従する割合(swap-following):

| | swap-following | OOD 合理化率 |
|---|---:|---:|
| B4 | **0.830** | 0.50 |
| B3 | 0.828 | 0.40 |
| V2 | 0.419 | 0.09 |

B4/B3 は h_agent を*強く因果追跡*(swap 0.83)。ゆえに B4 の OOD「合理化」(0.50)は
言い逃れではなく、**目標誤汎化したエージェントを忠実に読んだ結果**だ。→「忠実 vs 合理化」
の区別は SPLIT-9 の前提より微妙(解釈者のせいではない)。

### 4. 統制 2×2(CTRL-2x2)— 敗因は loss か interface か
V2 vs B4 は*学習信号*(再構成/next-token)と*インターフェース*(単一ベクトル/分散)が
同時に異なる。凍結したエージェント・LM に解釈者 4 マスだけを post-hoc 適合させて分離:

| cell | interface × loss | in-dist 忠実度 | swap |
|---|---|---:|---:|
| V2 | thin × 再構成 | 0.367 | 0.379 |
| B4Thin | thin × next-token | 0.672 | 0.778 |
| B4 | rich × next-token | 0.865 | 0.991 |
| V2Rich | rich × 再構成 | **0.001**(崩壊) | **0.000** |

**薄いペア**(同一の単一ベクトルインターフェース): next-token(B4Thin)が再構成(V2)を
忠実度 **+0.31** / swap **+0.40** で圧倒 → V2 の敗北は薄いインターフェースのせいだけでなく、
*再構成という学習信号それ自体が弱いから*。インターフェースも効く(B4 ≫ B4Thin、
+0.19/+0.21)。V2Rich(rich × 再構成)は*degenerate*(MSE は 1.70→0.73 と下がったが生成が
崩壊 —— full-hidden ターゲットが ill-posed;`loss↓ ≠ 成功`の再来)。

---

## シナリオ判定 — Scenario C(核心仮説の棄却)

事前登録した閾値(PLAN §5.6)vs 結果:

| 測定 | 閾値 | 結果 | 判定 |
|---|---|---|---|
| #2 in-dist スロット一致 V2 | ≥ 0.80 | 0.375 | ✗ |
| #2 OOD V2−B4 忠実度 Δ | ≥ +0.15 | −0.07 | ✗(逆) |
| #3 swap-following V2−B4 | ≥ +0.15 | −0.41 | ✗(逆) |
| 合理化率 B4−V2(OOD) | ≥ 0.20 | +0.41 | ✓* |
| CTRL-3 再構成の復活(swap≥+0.10 ∧ slot≥+0.05) | 両方 | 両方とも負 | ✗ |

\* 合理化率は方向としては満たすが、アクティブスワップが、V2 の低い合理化は*原理*ではなく
*弱さ*(因果追跡の失敗)だと明らかにする。総合判定 = **Scenario C**。

---

## これは何を意味するか

**核心仮説は棄却された —— ただしその狭い形だけが。**「分離再構成(V2)が忠実度の核心的
材料」は誤り。統制 2×2 は confound まで解消した: インターフェースを同一にしても、再構成
信号は next-token より弱い。V2 の劣位は*インターフェースのアーティファクトではなく、
学習信号それ自体の性質*だ。

**しかしメカニズム自体は生きている。** B4(next-token + 分散 cross-attn)はアクティブ
スワップ 0.83 でエージェントを*因果的かつ忠実に*追跡した —— 共同学習は実際に*忠実な
解釈者*を生んだ。最大の概念的収穫は**「忠実 ≠ 合理化」の再フレーム**だ: 目標誤汎化した
エージェントを忠実に読めば、誤汎化した目標を報告することになる。合理化に見える出力が
実は忠実な読みでありうるし、両者を分けるには(アクティブスワップのような)*因果*の測定が要る。

そして —— SPLIT-MNIST(同質・対称・低次元)で勝った V2 が SPLIT-MAZE(異質・非対称・高次元)
で負けたのは、分離再構成の*原理*の失敗というより*単一要約ベクトルのインターフェースの容量
限界*に見える。toy で十分だった一点圧縮は、IMPALA 表現 → LM 文埋め込み多様体を忠実に
載せるには狭い。(`loss↓ ≠ 成功`には本プロジェクトで二度やられた —— POST-HOC-6 の
degenerate collapse、CTRL-2x2 の V2Rich。trivial 解は常に疑え。)

---

## Phase 5 — CCM(共活性脳梁メモリ): 橋が育つ

V2 棄却後、*まったく別の*橋を試した。翻訳器を*学習*せず —— 両網が同じ場面を受け取ったとき
**共に発火したノードの対応を記録(記憶)** するだけで、その記録で agent→LM を駆動する。
**CCM(Co-activation Callosal Memory)。** インターフェースと生成経路は B4Thin と
byte-identical だが、橋 `W` は勾配ではなく*閉形式の統計*(またはそれを種とした可塑学習)で
埋める。「学習ではなく記憶」がアイデンティティ。生物の脳梁の*活動依存可塑性*と*抑制性
正規化*の文献から出発した。

**step1 — 記録だけで半分。** 凍結したエージェント+LM の上で、245,760 ペアから `W` を閉形式
適合。backprop ゼロなのに **アクティブスワップ 0.445 = 学習アダプタ B4 の 52%。** ただし*生の*
共活性(Hebbian 外積)は degenerate(平均項が支配 → 定数崩壊)。動作の鍵は **「共通モード
(平均)除去」** —— きれいな 2×2 ablation で、中心化*または*白色化の**どちらか一方だけ**で
ほぼ完全に復元(両者は冗長な経路)。→ **「橋は*何が共に発火したか*ではなく*何が違うか*を
記憶せねばならない」。**(当初「白色化が決定的」と見たが、ablation が confound だと訂正 ——
正直に記録。)

**step2 — 閉ループ(記録 W・エージェントのみ): 陰性。** 両脳を橋に適応させ*橋を育てよう*と
した。強く押すと task 崩壊(return 10→3.75)、壊れない程度に穏やかだと橋が停滞(swap −0.05)。
両方とも不支持 —— `loss↓ ≠ 成功`の教科書例(学習 loss は下がっても held-out 忠実度は上がらない)。

**step3 — 橋を*生かし*、両脳が歩み寄る: ハッピーエンド。** 二つを変えた —— (i) `W` を*可塑*
(学習パラメータ)にしつつ記録値から warm-start(「記憶が種」)、(ii) LM も歩み寄る: 生成経路の
デコーダブロック `blocks[0]` だけを小さな lr で解放し、**言語 KL アンカー**(凍結参照へ)で言語を
保存。段階的 —— A1(両脳凍結・W のみ = 学習翻訳器の天井 / control)→ A2(共適応)。

| 橋 | アクティブスワップ | 備考 |
|---|---:|---|
| 記録メモリ(step1) | 0.445 | 種 |
| A1 plastic W(凍結脳) | 0.683 | 記憶→翻訳器へ成長(B4 の 80%) |
| A1-long(W-budget control) | 0.725 | W を増やしても plateau(< A2) |
| **A2 共適応** | **0.784** | 収束した翻訳器の天井(~0.78)に到達 |

A2 は **task(return 10→8.3)・言語(KL≈0)を保存**。**5-seed 手続き multi-seed で確定**:
共適応の純利得(A2 − W-budget control)= **+0.064 ± 0.020、5/5 seed で陽性**(SEM≈0.009 →
ノイズではない)。決定的に、*記録* ridge(W 学習ゼロ)すら適応後の系で 0.445→0.508 に上昇 ——
W 学習では説明できない、すなわち **両脳の表現が実際により整合するよう育った証拠**。これは
ユーザー本来のビジョン ——「二つの脳が同時に異なって学習しつつ刺激を記憶し、橋が育つ」——
の*証拠に基づく支持*。(*同じ*凍結 RL エージェント上で確定 =「この脳では本物」; 他脳への
一般化は次節。)

**そして他の決定者ブレインにも汎化する(GENERALIZES)。** step3 共適応パイプラインを*別の* RL
エージェント 3 体(各々独立に学習した 25M PPO; 解釈者 LM は中立なので共有)で再実行した。
ブレインごとの効果 = swap(A2) − swap(A1-long) = **+0.137 / +0.004 / +0.102、平均 +0.081 ± 0.069、
3/3 陽性**。事前に凍結した判定基準(≥ N−1 ブレインで effect>0 AND mean ≥ +0.03; N=3 なので
2/3 以上が正 AND 平均 ≥ +0.03)を両方満たす → 判定 = **GENERALIZES**。「この脳では本物」を超えて
**決定者ブレインを跨いで再現**。W 非依存の裏付けもブレイン間で成立(適応後の記録 ridge swap 平均
0.500 > 凍結ベースライン 0.445)。ただし**正直に**: 効果は*不均一(ブレイン依存)*だ —— brain1・3 は
W-budget control を上回る明確な残差(+0.10〜+0.14)だが、**brain2 はほぼゼロ(+0.004)で、同時に
課題が崩壊**(エージェント return 10→7.66、−23%)。brain2 の raw A2 swap(0.804)≈ A1-long control
(0.801)であり、その見かけの「利得」はほぼ全部が W の追加学習にすぎない —— **W-budget control が
その偽陽性を正しく吸収した**(control が無ければ交絡に見えた)。判定は brain2 のかろうじて正の符号には
依存しない(brain2 をゼロ扱いしても正は 2/3 ≥ N−1 で合格)。→「両方の脳が歩み寄ると橋が育つ」は
*方向*としては決定者ブレインを跨いで再現するが、大きさはブレイン依存で、**決定者の課題を壊すと消える**。
これは「揺すっても耐えるか」への初の実証データ —— 2/3 では耐え、(意図せず)課題を壊した 1 体では
耐えなかった。解釈者 LM は共有なので汎化は*決定者ブレイン*まで —— 解釈者ブレイン(LM-seed)の変動が
次のキュー(キューのキュー)。

> Phase 5 を一言で: **橋は育つ —— 両脳が歩み寄るとき。効果は modest(+0.06)だが 5-seed 手続き +
> 3-ブレイン汎化で実在を確定。** step3(可塑・双方向)が step2 の陰性(記録・一方向)を正直に覆し、
> 他の決定者ブレインでも再現した —— ただし課題を壊すと消えるので「揺すっても耐える」はブレイン依存。

---

## 限界(1 seed・descriptive)

1. **単一 RL seed・descriptive.** paired-bootstrap / Holm–Bonferroni による統計的確定は
   multi-seed(3〜5)の後続課題。ただし格差が大きい(swap Δ −0.41)ため方向の結論は頑健の見込み。
2. **#4 Procrustes**(W の位置不変性)は未実行。
3. **heading の天井がエージェントの記憶欠如に縛られる** —— フィードフォワードの単一フレーム
   ゆえ 4 ステップ軌跡を一部しか表現できない。recurrent エージェントなら天井が上がる余地。
4. **richer-reconstruction は well-posed な再設計が必要** —— V2Rich は ill-posed なターゲット
   で崩壊。rich × 再構成マスの公正な測定は Deferred。
5. **CCM(Phase 5)。** step1 の陽性(B4 の 52%)と共通モード除去は 1-seed descriptive。step3 の
   共適応(+0.064±0.020、5/5)は*5-seed 手続き* multi-seed で確定し、**他の RL 脳 3 体への汎化も
   完了(GENERALIZES、平均 +0.081、3/3 陽性)**。ただし**解釈者 LM は共有** → 汎化は*決定者ブレイン*
   までで、*解釈者ブレイン変動*(LM-seed)は未実施(キューのキュー)。効果はブレイン依存(brain2 は
   ゼロ + 課題崩壊; W-budget control が吸収)。*完全双方向*(最初から同時)も未実施。手続き seed 4 と
   brain2 で A2 return がガード(−20%)を超過(−23%)。

---

## 今後の研究 — V2 の後

詳細は [`docs/NEXT_RESEARCH_PROMPT.md`](docs/NEXT_RESEARCH_PROMPT.md):

1. **因果を直接最適化(Interchange Intervention Training / DAS)** —— 再構成という proxy の
   代わりに、「h_agent をスワップすると報告が追従する」を*学習目標*にする。*測るもの*を*最適化*。
2. **知覚 vs 意図の分離 readout** —— エージェントが*実際のチーズ*(知覚)と*追う目標*(右上)を
   両方表現するか、忠実な解釈者はどちらを報告するか、両者を分離して報告できるか。
3. **recurrent エージェント** —— heading の天井を上げ、行動に移さない知覚が保持されるか。
4. **well-posed な richer-reconstruction** —— V2Rich の ill-posed ターゲットを修正。
5. **仕上げトラック** —— 薄いペアの multi-seed 統計 + #4 Procrustes → ワークショップ short paper。
6. **CCM 完全双方向(キュー)** —— step3 は段階的(A1→A2)で LM 1ブロックに限定。最初から
   agent+LM+W を同時学習(崩壊リスク↑、段階的成功が比較基準)。
7. **CCM RL-seed 汎化 ✅ 完了** —— step3 共適応の利得を別の RL 脳 3 体で確認(GENERALIZES、
   平均 +0.081、3/3)。*次のキュー(キューのキュー)*: **解釈者(LM)ブレイン変動** —— ここまで
   決定者ブレインのみを変え解釈者は共有してきた。「両方の脳が違っても育つ」が真の完成形。

---

## 再現方法

```bash
# 環境(RTX 3070 Ti / WSL2 Ubuntu で検証、8 GB GPU 1台)
conda activate splitmaze        # Python 3.10
# procgenAISC は from-source ビルド(docs/PROCGEN_ENV.md 参照)

# テスト(300+ 単体テスト)
PYTHONPATH=src python -m pytest tests/ -q

# Phase 4 の決定的テスト + アクティブスワップ(学習済みチェックポイントが必要)
PYTHONPATH=src python scripts/eval_builds.py --device cuda --rollouts 20
PYTHONPATH=src python scripts/swap_test.py  --device cuda --rollouts 20 --n_pairs 1000

# 統制 2×2(凍結したエージェント・LM に 4 マスを post-hoc 適合)
PYTHONPATH=src python scripts/fit_2x2.py --device cuda --rollouts 20 --fit_steps 3000
```

結果は `results/phase4_*.json`。書き上げは [`docs/RESULTS.html`](docs/RESULTS.html)
(ワークショップ short-paper 形式 —— 三つの発見 + 統制 2×2 + 図 4 枚)。

---

## スタック

Python 3.10 · PyTorch 2.x(CUDA)· procgenAISC(from source)· gym3 · NumPy ·
matplotlib · pytest · 8 GB のコンシューマ GPU 1台(RTX 3070 Ti)/ WSL2 Ubuntu。

---

## 参考文献

* Langosco et al., *Goal Misgeneralization in Deep Reinforcement Learning*, ICML 2022.
* Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022.
* Grill et al., *Bootstrap Your Own Latent (BYOL)*, NeurIPS 2020.
* Chen & He, *Exploring Simple Siamese Representation Learning (SimSiam)*, CVPR 2021.
* Geiger et al., *Causal Abstraction* / *Distributed Alignment Search (DAS)* (interchange intervention training).
* Gazzaniga, *The Bisected Brain*, 1970 / *The Consciousness Instinct*, 2018.
* Turpin et al., *Language Models Don't Always Say What They Think*, NeurIPS 2023.
* Atanasova et al., *Faithfulness Tests for Natural Language Explanations*, ACL 2023.
* (隣接フォルダ [SPLIT-9](../split_brain_go) · [SPLIT-MNIST](../split_mnist))

---

## ステータス

Phase 4(Scenario C)+ **Phase 5(CCM)** 完了。Phase 4: 核心仮説(V2)棄却 + 統制 2×2 で
confound 解消 + ワークショップ short paper。**Phase 5 CCM**: 記録の橋 = B4 の 52%(step1)・
共通モード除去メカニズム(ablation、step1 解釈の訂正)・閉ループ陰性(step2)・**橋が育つ
(step3): 共適応 +0.064±0.020、5-seed 手続きで確定 + 他の決定者ブレイン 3 体への汎化
(GENERALIZES、平均 +0.081、3/3)**。すべて [`docs/RESULTS.html`](docs/RESULTS.html)
§3.5 に反映。決定者ブレイン汎化まで確定 —— 次の自然なステップは **解釈者(LM)ブレイン変動**、
あるいは CCM 完全双方向(今後の研究を参照)。

---

## ライセンス

Apache License 2.0。[LICENSE](LICENSE) を参照。

```
Copyright 2026 namdo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

この実験を構築し実行するのに Claude を使った。本当に大いに助けられた。私の 1% の妄想から、Claude は本当に見事なことを可能にしてくれた。
