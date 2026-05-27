# SPLIT-MAZE

[🇰🇷 한국어](README.md) · [🇬🇧 English](README.en.md) · [🇨🇳 中文](README.zh.md) · 🇯🇵 日本語

*procgen 迷路の RL エージェントと from-scratch の迷路言語 LM を人工脳梁でつなぎ、目標オーバージェネラライズしたエージェントの真の内部目標を LM が「合理化ではなく忠実に」語れるか ── 両方 from scratch で ── 検証したプロジェクト。*

ラインアップの三番目: SPLIT-9(事後アダプタ・陰性)→ SPLIT-MNIST(共同学習の分離再構成 V2・同質 toy で陽性)→ 本プロジェクト。その V2 パターンを *異質な実環境*(迷路 RL エージェント + 迷路言語 LM、両方 from scratch)で、*目標オーバージェネラライズ*(チーズが常に右上だった迷路で育ったエージェントが OOD ではチーズを無視して右上へ向かう)を忠実 vs 合理化の判別器として再検証した。
三段階。**Phase 4** ── 核心仮説(分離再構成 V2 > next-token アダプタ B4 の忠実性)は棄却(Scenario C)。ただし統制 2×2 で confound を解消し、豊富な陰性を 3 つ得た(目標オーバージェネラライズが表現レベルで読める・「忠実 ≠ 合理化」の再フレーム・ACC の単一ベクトルのボトルネック)。**Phase 5** ── 記録ベースの脳梁(CCM)は両脳が歩み寄ると学習翻訳器の天井まで育つ(共適応 +0.064±0.020、5 seed、決定者ブレイン 3 体に汎化)。**Phase 6** ── ライブ双方向フィードバックで解釈者をより grounded にしようとしたが陰性(Δ +0.037 < 閾値 +0.05 + オウム返しガード失敗;フィードバックは目標 prior 側に偏らせる/オフロードするだけ)。事前登録ガードが spurious な効果を捕えた綺麗な陰性。

**見出し:** Phase 4 V2 棄却(per-slot 0.375 / swap 0.42 vs B4 0.66 / 0.83)· Phase 5 橋が育つ +0.064±0.020(5 seed・3 ブレイン)· Phase 6 R2 陰性(Δ +0.037 < +0.05、in-dist ガード失敗)。

**[詳細を見る → 3 Phase 概要 + 全結果・グラフ(韓/英)](https://halmoneysonmat.github.io/split-maze/)**

---

### 研究ラインアップ

[SPLIT-9](https://github.com/HalmoneySonmat/split-9) → [SPLIT-MNIST](https://github.com/HalmoneySonmat/split-mnist) → **SPLIT-MAZE**

再現・スタックは詳細ページに。テスト: `pytest tests/ -q`。

---

このプロジェクトの作成に Claude を使った。
