# 『「生まれ」と「育ち」の数理』付録E のコード

本書の図と数値を再現する Python のコードです（第4.2版）。

- 必要なもの：Python 3、numpy（E6 と、E4 の第11章の表は scipy も。E4 の図は matplotlib があるときだけ描く）
- 実行：`python3 E1_kalman_gain.py` のように、一つずつ実行します。どれも数秒〜数十秒で終わります
- 各ファイルの末尾のコメントに「一つ変えてみる」の課題があります

| ファイル | 再現するもの |
| --- | --- |
| E1_kalman_gain.py | 第3章 図3-1 |
| E2_bet_hedging.py | 第6章 図6-1 |
| E3_forgetting.py | 第8章 図8-1 |
| E4_three_layers.py | 第10章 図10-1・10-2、10.5節の表、第11章11.4の表、応用1の表（モンテカルロ法を含む） |
| E5_alignment.py | 発展1 |
| E6_constraint_cost.py | 発展2、第11章の表 |
| E7_diversify.py | 発展6 |
| E8_kelly_cue.py | 第6章、付録B.5 |

---

## ライセンス
本フォルダ内のコードは **MIT License** の下で公開されています。学習・研究目的での自由な改変と実行を歓迎・推奨します。
