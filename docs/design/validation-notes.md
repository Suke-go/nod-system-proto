# 設計資料の整合確認

確認対象は設計ファイルと式であり、アプリケーションの動作試験ではありません。

- 4つのJSONファイルを読み取り可能。
- JSON Schema Draft 2020-12の構造が有効。
- 7種類の合成レコードがschemaに適合。
- 範囲外強度、送信action=NONE、負のopportunity、確率キー不足の4例をschemaが拒否。
- クラス順序・Jevモデル名・初期設定が資料間で一致。
- 遷移行列の行和・非負性、時間分割不変、q=0のprediction-onlyを算術確認。
- 5つの意思決定例について効用と選択結果を再計算。

## 合成policy例

- P01: continuer
- P02: understanding
- P03: empathic
- P04: none
- P05: none

## 未実施

- Jevの実API呼び出し、日本語の精度評価、API遅延測定。
- ACN/VAP/ASRの実モデル推論、同時実行測定。
- scheduling/motor/replayの実装テスト。22ケースの受入条件を定義した段階。
- 物理アバターへの命令送信。

閾値・utility・TTL等は暫定値であり、本確認で適切さを実証したものではありません。
