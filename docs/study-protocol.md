# 次の実験の手順 — research-v2

現在の条件定義は [research-v2.md](research-v2.md)、ACNの位置づけは [acn-role.md](acn-role.md) を参照。

## 起動と条件固定

```powershell
Set-Location C:\Users\kosuk\nod
.\.venv\Scripts\python.exe -m nod live --language ja --response bayes --open-browser
# 主対照: 同じ更新式でフィルタの履歴だけ外す
.\.venv\Scripts\python.exe -m nod live --language ja --response bayes-no-history --open-browser
# 意味モデルの直接利用
.\.venv\Scripts\python.exe -m nod live --language ja --response direct --open-browser
# プレゼン練習の別条件
.\.venv\Scripts\python.exe -m nod live --language en --response presentation --open-browser
# 英語は --language en
```

`start-ja.cmd` / `start-en.cmd` も既定のBayes研究版を起動する。画面に「Bayes研究版 v2 / 履歴あり」と表示される。モデル変更を実行中のセッションへ混ぜず、起動し直して新しいログへ分ける。完全な設定例は `configs/bayes.json` など。`--config configs/bayes.json live ...` で読み込める。

意味評価の生スコア、予測分布、更新後分布、ASR信頼度、確定状態、休止状態をログで追う。`semantic_trace` が更新過程、`decision_status` が抑制条件、`semantic_refinement` が同文の確定情報を反映した処理。

## オフラインの診断とラベル付け

終了済みのログを指定する。出力先は新しいディレクトリにする。

```powershell
.\.venv\Scripts\python.exe -m nod research-review sessions/対象-live.jsonl --out runs/review-001
```

- `comparison.json`: 更新式、事前分布、終了後拡張、確定情報の反映の比較。模擬ACKによる再計算で、実ロボットの観測結果とは別。
- `conditions.json`: 各条件の完全な設定。
- `annotations.jsonl`: 発話ごとに採用済みの要求を1件選び、その要求が見た認識文と文脈を提示。モデルの予測は表示しない。ラベルはすべて未入力。
- `predictions-by-condition/`: 同一ラベル項目に対応する各比較条件のraw/filtered分布。research-scoreの第2引数に指定できる。
- `predictions.jsonl`: ラベルから分離した元モデルの出力。評価者はラベルを確定するまで見ない。

ラベルは `no_bc / continuer / understanding / empathic`。評価者は `label`、`annotator` を記入し、判断を確定したものだけ `reviewed: true` にする。後から得た発話の結末を使わず、その要求時点の文脈で判断する。曖昧な例は保留し、保留率・除外理由も報告する。2名以上が独立に評価し、不一致を記録してから合議することを推奨する。ツールは合議結果の単一ラベルを評価する段階で使う。

```powershell
.\.venv\Scripts\python.exe -m nod research-score runs/review-001/annotations.jsonl runs/review-001/predictions.jsonl
```

未確認のラベルを正解には使わない。確認済み0件なら精度は未算出。出力はraw/filteredそれぞれのAccuracy、NLL、Brier score、混同行列。調整済みの既存ログはdevelopmentと表示され、評価用の未知データとは呼ばない。現ツールはコーパス全体の最終論文評価器ではなく、開発段階の意味診断用。

発話単位の代表要求は、確定フラグ、ASR信頼度、音声時刻の順で選ぶ。このサンプリングはオンラインの全ての部分認識を代表しない。反応時点の意味精度を評価するときは、別途、実際の判断時刻の文脈を評価対象に含める。

## 有効性を調べる実験

1. 開発ログと切り離した新しい話者・セッション・文章を確保する。意味ラベルと自然さの評価方法を先に固定する。
2. 同じ音声・同じASR/Jev出力によるオフライン比較では、変更する要因を1つずつにする。精度改善と動作回数増加を区別する。
3. ライブのユーザ実験では条件順をランダム化またはカウンターバランスする。日本語/英語、話者、文章、セッションを保持し、同一話者の繰り返しを独立サンプルとして数えない。
4. 意味の適切さ、時間的な適切さ、自然さを別々に評価する。頻度・遅延・誤反応・無反応・ASR修正・API失敗も併記する。まずパイロットで分散と評価一致を確認し、その後に必要人数を設計する。未実施の検出力計算に基づく人数は断定しない。
5. 主要比較は同じ更新式でフィルタの履歴あり/なし。Jevに渡す文脈、タイミング、効用、強度、動作後休止はそろえる。プレゼン用の文区切り拡張、終了後拡張、更新式と事前分布は別分析にする。反応回数/分、命令上の動作時間も併記する。複数要因をまとめて変えた条件だけから個別要因の効果を主張しない。
6. 自然さやタイミングのラベルには音声/映像が必要。現在の自動ログは音声を保存しない。別途同意を得た実験録音を用意する。テキストログだけで自然さを採点しない。

実験パラメータを見直したらprotocol_versionを更新し、調整前後の結果を分ける。採点前の正解ラベル、調整済みデータ、未測定の実機遅延から精度や効果を作らない。
