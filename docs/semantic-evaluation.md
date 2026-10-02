# 統合モデルの評価・校正

対象はPC版の `production`。人工文での診断、人手による意味分類の評価、反応判断の比較、参加者実験を区別する。

## 1. 接続と意味分類の診断

```powershell
Set-Location C:\Users\kosuk\nod
.\.venv\Scripts\python.exe -m nod semantic-eval experiments/pilot/semantic-cases.jsonl --out runs/semantic-diagnostic
```

20件の日英人工文をJevへ送る。結果ディレクトリは新規作成のみ。API失敗時はそこで停止し、失敗と試行済みの結果を保存する。人工文のラベルは設計者による仮ラベルであり、独立した意味精度の評価ではない。`--semantic mock` は通信を伴わない配線確認。

## 2. 実験ログを再生し、注釈を作る

```powershell
.\.venv\Scripts\python.exe -m nod replay sessions/対象-live.jsonl
.\.venv\Scripts\python.exe -m nod research-review sessions/対象-live.jsonl --out runs/review-01
```

`replay` は全判断の一致を検査する。`research-review` は同じ入力で full / no_history / direct / argmax / acoustic_only を比較する。動作のACKは比較用のシミュレーションで、反応回数だけから自然さの優劣を結論しない。acoustic_onlyは認識区間・安定性を共通利用し、意味観測を除いた対照条件である。

`annotations.jsonl` には現在文と文脈を、別ファイル `predictions.jsonl` には確率を保存する。注釈者へは予測を見せない。各行に次を記入する。

- `label`：`src/nod/semantic/frames.py` の15種のいずれか。判断不能なら `unclear`。
- `reviewed: true` と注釈者識別子 `annotator`。
- `split`：`development` / `calibration` / `test`。会話・話者単位で分け、同じ発話の重なる途中結果を異なるsplitへ分散させない。

複数注釈者の一致率と不一致の理由を別途確認する。プログラムは「reviewed」の宣言を確認するが、実際に独立した人手評価が行われたことまでは保証しない。予測とラベルの対応はIDで検査し、重複IDを拒否する。

```powershell
.\.venv\Scripts\python.exe -m nod semantic-score runs/review-01/annotations.jsonl runs/review-01/predictions.jsonl --out runs/score-01.json
```

未レビュー行は採点しない。accuracy、NLL、多クラスBrierスコア、混同行列を出す。

## 3. 校正用データで温度を推定する

校正専用ファイルを作り、レビュー済み20件以上を用いる。20件は処理の最低条件であり、統計的に十分という基準ではない。校正に使う全レビュー済み行のsplitが `calibration` である必要がある。

```powershell
.\.venv\Scripts\python.exe -m nod semantic-calibrate runs/calibration/annotations.jsonl runs/calibration/predictions.jsonl --out runs/temperature.json
.\.venv\Scripts\python.exe -m nod live --language ja --calibration runs/temperature.json --open-browser
```

スキーマが一致する成果物のみ読み込む。元ファイルのSHA256を成果物に記録する。同じJevモデル・質問定義・対象言語／課題のデータを使い、変更時は再校正する。この互換性判断は実験者が行う。温度は0.5〜5.0を0.05刻みで探索しNLLを最小化する。未知の評価データで改善を確認するまでは、校正済み成果物の存在を精度改善の証拠とは扱わない。

## 4. 人への効果を評価する

無反応、音声中心、意味を使う条件などの主比較を事前に決める。反応の総量・種類・強度の差が内容対応の効果と混ざるため、回数や動作量を対応させた条件も検討する。応答の適切さ、誤った受領・同意の知覚、聞いてもらえた感覚、練習や語りの課題成績を分けて測る。

人工音声・模擬ACKによるソフトウェア検証と、参加者を使った因果的な効果検証は別の結果として報告する。
