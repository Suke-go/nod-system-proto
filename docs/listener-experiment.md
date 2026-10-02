# nod 0.8 の実験

現在の既定は0.8です。[今回の修正設計](incremental-listener.md)に、内容範囲の保持・改訂と、頷き／表情の条件の違いを記載しました。以下の0.7の説明は基礎モデルと旧版の記録です。

# PC版の起動・比較・校正

既定はnod 0.7のベイズ聞き手モデル。`start-ja.cmd` / `start-en.cmd`で起動する。画面には知覚P、理解U、理解を伴う態度の信念を分けて表示する。旧15分類は `--response legacy-production` で選べる。

```powershell
Set-Location C:\Users\kosuk\nod
.\.venv\Scripts\python.exe -m nod live --language ja --open-browser
.\.venv\Scripts\python.exe -m nod live --language en --condition acoustic_only --open-browser
```

`acoustic_only`はJevを呼び出さずキーも不要。ASR区間・信頼性・VAPは共通利用し、意味観測を外す比較条件である。カメラ・実機は今回の範囲に含まない。Jevを使う条件では認識テキストと直近文脈を公式APIへ送る。

## 技術検証

```powershell
.\.venv\Scripts\python.exe -m nod listener-eval experiments/pilot/listener-cases.jsonl --out runs/listener-diagnostic
.\.venv\Scripts\python.exe -m nod replay sessions/対象-live.jsonl
.\.venv\Scripts\python.exe -m nod research-review sessions/対象-live.jsonl --out runs/listener-review
```

出力先は新しい名前を指定する。診断は12件の日英人工文で、独立した意味精度や心理的妥当性を測ったものではない。replayは全判断の一致を確認する。reviewは同一ログをfull / no_history / argmax / acoustic_onlyで再計算し、模擬ACKで動作量・種類・時刻を比較する。argmaxは分類器の出力を最尤ラベルへ量子化するアブレーションで、別途最適化された古典分類器との比較ではない。

## 人手評価と観測モデルの校正

reviewの`annotations.jsonl`には当時の入力・文脈・ASR信頼性を、`predictions.jsonl`には観測・フィルタ出力を分けて保存する。注釈者には予測を見せない。`state`を`belief/listener.py`の状態名で記入し、`reviewed: true`、`annotator`、`split`を設定する。

注釈対象は「この時点の証拠からどこまで受領・解釈を支持できるか」という操作的な基準である。Pの正しさを採点するには、音声または独立した参照転記とASRを比較する必要がある。ログだけから実際の聞き取り成功や隠れた心理状態の正解を作らない。参照がない場合は未レビューのまま残す。複数注釈者の一致度と不一致の理由を確認する。

話者・セッション単位でdevelopment / calibration / testへ分割する。同じ発話の重なる途中結果を分割してリークさせない。各状態8件以上というプログラム上の最低条件は、統計的に十分という基準ではない。

```powershell
.\.venv\Scripts\python.exe -m nod listener-score runs/review/annotations.jsonl runs/review/predictions.jsonl --out runs/state-score.json
.\.venv\Scripts\python.exe -m nod listener-fit runs/calibration/annotations.jsonl runs/calibration/predictions.jsonl --out runs/observation-model.json
.\.venv\Scripts\python.exe -m nod live --language ja --observation-model runs/observation-model.json --open-browser
```

fitはレビュー済みcalibration行だけで平均と共分散を推定する。共分散は対角への20%縮小と0.1のridgeを用いる。理解未確定の状態ではP/Uの平均を用い、態度の平均は理解可能な各態度状態から周辺化する。成果物は分類器のモデルID・質問定義のSHA256・状態順序が一致する場合にだけ読み込む。遷移・効用はこの処理では学習しない。言語・対象課題・話者分布の適用可能性は実験者が判断し、独立したtestデータで評価する。

## HRIの主比較

主比較はfullとacoustic_only。前者は意味を観測して状態更新するPC聞き手、後者は意味を使わない聞き手である。まず少人数の開発用試行で機構と失敗条件を確認し、その後に評価指標と解析を固定する。PCアバターでの結果の範囲は仮想聞き手との相互作用であり、身体を持つロボットへ一般化したとは主張しない。

主要評価には第三者による反応の内容適合性・誤った同意の知覚と、話者の「聞いてもらえた」評価を置く。副次評価には練習課題の遂行、割込み感、音声から反応までの遅延を置く。状態分布の正確さ、反応の適切さ、利用者への効果は分けて報告する。

意味観測の有無で反応の種類・量・タイミングも変わるため、full対acoustic_onlyだけで意味対応の純粋な効果を断定しない。録画刺激を用いた動作量対応・別内容への反応再生などの対照を加える場合は、その条件生成と提示手順を別途固定する。この版はライブの意味なし条件と同一入力での再計算までを提供し、参加者割付や動作量対応条件を自動生成する実験管理システムではない。

API遅延は意味センサーだけの時間。画面の描画開始と実サーボの動作開始は計測していない。命令時刻や模擬ACKを人が反応を見た時刻と同一視しない。
