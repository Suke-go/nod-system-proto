# Jevが使えるようになったら行う実験

## 1. いま準備できているもの

`C:\Users\kosuk\nod` で実行します。既存の `.venv` とローカルモデルを使用できます。

| 部分 | 実験構成 |
|---|---|
| 音声 | 16kHz / mono / PCM16 WAV、1〜30秒。まず12秒 |
| ASR | faster-whisper base、日本語、CPU int8、4 threads |
| タイミング | MaAI 0.2.10、日本語VAP-BC、CPC、10Hz、20秒文脈、CPU 2 threads |
| 意味機能 | Jev Choice 4分類。接続前はmockで配線確認 |
| 強度 | ACN資産未取得のため既定値。ACN条件の実験として扱わない |
| 出力 | ローカルWebSocketとタイマー式コントローラ。可視アニメーションなし |

ASRとVAP-BCのモデルrevision・ファイルSHA256は `models/local-audio/manifest.json` に固定しています。
推論ではローカルパスを明示します。セットアップの `models-prepare` だけが重みをダウンロードします。

## 2. 先にJev単体を確認する

```powershell
cd C:\Users\kosuk\nod
.\.venv\Scripts\python.exe -m nod jev-eval .\experiments\pilot\cases.jsonl --semantic jev --prompt-key --limit 8 --out .\runs\jev-pilot-01
```

APIキーを非表示で入力します。最大8件、再試行なし。エラーが1件でも出たら、その時点で止めて結果を保存します。
次回は `jev-pilot-02` など新しい出力先を指定します。既存結果を上書きしません。
24件すべて試す場合は `--limit 24`。それぞれ独立した例文で、前の例の内容を持ち越しません。

生成されるファイル:

- `run.json`: 設定・質問hash・データhash。
- `results.jsonl`: 全分布、返却モデル名、応答時間、エラー。
- `review.csv`: 目視確認しやすい一覧。
- `summary.json`: 成功/失敗数、遅延の分位点、人手確認済み項目の精度・Brier score・NLL・混同行列。

同梱例文のreference_labelは**仮ラベル**です。人が確認した行だけlabel_statusを`reviewed`に変更します。
初期状態では精度はnullになり、仮ラベルを正解として数えません。
`--semantic mock` は接続テスト用です。意味分類の性能測定ではありません。

## 3. 録音する

```powershell
.\.venv\Scripts\python.exe -m nod audio-devices
.\.venv\Scripts\python.exe -m nod record .\recordings\pilot01.wav --seconds 12 --device 1
```

このPCでは確認時点のdevice 1が内蔵マイクで、16kHz mono入力に対応しています。デバイス番号は接続状態で変わるため一覧で確認します。
`record`を実行した時だけ録音します。現在の作業ではユーザーのマイク録音は行っていません。
録音台本は `experiments/pilot/recording-prompts.md`。説明、話の途中、言い直し、喜び、残念な出来事、意味の反転を含みます。
必要なら同じ話を「通常」「強調あり」で分けて録音します。

## 4. ローカルで観測を作る

```powershell
.\.venv\Scripts\python.exe -m nod prepare-audio .\recordings\pilot01.wav --out .\runs\pilot01.timeline.json
```

1秒ごとの音声prefixをASRに渡し、100msごとの音声chunkをVAP-BCに渡します。
モデルにその時点より未来の音声を与えません。最後の100ms未満の端数はVAPへpaddingせず、ASRの最終結果には含めます。
ASRが前の要求を処理中なら次の定期要求を省略し、最終要求のみ順番を待ちます。

タイムラインには音声sample終端、source時刻、測定した処理時間から計算したavailability時刻を保存します。
ASRとVAP-BCは別workerとみなして各々の遅れを累積します。これは**録音による因果的な処理の模擬実験**であり、
実時間の並列推論やCPU競合の測定ではありません。準備段階のモデル読込時間も実験の反応時間に含めません。
処理が5秒を超えて音声から遅れた場合、その準備処理はエラーで終わり、有効なtimelineとして保存しません。

1クリップを1発話と扱う初期実装です。自然な発話分割や連続会話は今後の工程です。
最終語境界はASR由来の参考情報です。日本語の正解境界やACN用に確定した境界とは扱いません。

録音前に試すための `experiments/pilot/synthetic-ja.wav` と `synthetic-ja.timeline.json` も配置済みです。
Windowsの日本語TTSで作った架空の例文で、ユーザーの録音ではありません。
次節のtimeline引数をこのファイルに置き換えれば、録音せずに試せます。

## 5. 同じ観測で制御実験する

```powershell
# Jev接続前の配線確認
.\.venv\Scripts\python.exe -m nod experiment .\runs\pilot01.timeline.json --semantic mock --log .\runs\pilot01-mock.jsonl

# Jevが利用可能になった後
.\.venv\Scripts\python.exe -m nod experiment .\runs\pilot01.timeline.json --semantic jev --prompt-key --log .\runs\pilot01-jev.jsonl

# 集計と再現確認
.\.venv\Scripts\python.exe -m nod analyze .\runs\pilot01-jev.jsonl --out .\runs\pilot01-jev.summary.json
.\.venv\Scripts\python.exe -m nod replay .\runs\pilot01-jev.jsonl
```

録音・特徴・VAP結果はローカルで処理します。Jevへ送信するのは発話文と過去の文脈です。
`experiment`は観測を時間通りに投入し、実際のJev応答を待ちながら判断します。WAVの再生音は出しません。
異なる設定を比較する場合は、同じtimelineに対して `--config` を変え、異なる出力先に記録します。

```powershell
.\.venv\Scripts\python.exe -m nod --config .\configs\comparison.json experiment .\runs\pilot01.timeline.json --semantic jev --prompt-key --log .\runs\comparison.jsonl
```

`comparison.json` は `src/nod/resources/default.json` のコピーから作り、1条件ずつ変更します。
動作が0回でも、自動的に閾値を下げません。`analyze` のsuppression_reason_event_countsとログで原因を調べます。
機会窓なし、古い意味推定、ASR信頼性不足、cooldown、効用によるNONEなどを区別できます。
抑止理由はイベント単位の件数であり、独立した発話数ではありません。

## 6. まず見る指標

1. Jevの成功率と応答時間。800ms期限に収まるか。
2. ASRの修正、sourceから結果までの遅れ、Jevの古い応答破棄。
3. 相づち機会の検出と実行回数。見逃し・不要な実行を人が音声と照合する。
4. 意味機能の妥当性。内容への理解と賛成を分け、喜びと困難の両方を見る。
5. ACN接続後に限り、強調条件による強度と自然さを比較する。

このPCの7.345秒の日本語合成音声で、再測定時のVAP-BCはp50=27ms/p95=32ms、ASRはp50=927ms/p95=1119msでした。
測定回数は73/7と小さく、自然会話や並列実行の保証値ではありません。
初回に1frameで約304秒の停止が観測され、原因は未特定です。5秒超の遅れを検出する処理を追加し、再測定しました。
base ASRには認識誤りもあり、Jevと合わせた鮮度1500msは評価すべき制約です。
今回の録音経路のmock実験は0動作・ログ再生一致でした。相づちを強制して成功扱いにはしていません。
合成イベントの従来デモでは引き続き2動作が確認できます。

## 再セットアップ

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-audio.lock
uv pip install --python .venv\Scripts\python.exe --no-deps -e .
.\.venv\Scripts\python.exe -m nod models-prepare
.\.venv\Scripts\python.exe -m pytest -q
```

上流資料: [MaAI BC](https://github.com/MaAI-Kyoto/MaAI/blob/main/readme/vap_bc.md)、
[配布モデル](https://huggingface.co/maai-kyoto/vap_bc_jp)、
[faster-whisper](https://github.com/SYSTRAN/faster-whisper)、
[元のVAP](https://erikekstedt.github.io/VAP/)。
MaAIは発話交替用VAPの確率をそのまま相づち確率と見なす代わりに、公式のBCモデルを使っています。
利用・発表時は各モデルの配布条件と引用要件を確認してください。
