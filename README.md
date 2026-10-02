# nod 0.9 — 高速なローカル認識と意味に応じた表情

現在の既定は0.9です。SenseVoiceによる400ms間隔の音声認識、分類分布を受け取る観測密度、小頷きから表情への更新を導入しました。[設計・計測の限界・比較方法](docs/speed-accuracy-v4.md)を参照してください。起動は `start-ja.cmd` / `start-en.cmd` のままです。以下は旧版の説明・記録です。

# nod 0.8 — 逐次的な内容保持と表情反応

0.8の設計記録です。[今回の修正設計](docs/incremental-listener.md)に、内容範囲の保持・改訂と、頷き／表情の条件の違いを記載しました。以下の0.7の説明は基礎モデルと旧版の記録です。

# nod 0.7 — 意味センサーを使うベイズ聞き手

**ローカル音声処理 → 知覚・意味の観測 → ロボットの信念更新 → 頷き・表情**を実行するPC傾聴エージェントです。音声入力は実験上の選択で、Jevは意味センサーの一実装です。

既定の `production` は、Koppらの知覚・理解・態度を区別する聞き手モデルと、傾聴対話の確率的制御を参照した有限状態ベイズフィルタです。分類器の分布をそのまま尤度とせず、相関を扱う観測密度を介して状態を推定します。観測の重複、認識の訂正、鮮度、実行済み反応を扱います。

- [定式化・理論・実装の対応](docs/listener-model.md)
- [PC起動・比較実験・観測モデルの校正](docs/listener-experiment.md)
- [今回の検証結果](docs/listener-validation.md)

起動は `start-ja.cmd` / `start-en.cmd`。既存セッションは終了し、次回起動から新モデルを使います。旧15分類は `--response legacy-production`、さらに以前の4分類は `--response bayes` で選択できます。旧ログの再生も保持しています。

係数は既定では設計値で、心理的妥当性と参加者への効果は未検証です。ACNは任意の強調度入力で、未接続時も意味に応じた強度を使えます。今回はPC版が対象です。

## 起動

Python 3.11以上。PowerShellで:

```powershell
Set-Location C:\Users\kosuk\nod
.\.venv\Scripts\python.exe -m nod doctor
.\.venv\Scripts\python.exe -m nod live --language ja --open-browser
```

上のコマンドはマイクを使用します。音声処理はローカル、認識テキストと直近の文脈だけがJevへ送られます。
`demo` は約6秒で、2回の別々の合成相づち機会を再現します。ローカルWebSocketを実際に使います。
同じSMALL_NODでも別のaction_idで実行され、各機会で1回だけ出力します。
既定の待受は127.0.0.1:8765。競合時は `demo --port 0` で空きポートを選びます。
結果の `log` に表示されたJSONLを使って、外部接続なしに判断を再現できます:

```powershell
.\.venv\Scripts\python.exe -m nod replay .\sessions\表示されたファイル名.jsonl
```

ログは上書きしません。発話文・確率・入力イベント・判断・動作を記録します。
音声やAPIキーは記録しません。`replay` はJevもコントローラも呼び出さず、全derivedレコードの一致を検証します。

コントローラを別プロセスで試す場合は `demo --external-controller` を起動し、15秒以内に別のPowerShellで
`.\.venv\Scripts\python.exe -m nod controller` を起動します。

## Jevへの接続

`TYPESAFE_API_KEY` 環境変数、または起動ディレクトリの `.env` を読みます。指定ディレクトリの `.env` は設定済みです。
`--prompt-key` で非表示入力した場合はメモリにだけ置き、保存しません。キーの値をログや画面に表示しません。

```powershell
.\.venv\Scripts\python.exe -m nod jev-check --prompt-key
.\.venv\Scripts\python.exe -m nod demo --semantic jev --prompt-key
```

`jev-check` は固定の日本語例文を1件送ります。`demo --semantic jev` はデモ中の発話文を送信します。
Jevが返す確率によって動作数は変わります。2回の動作を保証するのはmockデモだけです。
キー入力を求めるのは上記オプションを付けた場合だけです。

HTTPXで公式の `POST https://api.typesafe.ai/v1/systemone` に接続します。
SDKの自動再試行は使用しません。モデル既定値は `jev-1.13.0`。
ライブ統合モデルはAPI期限1800ms・意味入力鮮度3500msです。従来のdemo/録音プロファイルは800ms・1500msです。
同時実行1件・最新の待機1件・入力の鮮度・ASR修正を検査します。
401/403を含む恒久エラーで意味推定を停止し、429/5xx等は次の新しい入力までバックオフします。
HTTP失敗を「no_bcの観測」に変換しません。
参考: [TypeSafe API](https://docs.typesafe.ai/api)

## テストと再セットアップ

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

仮想環境を作り直す場合:

```powershell
python -m venv .venv
# uvがある場合
uv pip install --python .venv\Scripts\python.exe -r requirements.lock
uv pip install --python .venv\Scripts\python.exe --no-deps -e .
# uvがない場合
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

`requirements.lock` は今回の検証環境のバージョン固定です。Python 3.11/Windowsで確認しています。
音声実験を含む環境の固定値は `requirements-audio.lock` です。再構築時はそちらを指定してください。
今回のPCではCPUを使います。推論時の重み自動ダウンロードは行いません。

## 設定とコード

- `src/nod/resources/default.json`: 実行時の既定値。変更用に複製し `python -m nod --config 設定.json demo` で指定できます。
- `src/nod/resources/jev-observation.json`: 現在の意味センサー定義。旧15分類・4分類の定義も履歴再生用に保持。
- `src/nod/core/listener.py`: 現在の状態更新。`core/engine.py`がログ・設定に応じて旧方式と選び分ける。
- `src/nod/semantic/`: Jev HTTPアダプタと要求スケジューラ。
- `src/nod/belief/`, `timing/`, `policy/`: 信念、機会窓、効用、動作状態。
- `src/nod/output/transport.py`: WebSocketとシミュレータ。
- `docs/design/`: 先に作成した全体設計。
- `docs/implementation-d1-d4.md`: 実装範囲と残作業、時刻・通信契約。

## 評価と今後の拡張

[検証結果](docs/listener-validation.md)に、人工音声での一連の処理、APIの失敗例、テスト結果を記録しています。[評価手順](docs/listener-experiment.md)から人手ラベル付け、比較再生、確率校正を進められます。自然な会話での適切さ、利用者への効果、長時間の負荷は今後の評価対象です。

ACNには重み、学習時の正規化、特徴抽出条件、strengthへの変換を明記したモデルmanifestが必要です。現状はprominence特徴の既定値0.25を使い、ログにdegradedを明示します。動作強度そのものは意味への支持に応じて変化します。

既存のACN参照実装は `work/acn_reference/` にあります。ディレクトリの旧名 `esa_teo_detector` とモデル名ACNは区別します。
MLPの軽さだけで、単語区切り・前後語の特徴・学習時の正規化を省略してよいとは扱いません。
