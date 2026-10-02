# nod 実装計画

更新日: 2026-09-18

実装先: `C:\Users\kosuk\nod`

## 1. 確定した方針

- JevのみTypeSafeのオンラインAPIを利用する。
- Jevのアクセス権はユーザーが取得済み。実接続確認を実装工程に含め、アクセス申請待ちとは扱わない。
- マイク取得、VAP、ASR、ACN、belief、policy、アバター出力、画面、記録はローカルPCで実行する。
- ProminenceモデルはINTERSPEECHのAuditory Contrast Network（ACN）。`esa_teo_detector` は保存先に残る旧名称で、採用モデル名ではない。
- SSHは元資産の取得に必要な場合だけ使用し、実行時の依存にしない。
- WHEN / WHAT / HOW STRONGを独立した状態として保持する。Jevには動作を生成させない。
- この文書は実装計画。モデルの学習・API呼び出し・依存導入・実行性能の測定はまだ行っていない。

## 2. 確認したローカル資産

参照ルート: `C:\Users\kosuk\nod\work\acn_reference`

| 資産 | 確認内容 |
|---|---|
| `acn/acn_v8_spectral.py` | 音響特徴のみのACN候補。`ACNv6`クラスをNumPyで実装。 |
| `acn/acn_v9_textfeat.py` | 音響特徴と任意のテキスト特徴を扱う`ACNv9`。 |
| `interspeech_2026_acn/main.tex` | 論文原稿のモデル・特徴量・実験条件・制約を確認。PDF版との一致は未検証。 |
| コピーされたファイル一覧 | Python実験スクリプト、論文原稿・PDF等。学習済み重み、scaler、音声からのMFCC抽出スクリプトは確認できない。 |

以下の行番号は今回の参照コピーを対象とする。

### ACNについて実装前に解決する点

1. **単語単位の入力である。**
   - 原稿 `main.tex:173–185` では、log単語長、単語内MFCC_0の平均、MFCC_2の標準偏差を使用する。
   - 25 ms窓・10 ms hop・13係数と記載されているが、sample rate、FFT長、Melフィルタ、padding等の完全な抽出条件は元の抽出コードで確認する。
   - 単語境界は原稿ではMontreal Forced Alignerに依存する。VADの発話区間だけでは同じ特徴を作れない。

2. **前後1単語を使う双方向モデルである。**
   - `acn_v8_spectral.py:61–75`、`acn_v9_textfeat.py:70–92` は前後のcontextを入力する。
   - 次単語の特徴を使うには、その単語の終わりと境界確定を待つ必要がある。MLP自体が軽くても待ち時間はゼロにならない。
   - 前方向のattentionが大きいという原稿の結果だけでは、次単語を除いたモデルと等価とはいえない。

3. **正規化の記述とコードに差がある。**
   - 原稿は話者別z正規化。
   - v8 `run_exp` およびv9 `prep_ctx` はHelsinki側の特徴全体に`StandardScaler.fit`を行い、そのscalerをEmphases側に適用する。
   - 採用checkpointに対応した処理を優先する。実装時に話者別やオンライン正規化へ無断で変更しない。

4. **出力はそのまま[0,1]のスコアではない。**
   - 最終層は線形回帰で、出力範囲は制約されていない。
   - `fit`は教師値を平均・標準偏差で正規化するが、その統計量は現在の`save()`に含まれない。
   - 元尺度の復元が必要なら教師統計量を保存する。動作強度への変換は別の校正器として定義し、raw scoreも記録する。
   - Pearson相関が良いことと、動作強度として校正されていることは別に評価する。

5. **`save()`はディスク保存ではない。**
   - v8 `109–116`、v9 `120–127` は重み辞書のメモリ内コピー／復元。
   - コピーされたv8/v9の実験処理では、採用モデルの重み・scalerを推論用ファイルとして保存する工程を確認できない。
   - 既存の書き出し済み資産があれば取得する。存在しなければ、学習資産を確認して別工程でexportする。ランダム初期化を実モデルとして扱わない。

6. **採用構成をファイル名から決めない。**
   - 原稿の音響版は202パラメータ、音響＋テキスト版は238パラメータの記載。
   - コードでは`nc/hc/ha/nt`により構成が変わる。例えば音響3cue・hc=8・ha=8は170、ha=12は202となる。
   - v8には複数の特徴・hidden幅、v9には複数のテキスト条件がある。採用した構成、seed、単体／ensembleとcheckpointを一組で固定する。

7. **日本語への適用は追加評価が必要。**
   - 原稿の評価対象は英語朗読。日本語の自発発話、ASRで推定した単語境界への性能は未確認。
   - 日本語での区切り単位を決め、境界誤差と強度の自然さを評価する。英語での報告値を日本語の保証にはしない。

## 3. 採用する構成

| モジュール | 初期方針 | 決定前に確認すること |
|---|---|---|
| Audio | ローカルで一度だけ取得し、共通バッファから各workerへ配信 | デバイス、sample rate、各モデルへのresampling |
| WHEN | 原著VAPの基準推論を確認し、日本語のMaAI BC adapterを第一検証候補にする | BC確率の定義、horizon、無音のlistenerチャネル、推論速度 |
| ASR | ローカルモデルとストリーミング処理を組み合わせる | 日本語partial、revision、単語境界と時刻の精度、遅延、CPU/GPU |
| WHAT | Jev Choiceで4分類の全確率を取得 | APIキー、日本語評価、校正、期限付き非同期処理 |
| HOW STRONG | 音響のみのACNをCPU実行する案 | 採用重み、scaler、特徴抽出、単語境界、先読み、[0,1]変換 |
| Belief / Policy | ローカルの数値処理 | 時間遷移、観測重み、効用表、重複抑止 |
| Avatar / Dashboard | localhost WebSocketを初期方式にする | 実動作の開始／完了通知、実行競合 |

ASRは`faster-whisper`等を比較候補とし、ストリーミング層としてSimulStreaming等の適合性も確認する。単にバッチASRが速いという理由だけでは採用しない。PCのGPU/VRAM/RAMは未確認で、モデルサイズと更新間隔は同時実行の測定後に固定する。

Jevは`state`にstable transcript、current partial、recent contextを入れ、`Choice`のcriteriaを4分類にする。実際の返却モデルID・質問定義の版・入力revisionを記録する。JevのconfidenceはASR reliabilityと分けて扱う。

音響のみのACNは意味内容を入力しない。ただし、V1でASRの時刻情報を単語境界に使う場合は、境界が得られるまでACNも待つ。この依存を明示し、「意味分類から独立」と「ASR処理から完全に独立」を混同しない。

## 4. ACNを実時間処理へ接続する方法

### 4.1 まず元の推論を再現する

採用モデルの重みとscalerを読み込み、保存済みの単語特徴・境界・期待出力に対して一致を確認する。元モデルを再現してからオンライン向け変更を行う。

推論資産は次の情報をまとめる。

- `weights.npz`: 各cueのMLP、attention、aggregationの重み。ensembleなら各memberを保存。
- `manifest.json`: 構成、特徴の順序・単位、前処理、scaler、教師統計量、モデル版、checksum、学習／選定条件。
- 特徴抽出コードと依存バージョン。
- 数例の期待入出力。可能なら短い音声と単語境界も添える。

### 4.2 2種類の運用を比較する

| 運用 | 方法 | 判断 |
|---|---|---|
| 双方向・遅延あり | 前後単語が揃った時点で原モデルを実行し、最新の有効スコアを短時間保持 | 元モデルの基準。単語境界待ちと次単語待ちを含めて遅延評価 |
| 過去のみの候補 | 次単語を使わない条件を評価し、必要ならその条件で再学習／校正 | 低遅延化が必要な場合。原モデルと同等とは扱わない |

未来単語がまだ未観測である状態と、発話末尾で存在しない状態は別にする。既存maskの0を未観測にも適用する場合は近似として明示し、精度と校正の変化を検証する。既存重みのattentionを勝手に再正規化しない。

語の途中の長さや不完全なMFCC統計を完成した単語の特徴として渡さない。ASRの単語境界revisionには音声区間IDで追従し、訂正後の語および隣接語を必要に応じて再計算する。既に実行したアバター動作は遡って実行し直さない。

`ProminenceState`には、対象音声の開始／終了sample、観測可能になった時刻、raw score、校正済みscore、境界の信頼度、モデル版、先読み方式を持たせる。MLP由来のconfidenceは現状ないため、固定の高confidenceを作らない。境界信頼度はモデルの予測不確実性と区別する。

Action Selectorは新しいACN結果を待って停止せず、有効期限内の結果を使う。未取得・古すぎる・境界不確実の場合は控えめな既定強度を用い、その理由をログに残す。保持時間・平滑化は評価に基づくconfigとする。

## 5. イベントと実行管理

- `asyncio`でイベントを配信し、音声取得、VAP、ASR、ACN、Jev、出力、記録のworkerを分ける。
- 重い同期推論をイベントループ内で直接実行しない。CPU/GPU競合は同時実行測定から制御する。
- `RuntimeState`への書き込みは一つのreducerに集約する。
- bounded queueを使い、音声欠落・推論遅れを記録する。dashboard更新はまとめられるが、ASR訂正や動作制御と同じ破棄方針にはしない。
- 共通の音声sample index、monotonic clock、session ID、event IDを使う。
- ASR発話ID・revision、semantic request ID、word span IDを結び付ける。
- Jevの並列要求数を制限し、debounce、重複排除、timeout、古い結果の破棄を実装する。再試行も要求の有効期限内に限る。

主な追加イベントは`WordBoundaryUpdateEvent`、`ProminenceUpdateEvent`、`ActionStartedEvent`、`ActionCompletedEvent`、タイマーイベント。無音とphrase boundaryもSemantic Triggerへ入力する。

## 6. beliefとpolicy

- Listener FunctionはNO_BC / CONTINUER / UNDERSTANDING / EMPATHICのみ。
- 確率の有限性・非負性・総和を検証し、log空間で観測更新する。
- 遷移行列の基準時間を定め、経過時間に応じてpredictionする。イベント数に比例して遷移を重ねない。
- q=0では観測更新なし、predictionのみとする。
- Jevの校正済み分布はV1では近似観測尤度として扱い、その仮定を記録する。必要に応じて事前分布補正や別の観測モデルを比較する。
- 同一入力の再観測を排除する。異なる発話が同じ分布を返す場合まで一律に重複としない。
- WHEN×beliefは初期段階では意思決定用スコア。条件付けと校正を検証するまで厳密なjoint probabilityと呼ばない。
- 期待効用policyでNONEを含む候補を評価し、prominenceは強度と許可された動作variantの選択にだけ使う。
- 動作ID、実行状態、cooldownを導入する。同じnodを後で再実行できるようにし、`action != last_action`だけで制御しない。
- PREPAREと実発火を分け、実行直前に予測の有効性を再確認する。horizon内の確率だけから厳密なonset時刻を捏造しない。
- CLAPはenumと通信には残すが、4分類だけでは祝福と悲しい話への共感を区別できないため、自動選択は初期設定で無効とする案。

## 7. ディレクトリ

`C:\Users\kosuk\nod` に以下を配置する。

```text
pyproject.toml
README.md
configs/{default,mock,live}.yaml
src/nod/
  core/{events,clock,bus,types}.py
  audio/{stream,vad,features}.py
  timing/{vap,maai_adapter,opportunity}.py
  asr/{streaming_asr,stability,word_boundaries,semantic_trigger}.py
  semantic/{backend,jev,schema,calibration,context}.py
  prominence/{acn,features,normalization,calibration,realtime}.py
  belief/{state,transition,filter}.py
  policy/{actions,decision,selector,embodiment,scheduler}.py
  output/{avatar,websocket}.py
  telemetry/{event_logger,latency,replay}.py
  dashboard/server.py
  config.py
  main.py
tests/{unit,integration,fixtures}/
models/acn/
docs/implementation-plan.md
work/acn_reference/       # 取得済み参照資料。原本として保持
```

学習スクリプトをそのままimportして本体を起動しない。推論に必要なNumPy処理を分離し、scikit-learn等の学習依存は必要性を確認して限定する。

## 8. 実装順序と完了条件

| 段階 | 作業 | 完了条件 |
|---|---|---|
| 0: ACN資産の固定 | 採用構成、重み、scaler、抽出コード、期待出力を確認 | 既存推論の再現に必要な資産が揃い、未解決点が明確 |
| 1: 再生可能な基盤 | 共通型、イベント、設定、mock、JSONL、仮想時計、基本belief/policy | 合成入力から動作判断を記録し、同じaction列を再生可能 |
| 2: ローカル音声処理 | 一度の音声取得からVAP、ASR、ACNへ配信。ACNの境界管理を追加 | 録音で各モデルの対象区間が揃い、同時実行の遅延が判明 |
| 3: Jevと観測統合 | 4分類、質問定義、日本語評価、reliability、revision、期限管理 | timeoutや順序逆転でも音声系が継続し、古い結果が混入しない |
| 4: 実時間ACNの選定 | 双方向遅延版と過去のみの候補を評価。score校正とTTLを設定 | 実際の動作時刻に利用できる強度と、その精度・遅延が確認できる |
| 5: アバター・画面 | localhost出力、動作ID、準備／実行／完了、dashboard | 同じnodの適切な再発火と実onset計測ができる |
| 6: 統合評価 | 30分連続動作、障害注入、録音比較、日本語での確認 | 完成条件15項目を実モデルで検証し、測定結果を記録 |

段階0の資産待ちでも段階1は進められる。Jevのアクセスは取得済みで、環境変数設定と実接続確認を段階3の作業に含める。全システムを完成扱いにするには実資産と実モデルでの確認が必要。

以前の15〜25人日は仮見積もりで、ACN重みの再生成、因果化に伴う再学習、日本語データ収集は含めていない。段階0とローカル同時実行測定の後で更新する。

## 9. 記録と評価

### 記録

- 入力イベントと中間状態、採用／破棄理由、raw／calibrated score、policyの判断理由。
- モデル・設定・質問定義・特徴抽出器の版、ランダムseed、校正データの版。
- audio capture、VAP、ASR partial、単語境界確定、次単語待ち、ACN特徴抽出、ACN本体、Jev、belief、command、実animation onsetの各遅延。
- stage別p50/p95/p99と、音声区間終了から情報が利用可能になるまでの経過時間。
- 元音声の保存を有効にしたセッションではsample indexでログと対応させる。特徴量・観測だけのログから音声モデル再実行ができるとはしない。

### 主要テスト

1. ACNの元推論との数値一致とexport/loadの一致。
2. 未来音声・未来単語を利用していないことの検証（過去のみの候補）。
3. ASR境界訂正で対象語と隣接語が再計算され、過去の動作を再発火しない。
4. score校正の範囲、有効期限、欠損時の強度fallback。
5. beliefの有限性・非負性・総和1、q=0の挙動。
6. Jevのtimeout／429／遅延／応答順序逆転時もVAP・ACNが継続。
7. prominenceを変えてもListener Functionが変わらない。
8. 重複nodの抑止と、cooldown後の同じnodの再実行。
9. 同じ観測イベント・設定・仮想時計でのaction再現。
10. 30分連続動作でキュー・メモリが増え続けない。

評価データは校正／選定用と最終評価用を分ける。参照v8/v9はEmphases側の相関でseed上位ensembleを選ぶ処理があるため、実運用モデルの新規選定では独立の検証データを使い、最終評価に選定結果を混入させない。

## 10. 追加で必要な資産

優先順:

1. 採用ACNの学習済み重み・構成・単体／ensemble情報。
2. 対応するscalerと、必要に応じて教師平均／標準偏差。
3. 音声から単語単位MFCCを作った抽出スクリプト・設定・メタデータ。
4. 元推論の期待出力がある小さなfixture。
5. 重みが未保存の場合に限り、学習特徴・ラベル・splitとexport手順。
6. ローカルPCのGPU/VRAM/RAM、およびJev APIキーの環境変数設定。

SSH先の元プロジェクト全体を取得する必要はない。まず保存済みcheckpointと抽出コードの場所を確認し、必要なファイルのみ追加コピーする。APIキーやパスワードは計画書・ソース・ログに保存しない。

## 11. 外部資料

2026-09-18の調査に基づく。導入時にバージョン・利用可能モデルを再確認する。

- VAP研究ページ: https://erikekstedt.github.io/VAP/
- 原著公開実装: https://github.com/ErikEkstedt/VoiceActivityProjection
- MaAI: https://github.com/MaAI-Kyoto/MaAI
- MaAI BC: https://github.com/MaAI-Kyoto/MaAI/blob/main/readme/vap_bc.md
- Jev API: https://docs.typesafe.ai/api
- Jevモデル・言語対応: https://docs.typesafe.ai/models
- Jev SDK: https://docs.typesafe.ai/introduction/quickstart
- Jev confidence: https://docs.typesafe.ai/confidence
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- SimulStreaming: https://github.com/ufal/SimulStreaming

## 12. 詳細設計への具体化

`C:\Users\kosuk\nod\docs\design\detailed-design.md` を追加し、以下を具体化した。

- Jevの4分類質問文と要求JSON、単一inflight・最新pending・deadline・ASR訂正時の採否。
- 同一発話の重複証拠を加算しない観測置換と、時間分割に依存しない4状態の連続時間遷移。
- WHEN×WHATの意思決定質量、期待効用表、強度経路の独立、opportunity単位の一回限りの実行。
- ACNのraw score校正、境界と先読み、対象音声時刻に基づく期限、欠損時の明示的な既定強度。
- アバターのcommand ID・TTL・ACK・開始・完了・再接続時の状態確認。
- 7種の境界データschema、暫定設定、22件の受入シナリオ。

設計JSON・schema・遷移式・utility例の整合確認を実施。モデル性能・ネットワーク遅延・アバター動作は未検証。詳細設計に記した閾値等は初期案であり、測定後に校正する。
