# D1–D4実装メモ

## 実装した範囲

- asyncioイベントバスと単一reducer。推論用HTTPはreducerから独立。送信も別task。
- ASR partialのappendとrepairを区別。2更新・300msで安定prefix、finalは確定扱い。
- Jevの日本語Choice問い合わせ、確率・ラベルの整合検査、期限、最新待機入力への置換。
- 同一発話内の重なる観測はアンカー事前分布に対して置換。連続時間の遷移はtick間隔に依存しない。
- VAP相当の入力スコアからヒステリシス付き機会窓。1窓1動作、期限切れ後は低スコアを経由して再開。
- 効用による意味機能選択と、別段階の強度選択。CLAPとprepareは無効。
- 受付・開始・完了・cooldown・FAULT。送達不明時は状態照会し、executeを自動再送しない。
- 実WebSocket通信、コントローラ能力確認、時刻差推定、TTL、重複動作防止。
- JSONL全入力記録と、導出結果の一致を検査するネットワーク不要のreplay。

## 時刻と入力

D1–D4の入力イベントは `Event(kind, at_ms, data)`。`at_ms` はローカルセッション開始からの整数msです。
`source_ms` は同じ時間軸へ変換済みの入力時刻で、未来・NaN・後退するASR時刻を拒否します。
同じmsの入力はJSONLの `ingest_seq` で順序を固定します。
これは音声サンプル時刻を代替する完成版契約ではありません。D5の音声アダプタで、16kHzサンプル番号と
monotonic時刻の対応、遅延・欠落・再接続を管理した上でこのreducerへ渡す必要があります。
設計中のASR/VAP/ACN境界JSON Schemaの完全実装はD5–D6に残っています。

実装済みのイベントは `asr`, `opportunity`, `prominence`, `semantic_result`, `semantic_error`,
`controller_ready`, `controller_disconnected`, `feedback`, `input_health`, `tick`。
`prominence.strength` は生のACN出力ではなく校正済みの[0,1]です。
入力アダプタの異常は `input_health.healthy=false` で出力停止できます。
キュー溢れはラッチし、判断ループを停止します。イベントを黙って間引きません。

## コントローラ通信

接続方向は、nodがloopbackサーバ、アバター側がクライアントです。
`controller_hello`, `action_command`, `controller_event` は同梱の設計JSON Schemaで検証します。
helloではSMALL_NOD/STRONG_NOD、60秒以上のdedup、status queryを要求します。
EMPATHIC_EXPRESSIONは対応宣言とneutral_acknowledgement指定がある時だけ選択します。
接続は1台まで。再接続時に古いcommandキューを再送しません。

時計同期用の制御メッセージは次の順です:

1. server → `{kind: clock_ping, t0}`
2. controller → `{kind: clock_pong, t0, t1, t2}`
3. serverでt3を取得し、controller−serverの時刻差 `((t1−t0)+(t2−t3))/2` と不確実性を推定。
4. server → `{kind: session_ready, session_id, connection_epoch, offset_ns, uncertainty_ns}`

単位はmonotonic ns。不確実性が50msを超える接続は拒否します。
commandの作成時刻にoffsetを加え、配送時間と不確実性を含む最悪側のageでTTLを検査します。
動作を実行した履歴は(session_id, action_id)ごとに60秒保持し、重複受信では状態だけ返します。
同じ種類の動作でも新しいaction_idなら、前の完了とcooldown後に実行できます。

状態照会は `{kind: status_query, session_id, connection_epoch, action_id, command_id}`。
既知なら通常のcontroller_eventで状態を返し、不明ならunknown。
FAULTは実際の完了/キャンセルを確認した時にcooldownへ移ります。不明のまま再開しません。
再接続でcontroller_readyを受けると、FAULT中の動作を再照会します。
シミュレータを終了・再起動して履歴が失われた場合、unknownのままなのでセッションを新規起動します。

タイマー式シミュレータは実アニメーションではありません。切断時はそのシミュレータ内のtimerをキャンセルし、
同一インスタンスが再接続した場合はキャンセル状態を返せます。実アバターには独自の中断・完了確認が必要です。

## ログと観測可能性

セッションheaderには設定とbackend、各行にはingest_seq、イベントと全導出レコードを保存します。
ASR本文・raw Jev確率・破棄理由・効用・動作・prominence_degradedを追跡できます。
ログを全件比較するreplayは、入力モデルの再推論ではありません。録音を行わないため音声からの再評価もできません。
出力ファイルの生成は排他的です。同じパスを再指定するとエラーにします。

## 検証範囲と未確認事項

`pytest` は確率異常、時刻遷移、観測二重計上、ASR修正、応答破棄、バックオフ、最新要求、
機会窓、動作通知順序、TTL、dedup、実WebSocket、デモ、replayを検証します。
HTTPはMockTransportで正常/認証失敗/負荷制限/サーバ異常/遅延/不正応答を再現します。
APIキー未設定のため実Jevの疎通・費用・応答時間・モデル提供状況は未確認です。

30分負荷試験、実マイク遅延、VAP日本語性能、ASRによる修正頻度、Jev意味分類の校正、
ACNの特徴量互換性、実アバター表情の適切さは未検証です。
設計で示した性能値や各閾値は暫定で、今回のテスト成功は実会話品質の保証を意味しません。
実行時のbelief calibrationはidentity、機会スコアとbeliefの積は校正済み結合確率として扱いません。
