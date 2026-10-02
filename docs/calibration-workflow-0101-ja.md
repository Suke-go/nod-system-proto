# 意味観測の校正と、残るモデル仮定（0.10.1）

今回実装したのは、人の評価から意味観測の係数を推定し、独立したデータで確認してから実行設定へ組み込む経路である。人による正解ラベルはまだない。配布時の起動設定・係数・反応方策は 0.10.0 と同じであり、校正済み・反応品質改善済みとは主張しない。

## 修正の理由

従来の listener-fit は、知覚・理解・態度を統合した6状態の正解ラベルを必要とした。しかし、音声のないログだけで「正しく知覚した」を判定することはできない。今回は、Jev に渡ったその時点の文章と文脈をそのまま示し、解釈可能性 U、明示された個人的評価 E、局所的完結 C、実質的返答要求 D を個別に評価する。分類器の出力やロボットの状態を評価者には見せない。判断不能は欠測であり、否定ラベルへ置換しない。

この4項目は既存の質問定義に対応する操作的な評価基準であり、心理学で確立した4因子尺度ではない。人の内心・実際の理解・共感欲求を直接測定したとも解釈しない。知覚と解釈を区別する構造、増分処理で観測を取り消す考え方の位置づけは既存の model-audit-010-ja.md を参照する。分類器の質問文は今回変更しない。

## 数理と適用範囲

意味の choice 出力 q と、人が付けたカテゴリ y に対し、現在の観測モデルを保ったまま

`q | y ~ Dirichlet(1 + κ e_y)`

の κ を U、E ごとに最尤推定する。Kカテゴリの対数密度は `Σ(j=1..K−1) log(κ+j) + κ log(q_y)`。κ の探索範囲は既存の設定仕様と同じ [0.05, 20]。各成分に 1e−6 の下限を置いて再正規化する。この密度の対称形・条件付き独立性自体はモデル仮定として残る。片方のラベルが判断不能でも、他の項目の評価は利用できる。

C、D の noul 出力は、それだけで校正された確率とはいえない。操作的な二値ラベルに対して `p' = sigmoid(logit(p) / T)` とする温度をNLL最小化で推定する。数値端点には同じ下限処理を使う。分類の順位は変えず、スコアの確信度を調整する。温度スケーリングの方法的根拠は [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html)。同論文はこのJevの出力や心理的構成概念の妥当性を保証するものではない。

U/E の κ はベイズフィルタの観測密度へ、C/D の温度は意味観測の受理時に一度だけ適用する。元の API 応答はイベントログに残り、変換後の値は semantic_trace に残る。設定にモデル名・質問文ハッシュ・校正成果物ハッシュを記録し、モデルや質問の不一致を拒否する。通常起動では校正成果物を自動的に読み込まない。

評価出力は各項目の accuracy、NLL、Brier score を全体・言語別・セッション別に返す。U/E の transformed は一様な参照クラス事前分布を置いた `normalize(q^κ)` であり、ロボットの6状態事後分布ではない。また、密度最尤推定と分類NLL最小化は異なる目的なので、U/Eの分類NLLが改善する保証はない。

## 今すぐ確認できる実験ログ

`experiments/review-20260920/review.html` をブラウザで開く。20260919-201238-298917-live.jsonl の受理済み観測から、意味単位ごとの最後の観測57件を選んでいる。全480更新を独立標本とは扱わない。この選び方は確認の負担を抑えるためであり、途中の未完発話すべての校正を代表しない。ストリーミング全体への一般化は別途検証が必要である。

評価者IDを入力し、判定できる項目を選んで「評価を確定して次へ」を押す。最後に「評価ファイルを保存」で annotations-reviewed.jsonl を保存する。途中保存と再読込もできる。予測は別の predictions.jsonl にあり、評価画面には埋め込まない。

このログは過去の調整に使っているので development 専用。独立テストとして報告せず、校正の fit にも使用しない。録音がないため、音声認識の正解率や表情を出すべき時刻は判定できない。

## 再現手順（PowerShell、C:\Users\kosuk\nod）

既存の評価画面から保存したファイルを指定して集計する。全コマンドは出力を上書きしない。

```powershell
.\.venv\Scripts\python.exe -m nod.experiments.calibration score .\experiments\review-20260920\annotations-reviewed.jsonl .\experiments\review-20260920\predictions.jsonl --out .\experiments\review-20260920\scores.json
```

新しいセッションは、収集前に development / calibration / test を分ける。校正・テスト間で話者、セッション、同じ発話内容を再利用しない。個々の確率更新をランダム分割しない。既存ログの split を書き換えると整合性検査で拒否する。ただし同じ元ログを別区分で再エクスポートする操作まで研究履歴として追跡する機能ではないため、収集計画を別途固定しておく。

音声認識も検証するため、次回は原音声を保存する。既存の録音コマンドで最大30秒の16kHzモノラルWAVを作り、そのWAVを実時間で再生してログを取れる。次のコマンドはユーザーが実行した時に録音／API利用を開始する。今回の更新作業では実行していない。

```powershell
.\.venv\Scripts\python.exe -m nod record .\experiments\person01-ja.wav --seconds 30
.\.venv\Scripts\python.exe -m nod live --language ja --wav .\experiments\person01-ja.wav --seconds 30 --log .\sessions\NEW-SESSION.jsonl --open-browser
```

日本語／英語、完結／未完、正負／中立の内容を事前の計画に沿って収集する。録音内容の人による書き起こしは意味ラベルとは別に作成する。`record` で保存した音声は削除せず、対応するログと一緒に管理する。

```powershell
.\.venv\Scripts\python.exe -m nod.experiments.calibration export .\sessions\NEW-SESSION.jsonl --out .\experiments\cal-person01 --split calibration --participant person01
```

複数の評価済み annotations と対応する predictions をそれぞれJSONLとして結合して fit に渡す。同じIDの重複は拒否する。校正には2セッション以上、各カテゴリ8件以上の人の評価が必要。これは推定不能な入力を防ぐ最低限の実装上の条件であり、統計的検出力や十分な標本数の根拠ではない。自然な頻度を記録し、カテゴリを集めるための人工的な均等化を実利用分布と混同しない。

```powershell
.\.venv\Scripts\python.exe -m nod.experiments.calibration fit .\experiments\cal-annotations.jsonl .\experiments\cal-predictions.jsonl --out .\experiments\head-fit.json
.\.venv\Scripts\python.exe -m nod.experiments.calibration score .\experiments\test-annotations.jsonl .\experiments\test-predictions.jsonl --artifact .\experiments\head-fit.json --out .\experiments\head-heldout.json
.\.venv\Scripts\python.exe -m nod.experiments.calibration config .\experiments\head-fit.json --out .\configs\listener-head-fitted.json
.\.venv\Scripts\python.exe -m nod --config .\configs\listener-head-fitted.json live --language ja --open-browser
```

`config` は推定済み設定を生成するだけで、held-out 合格の証明や自動採用判定ではない。テストで悪化した場合は適用しない。英語についても独立に成績を確認する。モデル・質問文が混在する校正、校正とテストのセッション／入力重複、既知の話者重複は拒否する。話者IDがないデータでは話者独立性を検証済みと表示しない。同じ原稿を少し変えたもの等、ハッシュで検出できない重複は収集計画で防ぐ。

## 残る設計上の課題

| 対象 | 現状 | 推定・検証に必要なもの |
|---|---|---|
| 知覚 r | ASRの安定性・改訂量による代理指標 | 音声と人の書き起こし、認識誤りに対する校正 |
| U/E 観測密度、C/Dスコア | 今回、人の評価から推定する経路を実装。実データでは未推定 | 独立した評価者、ラベルの一致度、校正・テスト分割 |
| 初期分布・状態遷移・時間減衰 | 設計値 | 時系列データ、保持／取り消しの評価、感度分析 |
| 行動ごとの損失 | 誤反応と無反応の相対的費用を設計 | 反応候補を比較した人の評価、課題ごとの重み、独立した方策評価 |
| 応答タイミング・認識遅延 | ログから計測可能だが、適切さは未確定 | 元音声と提示時刻、反応可能区間の人の評価 |
| HRI効果 | 未検証 | 音響のみ・意味の直接選択・状態更新を伴う提案の比較、反応数を揃えた対照 |

同じ文章を複数人で独立に評価し、一致度と不一致の原因を確認してから合議ラベルを作る。同一IDに複数のラベルをそのまま混ぜて独立標本数を増やさない。現段階の score は一致度・信頼区間を算出しない。反応の良し悪しを classifier の自己評価で正解化せず、知覚の正解をASRの安定度から作らず、反応数だけを最大化して損失係数を学習しない。

## 実装検証

合成データは温度推定・密度最尤推定の数式、重複拒否、未評価の除外、実行時の一度だけの変換を検証するためにのみ用いた。人の評価や実験成績として保存・報告しない。旧セッションの再生一致、全テスト、評価画面の表示も確認する。現在の実データ評価件数は0件である。
