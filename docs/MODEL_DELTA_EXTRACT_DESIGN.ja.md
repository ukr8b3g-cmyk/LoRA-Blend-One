# LoRA Blend One：モデル差分抽出の実装設計

決定日：2026-10-23。この文書は設計時の仕様であり、将来拡張も含みます。現在の実装は単体・非量子化MOSS-Image Safetensorsのみです。分割シャード・他モデルは未実装です。実ファイルCPU抽出・保存・ネイティブ読み込みは確認済みですが、実画面・GPU生成・画質は未検証です。採用ランク・99％保持の実測値とBF16丸めの制約は [英語README](../README.md#extract-a-model-difference-moss-image)、[日本語README](../README.ja.md)、[VALIDATION](../VALIDATION.md) を正本としてください。

## 1. パッケージとノード

別リポジトリを作らず、LoRA Blend One内に抽出専用の経路を追加する。カテゴリは `model/merging/LoRA Blend One`。

| node ID | 英語表示 | 日本語表示 | 役割 |
|---|---|---|---|
| `LoRABlendModelInspect` | LoRA Blend Model Inspect | LoRA Blend モデル差分確認 | ファイル構造・対応範囲・容量見積もり。任意の事前確認 |
| `LoRABlendModelExtract` | LoRA Blend Model Extract | LoRA Blend モデル差分抽出 | 再検査・差分・分解・保存・再検証 |

両ノードは元／加工済みファイルを直接選ぶ。Inspectは独立した補助ノードで、Extractへの配線は不要。Inspectの実行有無にかかわらずExtractが同じ検査を行う。初期版ではMODEL／CLIP／VAE入力を設けない。

出力はSTRING情報（Inspectは検査結果、Extractは保存先・品質集計・警告）。ファイルはサーバー側へ保存する。出力ノードとして動き、抽出・画像生成を自動的にキューへ追加しない。

## 2. ファイルの選択・入力範囲

- 初期版は単一Safetensors、BF16／FP16／FP32の拡散モデル本体。チェックポイント全体の場合も、系列別に定義した拡散モデル領域だけを処理する。
- ComfyUIの `diffusion_models`／`checkpoints` 検索パスを使い、選択値は「検索カテゴリ＋相対ファイル名」で識別する。同名ファイルを区別し、読み込み時も既存のパス解決APIで検査する。
- Pickle系、GGUF、FP8／INT8等の量子化モデルは拒否する。FP16ファイルに偽装した量子化スケール／整数重みも検出する。
- テキストエンコーダー、VAE、EMAの別コピー、学習状態は対象外として一覧に記録する。非浮動小数バッファは既知の同一／実行時未使用バッファだけを許容し、その他は停止する。
- index付き分割Safetensorsは第2段階。初期版で単一ファイルと誤認して扱わない。対応時は重複キー、索引と実シャードの不一致、親フォルダー外への参照を拒否する。
- 外部コードの実行、`trust_remote_code`、自動ダウンロードは不要。
- 選択ファイルのサイズ・mtime・ヘッダー識別値で検査結果を識別し、処理前後に変更を検出する。Inspectの古い結果を抽出の許可証にしない。

## 3. 対応モデルの有効化

共通処理はモデル系列に依存させず、キー対応・構造識別・既知バッファを系列別に定義する。手動Architecture指定でも構造検査を省略できない。未知系列を形状一致だけで正式対応と扱わない。

| 系列 | 最初の実装方針 | 正式対応の条件 |
|---|---|---|
| MOSS-Image | 最初の実用ゲート。既知prefix、QKV、Linearの対応を実装 | 実ファイルの抽出・保存・ネイティブ適用確認 |
| SD1／SDXL | 別構造ID。ComfyUIネイティブUNetキーとConv2d | 各系列で同じ実ファイル確認 |
| FLUX1／FLUX2 | 別構造ID。ネイティブレイアウト同士から開始 | 各系列・版ごとに確認 |
| Qwen-Image | ネイティブレイアウト一致から開始。版固有の結合／分割は別対応 | 各版で確認 |
| Krea2 | ネイティブキーに加え独立scale／modulationパラメーターを分類 | 通常／混合／フル差分の適用確認 |

MOSS以外も拡張対象として設計するが、実ファイル確認前は対応モデル一覧へ掲載・UIで正式有効化しない。Krea2 Gate-Awareの合成規則は抽出へ持ち込まない。

構造指紋は系列ID、ブロック番号集合、必須キー、入出力次元、QKV配置、利用可能なモデル設定から作る。元モデルと学習・マージ版を区別する出所は名前だけでは証明できない。形状互換と、意味的に元モデルが正しいことは区別して表示する。

### 手元で確認できたMOSSの構造上の注意

ヘッダー確認のみ実施。SFTの一例は379テンソル（2D weight 185、1D weight 177、bias 16、`rope.bands` 1）。派生2例はprefixを除く378キーと形状が一致し、`rope.bands`がない。ComfyUIのMOSS実装はこのバッファを除去してRoPEを再構成するため、この1キーは既知の `runtime_unused_buffer` として記録する。他の欠落キーへ例外を広げない。

公式BaseとMOSSkr2と確認できる組は今回未確認。この組での抽出成功を前提にしない。

## 4. キー対応と不一致

1. ヘッダーからモデル本体の全キー・形状・dtypeを取得。
2. 既知prefixのみ除去してネイティブキーへ正規化。同名への重複変換は停止。
3. 既知のQKV配置は論理モジュールと行スライスへ変換する。
4. すべての対象パラメーターを対応付け、欠落・追加・形状不一致・未対応を列挙。
5. 値を読み込む段階でNaN／Infと実際の差分を検査。

MOSSの結合QKVはQ／K／Vへ分けて差分・分解し、通常LoRAでは分離ターゲットを出力する。text_embedder等の別QKV構造はjoint QKVと混同しない。全系列へ等分QKV変換を適用しない。

既定は厳密。未知の構造、必須キー欠落、変換の曖昧さ、形状不一致は保存前に停止。高度な部分抽出は明示選択時だけ許可し、除外キー・差分量と「モデル全体の差分ではない」表示を必須にする。未知系列への安全検査回避には使えない。

## 5. 抽出と保存形式

基本式は `D = FP32(modified) - FP32(base)`。差分倍率は1、符号は加工済み−元で固定。

| 抽出方式 | 出力 | 現行Blend Oneへ再入力 |
|---|---|---|
| Fixed Rank | Standard LoRA | 可：対応する通常因子のみ |
| Auto Rank | Standard LoRA、層別rank | 同上 |
| Full Delta | ComfyUI拡張差分 | 不可 |
| Fixed／Auto＋非行列差分の保存 | LoRAとComfyUI差分の混合 | 不可 |

### 通常LoRA

2D Linearと対応するConvを行列化する。Conv2dは `[out,in,kH,kW] → [out,in*kH*kW]`。rank上限は `min(out,in_flat)`。Conv因子はB=`[out,r,1,1]`、A=`[r,in,kH,kW]`へ戻す。

`D ≈ U_r S_r Vh_r` とし、B=`U_r sqrt(S_r)`、A=`sqrt(S_r) Vh_r`。保存キーはネイティブtargetに対する `.lora_B.weight`／`.lora_A.weight` と `.alpha=r`。強度1.0で係数が1になる。クランプや無説明の再倍率は加えない。

### 非行列差分の方針

Advancedに `Non-LoRA Changes = Stop / Store Diff / Exclude With Report` を設け、既定はStop。

- bias／norm／独立パラメーターに変更があれば、Stopでは通常LoRAのまま黙って落とさず停止する。
- Store Diffは同じファイルへ対応する`.diff`を追加し、混合形式と明示する。
- Exclude With Reportは部分抽出として保存。除外した変更のキー・エネルギーを集計する。
- Full Deltaでは低ランク化せず、モデル本体の対応可能な変更を`.diff`として保存する。未知の適用先は止める。

ComfyUIネイティブキーに対する規則：`foo.weight → foo.diff`、`foo.bias → foo.bias.diff`、`foo.scale → foo.scale.diff`。biasに`.diff_b`を使わず、既存ローダーの直接パラメーター対応を使う。スライスを要する場合はネイティブ適用試験で対象・offsetを確認する。同じ適用先にLoRA因子と`.diff`を重ねて出力しない。

現行Blendは`.diff`を拒否する。この仕様を維持する。拡張形式は標準ComfyUI LoRAローダーでの適用を目標とし、他ツールへの互換性は保証しない。

## 6. SVDと数値検証

- Fixedは指定rankを実際の行列サイズで制限して採用する。
- Autoは元差分全体の二乗ノルムを分母として、近似特異値の上位エネルギーが目標に達する最小rankを選ぶ。上限で未達なら達成値・層名を警告する。
- SVD MethodはAuto／Exact／Randomized。Autoは `min_dim<=512 かつ numel<=4,000,000` でExact、それ以外はRandomizedを初期基準とする。性能保証ではなく、測定後の閾値調整が可能な内部定数。
- Exactは `torch.linalg.svd(..., full_matrices=False)`。Randomizedは既存QR／小行列SVD方式を使い、oversample=8、power iterations=2、固定Seed=0を既定にする。
- 既存合成のmath処理は変更しない。抽出側で差分をmax-absで内部正規化してFP32二乗和のあふれを防ぎ、因子を逆スケーリングして元の差分倍率へ戻す。ノルムと誤差はFP64／安定スケールで集計する。
- Seedはキーごとに `stable_seed(seed, architecture, canonical_key)` から作り、実際のdevice指定の局所Generatorを使う。グローバルRNGを変えない。CPU／CUDA間のビット完全一致は保証しない。
- 入力値、FP32差分、生成因子、保存dtype変換後を有限性検査。異常時に保存dtypeを勝手に変更しない。
- 保存dtypeに丸めたB／Aを行ブロックごとに再構成し、`||D-B_saved A_saved||F / ||D||F` と最大絶対誤差を測定する。特異値による保持率と、丸め後の実測誤差を別欄にする。
- Full Deltaと混合形式の`.diff`にも、丸め後の `||D-D_saved||F / ||D||F` と最大絶対誤差をチャンク計算し、保持領域全体の誤差集計へ含める。元差分・残差がともに0なら誤差0とし、元差分0に対して残差だけが非ゼロなら異常として停止する。0除算によるNaN／Infをレポートへ出さない。
- ゼロ差分は記録して省略。全対象がゼロなら「差分なし」を返し、空のLoRAは作らない。微小差分除外の既定値は0。
- 部分抽出では、保持した領域の近似誤差と除外した変更の量を分ける。画像の一致率として表示しない。

## 7. 画面と既定値

Extractの通常表示：元モデル、加工済みモデル、抽出方式、Rank=16（Fixed）、Target Energy=0.99（Auto）、Storage Type=BF16、Filename Prefix=`loras/LoRA_Extract`。

方式に応じた欄だけを表示するが、非表示にした値は保持し、シリアライズ順を変えない。

Advanced：Architecture=Auto、Compute Device=Auto、Max Rank=256、SVD Method=Auto、Seed=0/fixed、Non-LoRA Changes=Stop、対象キー／ブロック=全モデル本体、Strict、微小差分閾値=0、Hash Inputs=ON、Report=ON、Fallback=auto_cpu。今回の入力・出力の有限性検査は無効化不可。

Inspectは同じ2ファイルとArchitectureを使う。既定のHeader検査はキー・形状・精度・推定容量までで、値の有限性や差分量を確認済みと表示しない。AdvancedのFull Scanで全値を走査して差分量・不正値・非行列変更も表示する。

英語を基本とし、ComfyUI Localeが日本語なら名前・説明・対処方法を日本語化。Advancedは実際に開閉するアコーディオン。既存ノードのID、ウィジェット順、既定値は変更しない。

## 8. RAM／VRAMと保存

入力は `safe_open` でモデル2個を丸ごとロードせず、原則1論理層ずつ取得する。ただし最大層の差分行列・SVD作業領域は必要であり「一定の小容量で必ず処理できる」とは説明しない。

Autoは最大層とSVD方式の推定からCPU／CUDAを選択する。空きVRAMに余裕がある場合だけCUDAを使い、生成モデルのアンロードを要求しない。CUDA OOMのCPU再試行は同じ層・rank・精度・Seedを維持して警告する。CPU側も不足する場合は停止する。

既存 `core/writer.py::save_verified` は全出力辞書を要求するため、そのまま呼び出さない。

1. 層ごとの結果を公式 `save_file` で作業フォルダーの小さいSafetensorsへ書き出す。
2. RAMにはキー・dtype・shape・バイト長・チェックサムだけを保持。
3. 最後にSafetensorsのヘッダーを作り、各データ領域をチャンクコピーして単一の `.partial` を組み立てる。
4. 公式 `safe_open` でキー・形状・dtype・offset整合・メタデータを確認。値の有限性とpayloadチェックサムもチャンクで検証。
5. 本体・レポートの両方が未使用の連番を選ぶ。Windowsは既存先があれば失敗するrename、POSIXは同一FSのlinkによるno-replaceで本体を確定する。衝突は次番号へ。安全なno-replaceが使えないFSでは停止する。
6. レポートも一時ファイルからno-replaceで確定。本体成功後のレポート失敗・同名競合は警告と本体保存先を返し、既存レポートを保持する。
7. `finally`で一時層ファイル・未完成ファイルを掃除する。中断例外がBaseExceptionである点を含めて試験する。

必要な作業ディスクは安全側に出力約2倍＋最大層ファイル＋ヘッダー／レポートとして見積もる。固定rankは層の `(out+in_flat)*rank*保存byte数`、AutoはMax Rankの上限、Full Deltaは全対象の実要素数から計算する。OSキャッシュと分解ライブラリーの作業量は別途余裕が必要。速度の秒数は実測まで提示しない。

## 9. 保存先・メタデータ・状態

初期保存先は `ComfyUI/output/loras/`。本体は連番付き`.safetensors`、レポートは既存パッケージと統一して `.safetensors.report.json`。利用時はLoRA検索対象フォルダーへ配置する。

メタデータは `lbo_extraction_version=1`、artifact type（standard_lora／hybrid_delta／full_delta）、モデル系列、入力の表示名・必要なSHA256、設定、対象範囲。絶対パス・OSユーザー名を入れない。MOSS通常LoRAには既存のMOSS識別メタデータを付ける。

モデル差分の記録を「元LoRAのブレンドレシピ」と偽装しない。`lbo_recipe_version`や架空のSourcesを付けず、既存のレシピ復元／ドロップ機能に誤認させない。

進捗段階は「ヘッダー／互換性 → ハッシュ・値検査 → 差分・SVD → 組立保存 → 再検証」。層名、処理数、経過時間、採用rank、警告数をノードに出す。中断は層、ハッシュ、コピー、再検証のチャンク境界で確認。SVDカーネル中は即時停止できないことを表示する。

エラーには2入力の表示名、対象キー、停止理由、対処方法を出す。保存エラーを特定のモデルが壊れていると誤表示しない。ファイルを出さないInspectと、保存成功・警告付き成功・差分なし・失敗を区別する。

## 10. 追加ファイルと再利用範囲

| ファイル | 所有する処理 |
|---|---|
| `extract_nodes.py` | 2ノードの入出力、folder_paths、ComfyUI進捗／中断との接続 |
| `extractor/model_files.py` | ヘッダーと層取得、検索ID、ファイル指紋 |
| `extractor/plan.py` | 系列別対応表、キー分類、互換性、容量見積もり |
| `extractor/engine.py` | FP32差分、SVD、保存後精度の数値検証 |
| `extractor/writer.py` | 層別作業ファイル、単一出力組立、検証、no-replace確定 |
| `web/extractor.js` | 抽出ノードだけの表示切替、Advanced、進捗・エラー |

既存ロケール、エラー表示、MOSSのキー規則、安定Seed／SVDの数学的方式を再利用する。登録箇所は新node IDを加えるだけにする。既存 `core/engine.py`／writer／Blendの入力形式へ抽出用フラグを追加しない。系列ごとの大量のクラス階層や別サーバー、新規依存ライブラリーは作らない。

## 11. 実装順と完了条件

1. reader／planとInspect：MOSSの実ヘッダーでprefix・QKV・既知バッファを確認。
2. 小型の既知差分でFixed／Auto／Full Deltaと厳密・部分抽出を実装。
3. ディスク作業ファイル・単一出力保存と障害注入試験。
4. ComfyUIネイティブローダーで全出力キーの消費先と形状を照合し、実際のweight adapterで元重みへの適用を数値確認。MOSSの既存 `tools/validate_moss_local.py` のmetaヘッダー検証方法を参考にする。
5. MOSS実ファイルで抽出→保存→再読込→層ごとの適用・再構成誤差を確認。Inspect／Extractの実画面と英日切替を確認。元ワークフローを保持し、別の検証ワークフローを用いる。
6. 既存の合成・MOSS・Krea2・レシピ・UIテストを通す。追加試験と既存試験は別集計。画像生成比較はユーザー指定の設定・承認時に別途行う。
7. 他系列は同じ完了条件を満たしてから有効化。index分割入力はその後の独立対応。

必須の新規試験：同一モデル／既知低ランク差分／符号反転／rank上限／Auto未達／Conv／norm・bias／QKV結合分離／既知バッファ／未知キー・重複alias／NaN・Inf／保存変換overflow／量子化拒否／入力変更／ディスク不足／中断／同名競合／レポートだけの失敗／メタデータの絶対パス不在。

今回は設計・実装調査とヘッダー確認だけ。実抽出、SVDベンチマーク、モデルのGPUロード、画像生成、製品コード変更、コミット・プッシュは実施していない。

## 実装根拠

- ComfyUI既存抽出：<https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_lora_extract.py>。形式の参考とし、GPUモデルロード・FP16固定・99%分位クランプ・未対応キーの読み飛ばしは流用しない。
- ComfyUIローダー：<https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/lora.py>。ローカル版とも照合済み。ネイティブパラメーターと`.diff`の実適用は実装後に検証する。
- PyTorch SVD：<https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html>。
- Safetensors形式：<https://github.com/huggingface/safetensors#format>。単一出力の組立はこの形式に従い、公式readerを検証器として使う。
- Python rename：<https://docs.python.org/3/library/os.html#os.rename>。WindowsとPOSIXで上書きの挙動が異なる点を扱う。
