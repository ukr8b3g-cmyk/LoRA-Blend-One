[English README](README.md)

# LoRA Blend One — 日本語ガイド

複数の互換LoRAをブレンドし、ComfyUIで再利用できる1つの `.safetensors` に保存するカスタムノードです。

[English README](README.md) · [検証状況](VALIDATION.md) · [変更履歴](CHANGELOG.md)

## インストール

ComfyUIの `custom_nodes` 内で実行します。

```sh
git clone https://github.com/ukr8b3g-cmyk/LoRA-Blend-One.git ComfyUI-LoRA-Blend-One
```

ZIPの場合は `ComfyUI/custom_nodes/ComfyUI-LoRA-Blend-One/__init__.py` となるよう展開してください。ComfyUIを再起動し、ブラウザも再読み込みします。ComfyUI付属のtorch・safetensorsを使い、追加依存はありません。

## 基本操作

[LoRA Blend Oneテンプレート](example_workflows/LoRA_Blend_One.json)をComfyUIで開くか、ComfyUI再起動後に **ワークフロー → テンプレートを参照** から本パッケージを選びます。Sources → Blend One → Preview Anyを接続済みです。LoRAは自分のファイルを選択してください。出力ランク16、シード0・固定、詳細レポートオフで開始します。

1. `model/merging/LoRA Blend One` から **LoRA Blend Sources** と **LoRA Blend One** を追加します。
2. `sources` 同士を接続します。
3. 同じベースモデル向けのLoRAを選び、比率を指定します。
4. まずは合成方式 **Auto**、比率モード **normalize**、圧縮 **auto** で使用します。
5. 出力ランクと保存名を指定し、実行します。

既定では `ComfyUI/output/loras/LoRA_Blend_One_00001_.safetensors` などへ保存されます。生成に使う際は `ComfyUI/models/loras/` または設定済みのLoRAフォルダへコピーし、ファイル一覧を更新して通常のLoRAローダーで選んでください。

SourcesのAdvancedを開くとLoRA 3～8を指定できます。Seed（初期値0）や使用頻度の低い数値・診断項目もAdvancedにあります。2ノード構成の詳細JSONレポートは初期値オフです。旧ワークフロー向けに **LoRA Blend One (Legacy)** も維持しています。

## UIの言語

基本表示は英語です。ComfyUIの **設定** で **言語（Language）**（`Comfy.Locale`）を検索し、**日本語** にすると、項目名・ツールチップ・詳細設定ボタン・モデル判定表示が日本語になります。英語に戻すと英語表示になります。AutoやKrea2 Gate-Awareなどの方式名は共通です。

## 対応範囲とランク

| モデル | 処理 |
| --- | --- |
| MOSS-Image Base／SFT | 自動識別・キー名とQKV形式の統一 |
| Krea2 | 専用Gate-Aware方式と通常方式 |
| FLUX／Qwen Image／Wan／SD系など | 標準LoRAの共通処理。モデル別のキー変換はありません |

同じベースモデル向けで、適用先の層の名前と入出力サイズが互換な標準LoRAを使用します。**入力ランクは同じでなくて構いません。** ランク128＋32を合成して、出力ランク16で保存することもできます。

DoRA・LoHa・LoKr・LoCon-mid・モデル全体の差分などは対応範囲外です。MOSSでは未対応テンソル、欠けた因子ペア、曖昧な別名の重複も検出して停止します。各モデルの実画像生成・画質比較を網羅検証したものではありません。

## MOSS-Imageの使い方と表示

MOSS用LoRAを選び、合成方式を **Auto／Weighted** などに設定してください。**MOSS-Imageは合成方式の一覧に追加される項目ではなく、自動識別されます。**

正常に実行が完了すると、ノード下部に緑色で **「前回のモデル判定：MOSS-Image」** と表示します。未実行時は「モデル判定：実行後に表示」です。この表示は前回成功した実行結果を示します。

QKV結合・分割形式とキー名の違いを合成前に揃え、alpha／rankの倍率を維持します。出力はComfyUI用の分割Q／K／V形式です。高速化パッケージの専用サンプラーや時間埋め込みファイルまでは、LoRAへ統合しません。

## 比率モード

| 入力 | normalize | direct |
| --- | --- | --- |
| 30／70 | 0.3／0.7 | 0.3／0.7 |
| 80／40／20 | 約0.571／0.286／0.143 | 0.8／0.4／0.2 |

既存の複数LoRAローダーで使っている強度を引き継ぐ場合は **direct** を選びます。directでは負の係数も使用できます。Krea2 Gate-Awareは比率モードにかかわらず直接係数を使います。係数はモデル差分の強さであり、見た目の混合割合を保証する数値ではありません。

## 出力ランク・精度・サイズ

数値で指定した出力ランクは、保存する各層・各投影の上限になります。ランクを下げると通常はサイズが減り、表現の一部が失われる可能性があります。

- **auto**：出力ランクAutoならエネルギー基準、数値指定なら固定ランク圧縮。
- **energy**：エネルギー基準で選び、数値ランクを上限として使用。
- **fixed_rank**：数値ランクの指定が必要。
- **max_input_rank**：入力最大ランクを使い、数値ランク指定があればさらに制限。
- **none**：Weighted専用。Autoまたは結合ランク以上の指定なら厳密結合、低いランクを指定するとSVD圧縮。

Energy Targetは数値上の差分エネルギーで、画質の保持率ではありません。TIES／DARE／DELLAの自動エネルギー選択にはAdvancedの `auto_rank_cap` も適用されます。

**保存dtypeがAutoの場合、全入力で共通の精度を維持し、異なる精度が混在するとFP32で保存します。** FP32とBF16を混ぜてサイズが大きくなる場合は、保存dtypeを明示してください。計算精度はFP32です。

## 合成方式

AutoはWeightedです。Weightedは差分の重み付き和、TIESは大きさによる選別と符号合意、DAREはランダムな除去と再スケール、DELLAは大きさを考慮した確率的な保持と再スケールを行います。TIES／DARE／DELLAは1層ずつ密行列の差分を扱うため、Weightedより多くのメモリを必要とする場合があります。

Krea2 Gate-Awareは2本のKrea2用LoRAを対象とする専用方式です。ゲート競合の処理を行い、専用設定はAdvancedに表示されます。MOSSでは使用できません。

## 確認済みの範囲

Python 29件とJavaScript UI 3件の自動テストが成功しています。MOSS形式の互換性、通常方式／Krea2、ランク制限、ノード統合、言語切替、ワークフローの保存・復元を含みます。

手元のMOSS用LoRA（ランク128／32／32）を強度0.8／0.4／0.2で全252層ブレンドし、保存・再読込とComfyUIローダーでの全層解析、Q／O層の重み適用をCPUで確認しました。FP32保存時はランク16で約149MB、ランク32で約297MBでした。

### MOSS-Image SFT：実ブレンド・保存・読み込み・GPU画像生成

**2026-10-22**、起動中のComfyUIと既に開いていたChromeタブで、一連の操作が成功しました。

| 段階 | 確認結果 |
| --- | --- |
| 入力 | `LUC-LORA-MOSS-Epoch 12.safetensors` ＋ `LUC-LORA-MOSS-Epoch 20.safetensors`、50:50、`normalize` |
| ブレンド | Auto → Weighted、CPU計算、出力ランク16、BF16保存、シード0・固定 |
| 保存ファイル | **72,633,952 bytes（約72.6 MB）**、246レイヤー、全層ランク16、ブレンド警告0件 |
| 再読み込み | 全**738テンソル**が有限値で、ComfyUIパーサーがすべて認識。未使用テンソル・形状不一致0件 |
| 生成時の適用 | 保存したブレンドを **Power Lora Loader (rgthree)** で強度**0.3**に設定し、`moss_image_sft_bf16.safetensors` に適用 |
| 生成画像 | **896×1184**、8 steps、CFG 1、`er_sde`／`simple`、固定シード525173493099460。WEBP保存・目視確認まで成功 |

検証環境（RTX 5060 Ti、VRAM 16 GB）でのジョブ実行時間は、ブレンド3.774秒、画像生成19.079秒でした。この1回の実測値であり、一般的な所要時間の目安ではありません。実行履歴でも保存済みブレンドが強度0.3で有効だったことを確認し、出力画像を正しい寸法で読み込めました。黒画像やノイズだけの出力にはなっていません。

生成時は元ワークフローの他のLoRAも併用しています。この条件での読み込み・生成の確認であり、ブレンド単独の画質A/B比較や、すべてのMOSS用LoRA・MOSS-Image Base・他モデルの動作保証ではありません。

Chrome実画面ではテンプレート読み込み、Advancedの開閉、英語→日本語→英語の切替、保存後に閉じて開き直す操作も確認しました。Custom Rank 48、Energy Target 0.975、保存dtype BF16を維持し、API出力44項目も保存値・宣言された入力型と一致しました。

詳細な形式・検証方法は [English README](README.md) と [VALIDATION.md](VALIDATION.md) を参照してください。

## ライセンス

[MIT](LICENSE)。モデルやLoRAファイルには、それぞれのライセンスが適用されます。
