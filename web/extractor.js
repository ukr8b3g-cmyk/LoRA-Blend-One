import { app } from '../../scripts/app.js';
import { trackNodeRuns } from './run_tracker.js';
import { text, watchLocale } from './i18n.js';

// English labels are stable; only help follows Comfy.Locale.
const extractorHelp = {
    "purpose": ["Output Purpose", "ComfyUI Loading preserves nonmatrix changes as Store Diff. Blend One Source requires standard LoRA; partial extraction needs explicit consent.", "ComfyUI Loadingは行列以外の変更もStore Diffで保存します。Blend One Sourceは通常LoRA形式のため、変更の除外には明示的な部分抽出の選択が必要です。"],
    "allow_partial": ["Allow Partial Extraction", "For Blend One Source only: explicitly allow excluding changed nonmatrix parameters. The output cannot reproduce all model changes.", "Blend One Source用です。行列以外の変更を除外する部分抽出を明示的に許可します。モデル全体の変更は再現できません。"],
    "extraction_plan": ["Extraction Plan", "Connect Inspect for advice. Settings remain editable and are used as displayed. Apply Recommendations copies a completed analysis once; connecting or re-running Inspect never overwrites your settings.", "Inspectの分析結果を参照します。接続中も設定欄を編集でき、その表示値で実行します。Apply Recommendationsで推奨値を一度だけコピーします。接続・再分析では手動値を上書きしません。"],
    "plan_action": ["Plan Action", "Connected plans start in Review Only: display your current settings and recommendations without saving. Extract and Save uses the editable settings on this node. Recommendations are copied only by the Apply Recommendations button. Unconnected manual extraction works as before.", "接続時はReview Onlyで現在の設定と推奨値を確認します。Extract and Saveはこのノードで編集した設定を使用します。推奨値の反映はApply Recommendationsボタンを押したときだけです。未接続でも手動抽出できます。"],
    "base_model": [
        "Base Model",
        "Original full model used as the starting point before fine-tuning or merging. Select model weights, not a LoRA. The extracted difference is Modified Model minus Base Model. Both files must have compatible MOSS-Image architecture and shapes; matching shapes alone does not prove this is the correct training base.",
        "学習・マージする前の元のフルモデルです。LoRAではなくモデル本体を選びます。「Modified Model − Base Model」の差分を抽出します。両方とも構造・形状が互換なMOSS-Imageが必要です。形状が合っていても、正しい学習元かどうかは別途確認してください。"
    ],
    "modified_model": [
        "Modified Model",
        "Full model after fine-tuning or merging, based on the selected Base Model. This is not a LoRA file. Swapping the two models reverses the difference. Identical weights produce no output. Only single, non-quantized native MOSS-Image Safetensors are currently supported.",
        "Base Modelを元に追加学習・マージなどで変更したフルモデルです。LoRAファイルではありません。元と変更後を逆にすると差分の符号も逆になります。同じ重みなら出力しません。現在は単体・非量子化のMOSS-ImageネイティブSafetensorsに対応します。"
    ],
    "mode": [
        "Extraction Mode",
        "Fixed Rank: use the specified integer rank for each matrix, capped by its dimensions. Auto Rank: choose a rank per matrix for Target Energy, limited by Maximum Rank. Full Delta: store the complete difference without SVD compression; large ComfyUI extension file, not a standard LoRA.",
        "Fixed Rank：各行列を指定ランクで抽出します（行列寸法が上限）。Auto Rank：保持率を目標に層ごとのランクを選び、Maximum Rankで制限します。Full Delta：SVD圧縮せず差分をそのまま保存します。大容量のComfyUI拡張形式で、通常LoRAではありません。"
    ],
    "rank": [
        "Rank",
        "Used only in Fixed Rank mode; default 32. Rank Preset offers 16, 32, 64, 128 and 256. Any integer from 1 to 4096 is valid, including 17 or 24; it is not restricted to multiples of 8 or powers of two. Common examples: 8, 16, 32, 64, 128, 256. Lower ranks usually save space but lose more difference information. No rank guarantees faithful recovery. Each layer is also limited by its matrix dimensions.",
        "Fixed Rankで使い、初期値は32です。Rank Presetから16・32・64・128・256を選べます。1〜4096の整数を指定でき、17や24も有効です。8の倍数や2の累乗に限定しません。一般的な例：8／16／32／64／128／256。低いほど通常は小容量ですが、失う差分も増えます。十分な精度を保証するランクはありません。各層の行列寸法も上限になります。"
    ],
    "energy": [
        "Target Energy",
        "Used only in Auto Rank mode. 0.99 means retaining 99% of squared weight-difference energy before storage conversion, roughly 10% relative Frobenius error; it is not 99% image similarity. Maximum Rank can prevent reaching the target. BF16/FP16 rounding can leave individual saved layers slightly below it.",
        "Auto Rankで使います。0.99は保存精度へ変換する前の重み差分の二乗エネルギー99％を目標とし、相対Frobenius誤差では約10％です。画像の一致率99％という意味ではありません。Maximum Rankが低いと未達になります。BF16／FP16の丸め後は個別の層がわずかに下回る場合もあります。"
    ],
    "dtype": [
        "Storage Type",
        "Precision of saved factors/differences; subtraction and factor calculations use FP32. BF16 has a wider value range than FP16. FP32 uses more disk space. Non-finite values or storage overflow stop the save; precision is never changed automatically.",
        "保存する因子・差分の精度です。差分・因子の計算はFP32で行います。BF16はFP16より広い値域を扱え、FP32は保存容量が増えます。不正値や保存時のあふれがあれば停止し、精度を勝手に変更しません。"
    ],
    "filename_prefix": [
        "Filename Prefix",
        "Output name under ComfyUI's configured output directory. Subfolders are allowed; a unique numbered suffix is added without overwriting existing files. The result is not automatically installed in the LoRA folder. Put it in a configured LoRA folder to select it in a loader.",
        "ComfyUIで設定された出力フォルダー内の保存名です。サブフォルダーを指定でき、既存ファイルを上書きしない連番が付きます。LoRAフォルダーへ自動登録はしません。Loaderで使う場合は、設定済みLoRAフォルダーに保存ファイルを配置してください。"
    ],
    "scan": [
        "Inspection Depth",
        "Header checks supported model keys and shapes without comparing weight values. Full Scan reads all weights, checks finite values, and counts changed/nonmatrix parameters; it takes longer. Inspection does not extract or save a LoRA.",
        "Headerは対応するキーと形状を照合しますが、重みの値は比較しません。Full Scanは全重みを読み、有限値を検査し、変更箇所と行列以外の変更を数えます。その分時間がかかります。この検査ノードはLoRAの抽出・保存を行いません。"
    ],
    "max_rank": [
        "Maximum Rank",
        "Per-matrix rank ceiling for Auto Rank, not a target rank for every layer. If too small, the energy target is missed and reported. Larger ranks increase time and output size; a high-retention LoRA can exceed the full model in size. Only Auto Rank uses this setting.",
        "Auto Rankで各行列に許可するランク上限です。全層をこの値にそろえる指定ではありません。低すぎると保持率未達となり、レポートに記録します。上げるほど計算時間・容量が増え、高保持率では元モデルより大きくなる場合もあります。Auto Rankでのみ使用します。"
    ],
    "device": [
        "Compute Device",
        "SVD computation device. cpu avoids CUDA allocation. cuda requests GPU computation. auto chooses CUDA only when its memory estimate fits; otherwise CPU. CUDA out-of-memory retries on CPU with the same settings. Model files are read on CPU.",
        "SVDの計算先です。cpuはCUDAを使わず、cudaはGPUを指定します。autoはメモリ見積もりに余裕があるときだけCUDAを選び、それ以外はCPUを使います。CUDAメモリ不足時は同じ設定でCPUへ再試行します。モデルファイルの読み込みはCPUで行います。"
    ],
    "svd": [
        "SVD Method",
        "auto uses exact SVD for small matrices and randomized approximation for larger ones. exact is slower but uses the full spectrum for rank selection. randomized uses a seeded approximation and may retain less energy at the same rank. Fixed/Auto Rank only; Full Delta does not use SVD.",
        "autoは小さい行列に厳密SVD、大きい行列に乱数を使った近似SVDを使います。exactは時間がかかりますが、全特異値からランクを選びます。randomizedはシード付きの近似で、同じランクでも保持率が低くなる場合があります。Fixed／Auto Rank用で、Full Deltaでは使いません。"
    ],
    "non_lora": [
        "Non-LoRA Changes",
        "What to do with changed normalization/bias or other nonmatrix parameters: Stop rejects them. Store Diff preserves them as .diff in a ComfyUI hybrid file, which current Blend One Sources cannot blend. Exclude With Report drops them and marks the output PARTIAL; the complete model change is not reproduced. Full Delta stores all selected changes.",
        "正規化・biasなど、行列以外の変更の扱いです。Stopは停止します。Store Diffは.diffとして残すComfyUI混合形式で、現在のBlend One Sourcesには入力できません。Exclude With Reportは除外して部分抽出（PARTIAL）と記録するため、モデル全体の変更は再現しません。Full Deltaでは選択対象の差分をすべて保存します。"
    ],
    "key_filter": [
        "Target Key Pattern",
        "Wildcard filter on normalized parameter keys, e.g. diffusion_model.layers.*.proj.*.weight. * selects all targets. Excluded changed parameters are reported and count as reconstruction error; filtering can produce a partial model difference.",
        "正規化後のパラメーター名に使うワイルドカードです。例：diffusion_model.layers.*.proj.*.weight。*は全対象です。除外した変更箇所はレポートと再構成誤差に含まれ、部分的な差分抽出になります。"
    ],
    "seed": [
        "Seed",
        "Reproducibility seed for randomized SVD, not image generation. Keep this value and Seed Behavior fixed to repeat an approximation. Exact SVD and Full Delta do not use it for decomposition.",
        "近似SVDの乱数を再現するためのシードで、画像生成用ではありません。同じ近似を再現する場合は、この値とSeed Behaviorをfixedにします。厳密SVDとFull Deltaの分解処理では使用しません。"
    ],
    "control_after_generate": [
        "Seed Behavior",
        "Controls whether Seed stays fixed or changes after execution. fixed is recommended for reproducible randomized SVD. This changes extraction randomness, not an image-generation seed.",
        "実行後にSeedを固定するか変更するかを選びます。近似SVDを再現する場合はfixedを使います。抽出用の乱数を変える設定で、画像生成のシードではありません。"
    ],
    "control_before_generate": [
        "Seed Behavior (Before)",
        "Controls whether Seed stays fixed or changes before execution. Use fixed to repeat randomized SVD; this is not an image-generation control.",
        "実行前にSeedを固定するか変更するかを選びます。近似SVDの再現にはfixedを使います。画像生成用の設定ではありません。"
    ],
    "hash_inputs": [
        "Hash Inputs",
        "Calculate SHA256 of both complete input files and record it for identification and reproducibility. It adds disk reads and startup time, and does not change the extracted weights.",
        "両入力ファイル全体のSHA256を計算し、入力の識別・再現確認のために記録します。ファイル読み取りと開始時の待ち時間が増えますが、抽出される重みは変わりません。"
    ],
    "write_report": [
        "Write Report",
        "Write a sidecar .safetensors.report.json with settings, selected ranks, warnings, exclusions, and measured reconstruction errors. A report write failure warns while preserving a successfully saved main file.",
        "設定・採用ランク・警告・除外箇所・実測再構成誤差を.safetensors.report.jsonに保存します。レポートの書き込みだけが失敗した場合も、保存済みの本体は成功として残します。"
    ],
    "advanced": [
        "Advanced",
        "Show or hide infrequently changed extraction settings. Collapsing preserves all values and does not disable their effect.",
        "普段あまり変更しない抽出設定を表示・非表示にします。閉じても値は保持され、設定の効果も無効にはなりません。"
    ],
    "extract_status": [
        "Result",
        "Extraction progress, saved path, artifact type, layer count, relative weight error, and warnings. Weight error is not an image-quality score. Inspect shows model-structure/value checks without saving.",
        "抽出の進捗・保存先・形式・層数・相対重み誤差・警告を表示します。重み誤差は画像品質の点数ではありません。Inspectはモデルの構造・値の検査結果を表示し、保存しません。"
    ]
};

function settingsText(s) {
    return [`Base Model: ${s.base_model}`, `Modified Model: ${s.modified_model}`,
        s.mode === 'Fixed Rank' ? `Fixed Rank: ${s.rank}` : s.mode === 'Auto Rank'
            ? `Auto Rank / Target Energy: ${(s.energy * 100).toFixed(0)}% / Maximum Rank: ${s.max_rank}` : 'Full Delta',
        `Storage: ${s.dtype} / Device: ${s.device} / SVD: ${s.svd} / Seed: ${s.seed}`,
        `Non-LoRA Changes: ${s.non_lora} / Target: ${s.key_filter}`].join('\n');
}

function planText(plan, review = false) {
    const t = (en, ja) => text(app, en, ja);
    const reasons = {
        full_scan_required: t('Run Full Scan before applying recommendations. Header advice is provisional; manual extraction remains available.', '推奨値を適用するにはFull Scanを実行してください。Headerの情報は暫定です。手動抽出は利用できます。'),
        no_difference: t('No changed weights; no file to save.', '重みに変更がないため、保存する差分はありません。'),
        partial_consent_required: t('Nonmatrix changes cannot enter Blend One. Explicitly enable Allow Partial Extraction, or choose ComfyUI Loading.', '行列以外の変更はBlend Oneで合成できません。Allow Partial Extractionを明示的にONにするか、ComfyUI Loadingを選んでください。'),
        no_matrix_difference: t('Only nonmatrix changes exist; partial extraction would produce no LoRA.', '行列以外の変更だけなので、部分抽出してもLoRAは作成されません。'),
    };
    const s = plan.settings;
    const lines = [t('Recommended extraction settings', '推奨抽出設定'),
        `Base Model: ${plan.base_model}`, `Modified Model: ${plan.modified_model}`,
        `Output Purpose: ${plan.purpose}`,
        plan.ready ? t('Ready for extraction', '抽出可能') : reasons[plan.blocked_reason] || plan.blocked_reason,
        s.mode === 'Fixed Rank' ? `Fixed Rank: ${s.rank}` : `Auto Rank / Target Energy: ${(s.energy * 100).toFixed(0)}% / Maximum Rank: ${s.max_rank}`,
        `Storage: ${s.dtype} / Device: ${s.device} / SVD: ${s.svd} / Seed: ${s.seed}`,
        `Non-LoRA Changes: ${s.non_lora} / Target: ${s.key_filter}`,
        t('Rank 32 is a practical starting point, not a measured accuracy guarantee. Required rank / output size: not measured. Edit Rank freely, including 16, 64 or 256.', '32は実用的な開始値で、精度を測って決めた保証値ではありません。必要ランク／出力容量は未測定です。16・64・256など自由に変更できます。'),
        t('Fixed Rank does not target 99% energy. Target Energy applies only in Auto Rank mode.', '固定ランクでは保持率99％を目標にしません。Target EnergyはAuto Rank時だけ有効です。')];
    if (plan.changed_non_matrix) {
        lines.push(t(`Nonmatrix changes: ${plan.changed_non_matrix}`, `行列以外の変更：${plan.changed_non_matrix}個`),
            ...(plan.non_matrix_examples || []));
    }
    if (s.non_lora === 'Store Diff') lines.push(t('Store Diff preserves normalization/bias changes. Hybrid ComfyUI file; NOT a Blend One source.', '正規化・biasの変更を残すためStore Diffを推奨します。ComfyUI用混合形式となり、Blend Oneの合成元には使えません。'));
    if (plan.partial) lines.push(t('PARTIAL: changed nonmatrix weights will be excluded and reported.', '部分抽出：行列以外の変更を除外し、レポートに記録します。'));
    if (review) lines.push('', t('Review Only: no file saved. Select Extract and Save when ready.', '設定確認のみ：保存していません。確認後、Plan ActionをExtract and Saveに変更して実行してください。'));
    return lines.join('\n');
}

function extractText(raw) {
    try {
        const info = JSON.parse(raw);
        if (info.extraction_plan) return (info.effective_settings
            ? text(app, 'Settings used for this run (manual values)\n', '今回の実行設定（手動値）\n') + settingsText(info.effective_settings) + '\n\n' : '')
            + planText(info.extraction_plan, info.review_only) + (info.result_summary ? '\n\n' + info.result_summary : '');
    } catch {}
    return raw;
}

// Keep the STRING output untouched; present inspection JSON as readable local help.
function inspectionText(app, raw) {
    let info;
    try { info = JSON.parse(raw); } catch { return raw; }
    if (info?.architecture !== 'MOSS-Image' || !Number.isFinite(info.parameters)) return raw;
    const t = (en, ja) => text(app, en, ja);
    const bytes = value => Number.isFinite(value)
        ? `${(value / 2**30).toFixed(2)} GiB / ${(value / 1e9).toFixed(2)} GB (${value.toLocaleString('en-US')} bytes)` : '—';
    const largest = Number.isFinite(info.largest_fp32_tensor_bytes)
        ? `${(info.largest_fp32_tensor_bytes / 2**20).toFixed(2)} MiB` : '—';
    const ignored = index => info.ignored_buffers?.[index]?.join(', ') || t('none', 'なし');
    const lines = [
        t('Model inspection: MOSS-Image', 'モデル検査：MOSS-Image'),
        t(`Matched parameter entries: ${info.parameters} (logical entries after QKV normalization; not a layer count)`,
          `照合したパラメーター：${info.parameters}個（QKVを統一した後の論理項目数。層数ではありません）`),
        info.values_checked
            ? t('Inspection: Full Scan — weight values checked', '検査：Full Scan — 重みの値も検査済み')
            : t('Inspection: Header — keys and shapes only; weight values NOT compared', '検査：Header — 名前と形状だけを照合。重みの値は未比較です'),
    ];
    if (info.values_checked) {
        lines.push(t(`Changed entries: ${info.changed}; nonmatrix changes: ${info.changed_non_matrix}`,
            `変更のある項目：${info.changed}個／うち行列以外の変更：${info.changed_non_matrix}個`));
        lines.push(t(`LoRA matrix changes: ${info.changed - info.changed_non_matrix}; unchanged: ${info.parameters - info.changed}`,
            `LoRAに分解できる行列の変更：${info.changed - info.changed_non_matrix}個／変更なし：${info.parameters - info.changed}個`),
            t('Full Scan completed on CPU. Rank/retention analysis (SVD) was not performed.',
                'CPUでFull Scan完了。ランク・保持率を測るSVD分析は行っていません。'));
    } else {
        lines.push(t('This does not yet establish how much the models differ or their extraction accuracy.',
            'まだ差分の大きさや抽出精度を確認した結果ではありません。'));
    }
    lines.push('', t('Excluded buffers (not extracted)', '対象外のバッファー（抽出しないデータ）'),
        `  Base Model: ${ignored(0)}`, `  Modified Model: ${ignored(1)}`);
    if (info.ignored_buffers?.flat().includes('rope.bands')) lines.push(t(
        '  rope.bands: known rotary-position buffer reconstructed by ComfyUI; excluded from weight differences.',
        '  rope.bands：ComfyUIで再構成する既知の位置情報バッファーのため、重み差分から除外します。'));
    lines.push('', t('Data sizes (not a RAM/VRAM or LoRA output estimate)', 'データ容量の目安（必要RAM・VRAMやLoRA出力サイズの予測ではありません）'),
        t(`Largest single FP32 tensor: ${largest}`, `最大1テンソルのFP32容量：${largest}`),
        t(`BF16 Full Delta tensor-data upper bound: ${bytes(info.full_delta_bf16_bytes_upper_bound)}`,
          `BF16フル差分・テンソル部分の容量上限：${bytes(info.full_delta_bf16_bytes_upper_bound)}`),
        t('Working memory and temporary disk space also depend on mode, SVD and rank.',
          '作業メモリーと一時ディスク容量は、抽出方式・SVD・ランクでも変わります。'), '',
        info.values_checked
            ? t('Next: review the recommendations, then use Model Extract to save. The size estimates above depend on shapes, so they are the same in Header and Full Scan.',
                '次の操作：推奨設定を確認し、Model Extractで保存します。上の容量は形状から計算するため、HeaderとFull Scanで同じ値です。')
            : t('Next: Full Scan checks changes. Model Extract saves the difference; this Inspect node does not save a LoRA.',
                '次の操作：変更箇所を数えるならFull Scan、差分を保存するならModel Extractを使います。このInspectノードはLoRAを保存しません。'));
    return (info.recommendation ? planText(info.recommendation) + '\n\n' : '') + lines.join('\n');
}

const names = new Set(['LoRABlendModelInspect', 'LoRABlendModelExtract']);
app.registerExtension({
    name: 'LoRABlendOne.Extractor',
    beforeRegisterNodeDef(type, data) {
        if (!names.has(data.name)) return;
        const englishTitle = data.name === 'LoRABlendModelInspect' ? 'LoRA Blend Model Inspect' : 'LoRA Blend Model Extract';
        const defaultTitles = new Set([data.name, data.display_name, englishTitle,
            'LoRA Blend モデル差分確認', 'LoRA Blend モデル差分抽出']);
        const created = type.prototype.onNodeCreated;
        type.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            const node = this;
            const originals = [...(node.widgets || [])];
            const advanced = new Set(Object.entries(data.input?.optional || {}).filter(([,spec]) => spec[1]?.advanced).map(([key]) => key));
            // ComfyUI adds seed companions outside INPUT_TYPES; keep them with Seed.
            for (const widget of originals) {
                if (advanced.has(widget.name)) {
                    for (const linked of widget.linkedWidgets || []) advanced.add(linked.name);
                }
                if (advanced.has('seed') && ['control_after_generate', 'control_before_generate'].includes(widget.name)) advanced.add(widget.name);
            }
            const sizes = new Map(originals.map(w => [w, {type:w.type, computeSize:w.computeSize}]));
            node.properties ??= {};
            const button = node.addWidget('button', '', null, () => {
                node.properties.extract_advanced = !node.properties.extract_advanced;
                refresh();
            }, { serialize:false });
            button.serialize = false;
            node.widgets.splice(node.widgets.indexOf(button), 1);
            const firstAdvanced = originals.findIndex(w => advanced.has(w.name));
            node.widgets.splice(firstAdvanced < 0 ? originals.length : firstAdvanced, 0, button);
            const status = document.createElement('textarea');
            status.readOnly = true;
            status.style.cssText = 'width:100%;height:100%;background:#202020;color:#ddd;resize:none;font-size:12px;box-sizing:border-box;overflow:auto;';
            const statusWidget = node.addDOMWidget?.('extract_status', 'text', status, {serialize:false, getMinHeight: () => 100});
            if (statusWidget) statusWidget.serialize = false;
            node.lboExtractStatus = status;
            if (data.name === 'LoRABlendModelExtract') {
                trackNodeRuns(node, data.name, {
                    executed: detail => applyResult(detail.output),
                    'lora_blend_one.extract_status': detail => {
                        let value = detail.text;
                        if (value.startsWith('Stopped')) value = text(app, value,
                            value.replace('Stopped', '停止') + '\n入力ファイル・対象キー・保存精度を確認してください。');
                        node.lboExtractMessage = null;
                        status.value = value;
                    },
                });
            }
            let preset, apply;
            if (data.name === 'LoRABlendModelExtract') {
                preset = node.addWidget('combo', 'Rank Preset', '32', value => {
                    if (value === 'Custom') return;
                    const rank = originals.find(w => w.name === 'rank');
                    if (node.inputs?.some(i => i.name === 'rank' && i.link != null)) {
                        refresh();
                        status.value = text(app, 'Rank has an input connection. Change its source or disconnect it.', 'Rankに入力接続があります。接続元を変更するか切断してください。');
                        return;
                    }
                    rank.value = Number(value);
                    rank.callback?.(rank.value);
                    node.graph?.change?.(); refresh();
                }, {values:['16','32','64','128','256','Custom'], serialize:false});
                preset.serialize = false;
                node.widgets.splice(node.widgets.indexOf(preset), 1);
                node.widgets.splice(node.widgets.indexOf(originals.find(w => w.name === 'rank')), 0, preset);
                apply = node.addWidget('button', 'Apply Recommendations', null, () => {
                    try {
                        const input = node.inputs?.find(i => i.name === 'extraction_plan');
                        const link = node.graph?.links?.[input?.link];
                        const source = node.graph?.getNodeById(link?.origin_id);
                        const info = JSON.parse(source?.properties?.extract_inspection_result || 'null');
                        const plan = info?.recommendation;
                        if (!plan?.ready || plan.version !== 2 || !info.values_checked)
                            throw new Error(text(app, 'Run Full Scan on the connected Inspect and resolve its warnings first.', '接続したInspectでFull Scanを実行し、停止理由を解消してください。'));
                        for (const key of ['base_model','modified_model']) {
                            if (source.widgets?.find(w => w.name === key)?.value !== plan[key])
                                throw new Error(text(app, 'Inspect models changed. Run Full Scan again.', 'Inspectのモデルが変わりました。Full Scanを再実行してください。'));
                        }
                        const values = {base_model:plan.base_model, modified_model:plan.modified_model};
                        for (const key of ['mode','rank','energy','max_rank','dtype','device','svd','non_lora','key_filter','seed'])
                            values[key] = plan.settings[key];
                        // Validate everything before writing any widget; preserve saved serialization order.
                        for (const key of Object.keys(values)) {
                            if (!originals.some(w => w.name === key) || node.inputs?.some(i => i.name === key && i.link != null))
                                throw new Error(text(app, `Cannot apply ${key}: missing widget or connected input.`, `${key}を適用できません。設定欄がないか、入力が接続されています。`));
                        }
                        for (const [key,value] of Object.entries(values)) originals.find(w => w.name === key).value = value;
                        for (const key of ['control_after_generate','control_before_generate']) {
                            const widget = originals.find(w => w.name === key); if (widget) widget.value = 'fixed';
                        }
                        node.lboExtractMessage = null;
                        node.graph?.change?.(); refresh();
                    } catch (error) { status.value = error.message; }
                }, {serialize:false});
                apply.serialize = false;
                node.widgets.splice(node.widgets.indexOf(apply), 1);
                node.widgets.splice(node.widgets.indexOf(button), 0, apply);
                const action = originals.find(w => w.name === 'plan_action');
                if (action) {
                    // Keep the native value and serialization slot; replace only its presentation.
                    sizes.get(action).computeSize = () => [340, 56];
                    let buttonY = 0;
                    action.draw = (ctx, owner, width, y) => {
                        buttonY = y + 3;
                        const choices = ['Review Only', 'Extract and Save'];
                        const half = (width - 28) / 2;
                        ctx.save();
                        ctx.font = 'bold 12px sans-serif';
                        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                        choices.forEach((value, index) => {
                            const selected = action.value === value;
                            ctx.fillStyle = selected ? (index ? '#236c48' : '#72551d') : '#292d32';
                            ctx.strokeStyle = selected ? (index ? '#8fe2b1' : '#f0c96c') : '#777';
                            ctx.lineWidth = selected ? 2 : 1;
                            ctx.beginPath(); ctx.roundRect(10 + index * (half + 8), buttonY, half, 30, 5);
                            ctx.fill(); ctx.stroke(); ctx.fillStyle = '#fff';
                            ctx.fillText(`${selected ? '✓ ' : ''}${value}`, 10 + index * (half + 8) + half / 2, buttonY + 15);
                        });
                        ctx.font = '12px sans-serif'; ctx.fillStyle = '#ddd';
                        ctx.fillText(action.value === 'Extract and Save'
                            ? text(app, 'Next Run: extract and save', '次のRun：抽出して保存')
                            : text(app, 'Next Run: review without saving', '次のRun：保存せず確認'), width / 2, y + 47);
                        ctx.restore();
                    };
                    action.mouse = (event, pos) => {
                        if (event.type !== 'pointerdown' || (event.button != null && event.button !== 0)) return false;
                        if (node.inputs?.some(i => i.name === 'plan_action' && i.link != null)) return false;
                        const half = (node.size[0] - 28) / 2;
                        if (pos[1] < buttonY || pos[1] > buttonY + 30) return false;
                        const index = pos[0] >= 10 && pos[0] <= 10 + half ? 0
                            : pos[0] >= 18 + half && pos[0] <= node.size[0] - 10 ? 1 : -1;
                        if (index < 0) return false;
                        const value = ['Review Only', 'Extract and Save'][index], previous = action.value;
                        if (value !== previous) {
                            node.graph?.beforeChange?.();
                            action.value = value;
                            action.callback?.(value, app.canvas, node, pos, event);
                            node.onWidgetChanged?.(action.name, value, previous, action);
                            node.graph?.incrementVersion?.(); node.graph?.afterChange?.();
                            refresh();
                        }
                        return true;
                    };
                    action.onPointerDown = (pointer, owner, canvas) => pointer.eDown
                        ? action.mouse(pointer.eDown, [canvas.graph_mouse[0] - owner.pos[0], canvas.graph_mouse[1] - owner.pos[1]]) : false;
                }
            }
            function refresh() {
                const open = !!node.properties.extract_advanced;
                const connected = node.inputs?.some(i => i.name === 'extraction_plan' && i.link != null);
                if (!node.title || defaultTitles.has(node.title)) node.title = englishTitle;
                const raw = data.name === 'LoRABlendModelInspect'
                    ? node.properties.extract_inspection_result ?? node.lboExtractMessage : node.lboExtractMessage;
                if (raw != null) status.value = data.name === 'LoRABlendModelInspect'
                    ? inspectionText(app, raw) : extractText(raw);
                else if (data.name === 'LoRABlendModelExtract') status.value = connected
                    ? text(app, 'Inspect connected for reference. Settings above remain editable. Apply Recommendations copies them once; Review Only checks without saving.', 'Inspectの結果を参照しています。上の設定は接続中も編集できます。Apply Recommendationsで一度だけ反映し、Review Onlyで保存せず確認できます。')
                    : text(app, 'Manual extraction: settings on this node are used.', '手動抽出：このノードの設定を使用します。');
                button.name = `${open ? '▼' : '▶'} Advanced`;
                button.tooltip = text(app, ...extractorHelp.advanced.slice(1));
                status.title = text(app, ...extractorHelp.extract_status.slice(1));
                if (statusWidget) {
                    statusWidget.label = 'Result';
                    statusWidget.tooltip = status.title;
                }
                const mode = originals.find(w => w.name === 'mode')?.value;
                for (const w of originals) {
                    const help = extractorHelp[w.name];
                    if (help) {
                        w.label = help[0];
                        w.tooltip = text(app, help[1], help[2]);
                    }
                    const hidden = (!connected && w.name === 'plan_action')
                        || (advanced.has(w.name) && !open) || (w.name === 'rank' && mode !== 'Fixed Rank')
                        || (w.name === 'energy' && mode !== 'Auto Rank');
                    w.type = hidden ? 'lbo_hidden' : sizes.get(w).type;
                    w.computeSize = hidden ? () => [0,-4] : sizes.get(w).computeSize;
                }
                if (preset) {
                    const rank = originals.find(w => w.name === 'rank')?.value;
                    preset.value = ['16','32','64','128','256'].includes(String(rank)) ? String(rank) : 'Custom';
                    preset.type = mode === 'Fixed Rank' ? 'combo' : 'lbo_hidden';
                    preset.computeSize = mode === 'Fixed Rank' ? undefined : () => [0,-4];
                    preset.tooltip = text(app, 'Quick choices for Fixed Rank. Custom keeps any integer entered in Rank.', '固定ランクを簡単に選べます。CustomではRank欄に入力した任意の整数を維持します。');
                    apply.type = connected ? 'button' : 'lbo_hidden';
                    apply.computeSize = connected ? undefined : () => [0,-4];
                    apply.tooltip = text(app, 'Copy the completed Inspect recommendation once, including its models. You can edit all settings afterwards. Output filename is preserved.', '完了したInspectの推奨値とモデルを一度だけコピーします。適用後も全設定を編集できます。出力ファイル名は維持します。');
                }
                node.setSize?.([Math.max(node.size?.[0] || 340, 340), Math.max(node.size?.[1] || 0, node.computeSize()[1])]);
                node.setDirtyCanvas?.(true,true);
            }
            const mode = originals.find(w => w.name === 'mode');
            if (mode) {
                const callback = mode.callback;
                mode.callback = function (...args) { const r = callback?.apply(this,args); refresh(); return r; };
            }
            if (data.name === 'LoRABlendModelExtract') {
                for (const widget of originals) {
                    const callback = widget.callback;
                    widget.callback = function (...args) {
                        const r = callback?.apply(this,args); node.lboExtractMessage = null; refresh(); return r;
                    };
                }
            }
            if (data.name === 'LoRABlendModelInspect') {
                for (const widget of originals) {
                    const callback = widget.callback;
                    widget.callback = function (...args) {
                        const r = callback?.apply(this, args);
                        delete node.properties.extract_inspection_result;
                        node.lboExtractMessage = null;
                        status.value = text(app, 'Settings changed. Run Inspect again to refresh recommendations.', '設定が変わりました。Inspectを再実行して推奨値を更新してください。');
                        return r;
                    };
                }
            }
            for (const widget of originals.filter(w => ['control_after_generate', 'control_before_generate'].includes(w.name))) {
                const callback = widget.callback;
                widget.callback = function (...args) { const result = callback?.apply(this, args); refresh(); return result; };
            }
            const configured = node.onConfigure;
            node.onConfigure = function (...args) { const r = configured?.apply(this,args); refresh(); return r; };
            const connections = node.onConnectionsChange;
            node.onConnectionsChange = function (...args) {
                const r = connections?.apply(this,args);
                node.lboExtractMessage = null;
                refresh();
                return r;
            };
            function applyResult(message) {
                if (!node.title || defaultTitles.has(node.title)) node.title = englishTitle;
                if (message?.text) {
                    node.lboExtractMessage = message.text.join('\n');
                    if (data.name === 'LoRABlendModelInspect' && node.properties.extract_inspection_result !== node.lboExtractMessage) {
                        node.properties.extract_inspection_result = node.lboExtractMessage;
                        node.graph?.change?.();
                    }
                    status.value = data.name === 'LoRABlendModelInspect'
                        ? inspectionText(app, node.lboExtractMessage) : extractText(node.lboExtractMessage);
                }
            }
            if (data.name === 'LoRABlendModelInspect') {
                const executed = node.onExecuted;
                node.onExecuted = function (message) {
                    const result = executed?.call(this, message);
                    applyResult(message);
                    return result;
                };
            }
            watchLocale(app, node, refresh);
            refresh();
            return result;
        };
    },
});
