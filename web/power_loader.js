import { app } from '../../scripts/app.js';
import { text, watchLocale } from './i18n.js';

const LOADER_TYPE = 'Power Lora Loader (rgthree)';
const LOADER_KEY = 'lora_blend_power_loader_id';
const NONE = '(none)';

export function installPowerLoaderControls(node) {
    if (node.loraBlendPowerLoader || !node.addDOMWidget) return;
    let notice = null;
    const loaders = () => (node.graph?._nodes || []).filter(other => other.type === LOADER_TYPE);
    const label = loader => `#${loader.id} ${loader.title || LOADER_TYPE}`;
    const selected = () => loaders().find(loader => String(loader.id) === String(node.properties?.[LOADER_KEY]));
    const summary = document.createElement('textarea');
    summary.readOnly = true;
    summary.className = 'comfy-multiline-input';
    summary.style.cssText = 'width:100%;height:100%;resize:none;box-sizing:border-box;font:12px sans-serif;white-space:pre-wrap;overflow-wrap:anywhere;';
    const select = node.addWidget('combo', '', NONE, value => {
        const loader = loaders().find(loader => label(loader) === value);
        node.graph?.beforeChange?.();
        node.properties ??= {};
        if (loader) node.properties[LOADER_KEY] = loader.id;
        else delete node.properties[LOADER_KEY];
        notice = null;
        node.graph?.afterChange?.();
        render();
    }, { values: () => [NONE, ...loaders().map(label)], serialize: false });
    const button = node.addWidget('button', '', null, () => importSettings(), { serialize: false });
    const details = node.addDOMWidget('lbo_power_loader_summary', 'customtext', summary, { serialize: false });
    details.computeSize = width => [width, 100];
    for (const widget of [select, button, details]) {
        widget.serialize = false;
        widget.loraBlendAdvanced = true;
    }

    function render() {
        select.name = text(app, 'Power LoRA Loader', '取り込み元Power LoRA Loader');
        button.name = text(app, 'Import Enabled LoRAs', 'ONのLoRA設定を取り込む');
        select.value = selected() ? label(selected()) : NONE;
        summary.setAttribute('aria-label', text(app, 'Power LoRA Loader import', 'Power LoRA Loaderからの取り込み'));
        summary.value = notice ? text(app, notice.en, notice.ja) : loaders().length
            ? text(app, 'Select a loader, then import its ON rows (up to 8). Strength 0.8 becomes 80%. Connected Blend One nodes use direct mode. Import again after changing the loader.',
                '取り込み元を選び、ONの行を取り込みます（最大8本）。強度0.8は80%に変換し、接続先のBlend Oneをdirectにします。元のLoaderを変更したら再度取り込んでください。')
            : text(app, 'No Power LoRA Loader (rgthree) in this workflow. Add one to import its settings.',
                'このワークフローにPower LoRA Loader (rgthree)がありません。追加すると設定を取り込めます。');
        (node.graph || app.graph)?.setDirtyCanvas?.(true, true);
    }

    function stop(english, japanese) {
        const error = new Error(english);
        error.japanese = japanese;
        throw error;
    }

    function importSettings() {
        try {
            const graph = node.graph, loader = selected();
            if (!loader || !graph || graph.getNodeById(node.id) !== node
                || (graph !== app.graph && graph !== app.rootGraph))
                stop('Select a Power LoRA Loader in the current workflow.', '現在のワークフローのPower LoRA Loaderを選んでください。');
            const rows = loader.widgets.filter(widget => /^lora_\d+$/.test(widget.name)).map(widget => widget.value);
            const enabled = rows.filter(row => row?.on === true);
            if (!enabled.length) stop('No ON LoRAs. Enable a row in the selected loader.', 'ONのLoRAがありません。取り込み元の行をONにしてください。');
            if (enabled.length > 8) stop(`${enabled.length} ON LoRAs exceed the 8-source limit. Turn OFF enough rows, then import again.`,
                `ONのLoRAが${enabled.length}本あり、上限の8本を超えています。不要な行をOFFにしてから再度取り込んでください。`);
            const blends = graph._nodes.filter(other => {
                if (other.type !== 'LoRABlendOneSplit') return false;
                const input = other.inputs?.find(input => input.name === 'sources');
                const link = graph.links?.get?.(input?.link) ?? graph.links?.[input?.link];
                return link && String(link.origin_id) === String(node.id);
            });
            if (!blends.length) stop('Connect Sources directly to Blend One, then import. This sets its weight mode to direct.',
                'SourcesをBlend Oneへ直接接続してから取り込んでください。接続先をdirectに設定します。');
            const changes = [];
            for (let slot = 1; slot <= 8; slot++) {
                const row = enabled[slot - 1];
                let filename = NONE, weight = 0;
                if (row) {
                    const file = node.widgets.find(widget => widget.name === `lora_${slot}`);
                    const choices = typeof file?.options?.values === 'function' ? file.options.values(file, node) : file?.options?.values;
                    const normal = name => name.replace(/\\/g, '/');
                    if (typeof row.lora !== 'string' || !row.lora || !Array.isArray(choices))
                        stop(`LoRA ${slot}: unavailable filename list. Refresh the node definitions and select an installed file.`,
                            `LoRA ${slot}：ファイル一覧を確認できません。ノード定義を更新し、存在するファイルを選んでください。`);
                    let matches = choices.filter(name => name !== NONE && normal(name) === normal(row.lora));
                    if (!matches.length) matches = choices.filter(name => name !== NONE && normal(name).split('/').at(-1) === normal(row.lora).split('/').at(-1));
                    if (matches.length !== 1) stop(`${row.lora}: ${matches.length ? 'ambiguous filename' : 'file not found'}. Select its full installed filename in the loader, then import again.`,
                        `${row.lora}：${matches.length ? '同名ファイルを特定できません' : 'ファイルが見つかりません'}。取り込み元で存在するファイルのフル名称を選び、再度取り込んでください。`);
                    filename = matches[0];
                    if (typeof row.strength !== 'number' || !Number.isFinite(row.strength))
                        stop(`${row.lora}: invalid strength. Enter a finite number in the loader.`, `${row.lora}：強度が不正です。取り込み元に有限の数値を入力してください。`);
                    weight = row.strength * 100;
                    const limits = node.widgets.find(widget => widget.name === `weight_${slot}`)?.options;
                    if (!Number.isFinite(weight) || weight < (limits?.min ?? -Infinity) || weight > (limits?.max ?? Infinity))
                        stop(`${row.lora}: strength ${row.strength} is outside this Sources input range. Adjust the loader strength before importing.`,
                            `${row.lora}：強度${row.strength}はSourcesの入力範囲外です。取り込み元の強度を調整してください。`);
                    if (loader.properties?.['Show Strengths'] === 'Separate Model & Clip'
                        && loader.inputs?.some(input => input.name === 'clip' && input.link != null)
                        && (row.strengthTwo ?? 1) !== row.strength)
                        stop(`${row.lora}: different Model/CLIP strengths cannot use one blend coefficient. Set equal strengths or disconnect CLIP before importing.`,
                            `${row.lora}：ModelとCLIPの異なる強度を1つの合成係数で再現できません。同じ強度にするかCLIP入力を外してから取り込んでください。`);
                }
                changes.push([node, `lora_${slot}`, filename], [node, `weight_${slot}`, weight], [node, `enabled_${slot}`, !!row]);
            }
            for (const blend of blends) changes.push([blend, 'weight_mode', 'direct']);
            for (const [target, name] of changes) {
                if (!target.widgets.some(widget => widget.name === name)
                    || target.inputs?.some(input => input.name === name && input.link != null))
                    stop('An imported setting is missing or overridden by a connected input. Disconnect that input and use the current Sources/Blend One nodes.',
                        '取り込み先の項目がないか、入力接続で上書きされています。その接続を外し、現在のSources／Blend Oneノードを使用してください。');
            }
            graph.beforeChange?.();
            try {
                for (const [target, name, value] of changes) {
                    const widget = target.widgets.find(widget => widget.name === name), old = widget.value;
                    widget.value = value;
                    widget.callback?.(value, app.canvas, target);
                    target.onWidgetChanged?.(name, value, old, widget);
                }
                for (const target of [node, ...blends]) {
                    target.loraBlendClearFailure?.(true);
                    target.loraBlendAccordion?.refresh();
                }
                node.loraBlendSourceToggles?.refresh();
                graph.incrementVersion?.();
            } finally {
                graph.afterChange?.();
            }
            notice = {
                en: `Imported ${enabled.length} ON LoRAs from ${label(loader)}; ${rows.length - enabled.length} OFF rows skipped. ${blends.length} connected Blend One node(s) set to direct. Other merge settings were kept. Import again after loader changes.`,
                ja: `${label(loader)}からONの${enabled.length}本を取り込み、OFFの${rows.length - enabled.length}本を除外しました。接続先のBlend One ${blends.length}個をdirectに設定し、その他の合成設定は保持しました。元のLoaderを変更したら再度取り込んでください。`,
            };
        } catch (error) {
            notice = { en: `Cannot import.\n${error.message}`, ja: `取り込めません。\n${error.japanese || error.message}` };
        }
        render();
    }

    node.loraBlendPowerLoader = { refresh: render };
    const configured = node.onConfigure;
    node.onConfigure = function (...args) {
        const result = configured?.apply(this, args);
        notice = null;
        render();
        return result;
    };
    watchLocale(app, node, render);
    render();
}
