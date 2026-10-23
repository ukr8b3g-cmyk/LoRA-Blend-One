import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';
import { text, watchLocale } from './i18n.js';

const FILE_KEY = 'lora_blend_recipe_file';
const IMPORT_KEY = 'lora_blend_recipe_import';
const importChoice = node => node.properties?.[IMPORT_KEY] ? `📥 ${node.properties[IMPORT_KEY].filename}` : null;

export function applyRecipeToNodes(source, node, result, allowUnresolved = false) {
    if ((!result.ready && !allowUnresolved) || result.issues.length || !source.graph || source.graph !== node.graph) return false;
    const changes = [];
    for (let slot = 1; slot <= 8; slot++) {
        const item = result.sources[slot - 1];
        changes.push([source, `lora_${slot}`, item ? item.selected || item.name : '(none)'],
            [source, `weight_${slot}`, item ? item.weight * 100 : 0],
            [source, `enabled_${slot}`, !!item && !!result.ready]);
    }
    changes.push(...Object.entries(result.settings).map(([name, value]) => [node, name, value]));
    for (const control of ['control_after_generate', 'control_before_generate']) {
        if (node.widgets.some(widget => widget.name === control)) changes.push([node, control, 'fixed']);
    }
    if (changes.some(([target, name]) => !target.widgets.some(widget => widget.name === name)
        || target.inputs?.some(input => input.name === name && input.link != null))) return false;
    const graph = node.graph;
    graph.beforeChange?.();
    try {
        for (const [target, name, value] of changes) {
            const widget = target.widgets.find(widget => widget.name === name), old = widget.value;
            if (name.startsWith('lora_') && Array.isArray(widget.options?.values) && !widget.options.values.includes(value))
                widget.options.values.push(value);
            widget.value = value;
            widget.callback?.(value, app.canvas, target);
            target.onWidgetChanged?.(name, value, old, widget);
        }
        source.loraBlendSourceToggles?.refresh();
        source.loraBlendClearFailure?.(true);
        source.loraBlendAccordion?.refresh();
        node.loraBlendClearFailure?.(true);
        node.loraBlendAccordion?.refresh();
        graph.incrementVersion?.();
    } finally {
        graph.afterChange?.();
    }
    return true;
}

function connectedSources(node) {
    const input = node.inputs?.find(input => input.name === 'sources');
    const link = node.graph?.links?.get?.(input?.link) ?? node.graph?.links?.[input?.link];
    const source = link && node.graph.getNodeById(link.origin_id);
    return source?.type === 'LoRABlendSources' ? source : null;
}

export function installRecipeControls(node) {
    if (node.loraBlendRecipe || !node.addDOMWidget) return;
    let data = null, message = null, busy = false, loaded = false, requestId = 0;
    const files = ['(none)'];
    const summary = document.createElement('textarea');
    summary.readOnly = true;
    summary.className = 'comfy-multiline-input';
    summary.style.cssText = 'width:100%;height:100%;resize:none;box-sizing:border-box;font:12px sans-serif;white-space:pre-wrap;overflow-wrap:anywhere;';
    const select = node.addWidget('combo', '', node.properties?.[FILE_KEY] || '(none)', value => {
        node.properties ??= {};
        node.properties[FILE_KEY] = value;
        return inspect();
    }, { values: files, serialize: false });
    const read = node.addWidget('button', '', null, () => inspect(), { serialize: false });
    const details = node.addDOMWidget('lbo_recipe_summary', 'customtext', summary, { serialize: false });
    details.computeSize = width => [width, 240];
    const restore = node.addWidget('button', '', null, () => restoreSettings(), { serialize: false });
    for (const widget of [select, read, details, restore]) {
        widget.serialize = false;
        widget.loraBlendAdvanced = true;
    }

    function render() {
        select.name = text(app, 'Saved Blend LoRA', '保存済みブレンドLoRA');
        read.name = text(app, 'Read / Refresh Recipe', 'レシピを確認・更新');
        restore.name = text(app, 'Restore Sources + Blend One', 'SourcesとBlend Oneに復元');
        summary.setAttribute('aria-label', text(app, 'Saved blend recipe', '保存済みブレンドのレシピ'));
        if (busy) summary.value = text(app, 'Reading recipe and checking source hashes…', 'レシピと元LoRAのハッシュを確認中…');
        else if (message) summary.value = text(app, message.en, message.ja);
        else if (!data) summary.value = text(app, 'Select a saved blend LoRA to inspect its recipe.', '保存済みブレンドLoRAを選んでレシピを確認します。');
        else {
            const status = {
                matched: text(app, 'SHA256 matched', 'SHA256一致'),
                unverified: text(app, 'Name matched; no recorded hash', '名前一致・ハッシュ記録なし'),
                missing: text(app, 'Missing source', '元LoRAが見つかりません'),
                hash_mismatch: text(app, 'SHA256 mismatch', 'SHA256不一致'),
                ambiguous: text(app, 'Ambiguous filename', '同名ファイルを特定できません'),
                unreadable: text(app, 'Cannot read source', '元LoRAを読み込めません'),
            };
            const lines = [text(app, 'Saved blend recipe', '保存済みブレンドのレシピ'), data.file, ''];
            data.sources.forEach((source, index) => lines.push(
                `LoRA ${index + 1}: ${source.name}`,
                `${text(app, 'Coefficient', '合成係数')}: ${source.weight}  → direct ${source.weight * 100}%`,
                `${status[source.status]}${source.selected ? `: ${source.selected}` : ''}`, ''));
            const s = data.settings;
            lines.push(`${text(app, 'Method', '方式')}: ${s.merge_method}`,
                `${text(app, 'Rank', 'ランク')}: ${s.output_rank === 'Custom' ? s.custom_rank : s.output_rank}`,
                `${text(app, 'Compression', '圧縮')}: ${s.compression_mode}`,
                `${text(app, 'Storage / Global strength', '保存精度・全体強度')}: ${s.output_dtype} / ${s.output_scale}`,
                `${text(app, 'Seed', 'Seed')}: ${s.seed} (${text(app, 'fixed', '固定')})`, '',
                text(app, 'Restores effective coefficients in direct mode; original percentage entries may differ.', '保存された合成係数をdirectで復元します。元の入力欄の数値と異なる場合があります。'));
            if (data.issues.length) lines.push('', text(app, 'Unsupported settings:', '復元できない設定：'), ...data.issues);
            lines.push('', data.ready ? text(app, 'Ready to restore. Filename Prefix will be kept.', '復元できます。保存名は維持します。')
                : text(app, 'Cannot restore. Resolve the source/settings issues above, then refresh.', '復元できません。上記の元LoRA・設定の問題を解消し、再確認してください。'));
            if (node.properties?.[IMPORT_KEY] && !data.ready) lines.push(text(app,
                'Dropped recipe: all source slots start OFF until the issues are resolved and restored.',
                'ドロップしたレシピ：問題を解消して復元するまで、全LoRA枠はOFFで開始します。'));
            summary.value = lines.join('\n');
        }
        (node.graph || app.graph)?.setDirtyCanvas?.(true, true);
    }

    async function refreshFiles() {
        const response = await api.fetchApi('/lora_blend_one/recipe_files');
        if (!response.ok) throw new Error(text(app, 'Recipe API unavailable. Restart ComfyUI and reload the browser.', 'レシピAPIを利用できません。ComfyUIを再起動し、ブラウザーを再読み込みしてください。'));
        const names = await response.json();
        files.splice(0, files.length, '(none)', ...(importChoice(node) ? [importChoice(node)] : []), ...names);
    }

    async function inspect() {
        const id = ++requestId, graph = node.graph, filename = select.value;
        data = null; message = null;
        if (filename === '(none)') { busy = false; render(); return null; }
        busy = true; render();
        try {
            await refreshFiles();
            const imported = filename === importChoice(node) ? node.properties[IMPORT_KEY] : null;
            const response = await api.fetchApi(imported ? '/lora_blend_one/recipe_from_metadata' : '/lora_blend_one/recipe', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(imported || { filename }),
            });
            if (!response.ok && response.status === 404) throw new Error(text(app, 'Restart ComfyUI to enable the recipe API.', 'レシピAPIを有効にするためComfyUIを再起動してください。'));
            const result = await response.json();
            if (!response.ok) {
                const error = new Error(result.error || response.statusText);
                error.recipeCode = result.code;
                throw error;
            }
            if (id !== requestId || graph !== node.graph) return null;
            data = result;
            return result;
        } catch (error) {
            const reasons = {
                no_recipe: ['This file has no LoRA Blend One recipe.', 'このファイルにLoRA Blend Oneのレシピはありません。'],
                unavailable_file: ['Select an installed file and refresh the list.', '存在するファイルを選び、一覧を更新してください。'],
                unsupported_version: ['Unsupported recipe version.', 'このレシピのバージョンには対応していません。'],
            };
            const [en, ja] = reasons[error.recipeCode] || [error.message, error.message];
            if (id === requestId) message = { en: `Cannot read recipe.\n${en}`, ja: `レシピを確認できません。\n${ja}` };
            return null;
        } finally {
            if (id === requestId) { busy = false; render(); }
        }
    }

    async function restoreSettings() {
        if (busy || !data?.ready) return;
        const graph = node.graph;
        const result = await inspect(); // Check that the selected files still match before changing any widgets.
        if (!result?.ready || !graph || graph !== node.graph || graph.getNodeById(node.id) !== node
            || (graph !== app.graph && graph !== app.rootGraph)) return;
        const source = connectedSources(node);
        if (!source) {
            message = { en: 'Connect a LoRA Blend Sources node to Sources, then restore.', ja: 'LoRA Blend Sourcesを接続してから復元してください。' };
            render(); return;
        }
        if (!applyRecipeToNodes(source, node, result)) {
            message = { en: 'Cannot restore connected or unavailable inputs. Use the current node widgets and disconnect overriding inputs.',
                ja: '入力が接続済み、または必要な項目がありません。現在のノードを使い、設定を上書きする入力接続を外してください。' };
            render(); return;
        }
        message = { en: 'Restored Sources and Blend One. Coefficients use direct mode; seed is fixed. Filename Prefix was kept.',
            ja: 'SourcesとBlend Oneへ復元しました。比率はdirect、Seedは固定です。保存名は維持しました。' };
        render();
    }

    node.loraBlendRecipe = {
        setImported(recipe, inspection) {
            node.properties ??= {};
            node.properties[IMPORT_KEY] = { filename: inspection.file, recipe };
            select.value = node.properties[FILE_KEY] = importChoice(node);
            if (!files.includes(select.value)) files.push(select.value);
            data = inspection; message = null;
            render();
        },
        async refresh(open) {
            if (!open || loaded) return;
            loaded = true;
            try { await refreshFiles(); }
            catch (error) { message = { en: error.message, ja: error.message }; loaded = false; render(); }
        },
    };
    const configured = node.onConfigure;
    node.onConfigure = function (...args) {
        const result = configured?.apply(this, args);
        ++requestId; data = null; busy = false; message = null;
        select.value = this.properties?.[FILE_KEY] || '(none)';
        if (select.value === importChoice(this) && !files.includes(select.value)) files.push(select.value);
        render();
        if (select.value === importChoice(this)) queueMicrotask(() => void inspect());
        return result;
    };
    const removed = node.onRemoved;
    node.onRemoved = function (...args) { ++requestId; return removed?.apply(this, args); };
    watchLocale(app, node, render);
    render();
}
