import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';
import { text } from './i18n.js';
import { applyRecipeToNodes } from './recipe.js';

// Only the safetensors header is read. Tensor bytes remain in the dropped file.
export async function droppedRecipe(file) {
    if (!/\.safetensors$/i.test(file?.name || '') || file.size < 8) return null;
    const prefix = await file.slice(0, 8).arrayBuffer();
    if (prefix.byteLength !== 8) return null;
    const length = new DataView(prefix).getBigUint64(0, true);
    if (length < 2n || length > 16n * 1024n * 1024n || length > BigInt(file.size - 8)) return null;
    let header;
    try {
        header = JSON.parse(await file.slice(8, 8 + Number(length)).text());
    } catch { return null; }
    const raw = header?.__metadata__?.lbo_recipe_json;
    if (typeof raw !== 'string') return null;
    if (new TextEncoder().encode(raw).byteLength > 256 * 1024)
        throw new Error(text(app, 'The saved recipe is too large.', '保存されたレシピが大きすぎます。'));
    return JSON.parse(raw);
}

export function recipeWorkflow(recipe, result, liteGraph) {
    if (result.issues.length) throw new Error(text(app, 'Unsupported recipe settings:', '復元できない設定：') + '\n' + result.issues.join('\n'));
    const graph = new liteGraph.LGraph();
    graph.id = crypto.randomUUID();
    try {
        const nodes = ['LoRABlendSources', 'LoRABlendOneSplit'].map(type => {
            const node = liteGraph.createNode(type);
            if (!node) throw new Error(text(app, 'LoRA Blend One nodes are unavailable. Restart ComfyUI and reload the browser.', 'LoRA Blend Oneノードを利用できません。ComfyUIを再起動し、ブラウザーを再読み込みしてください。'));
            graph.add(node);
            return node;
        });
        const [source, blend] = nodes;
        source.pos = [80, 100]; blend.pos = [660, 100];
        source.setSize([500, source.size[1]]); blend.setSize([430, blend.size[1]]);
        source.properties.lora_blend_advanced_open = result.sources.length > 2;
        blend.properties.lora_blend_advanced_open = true;
        if (!source.connect(0, blend, blend.inputs.findIndex(input => input.name === 'sources')))
            throw new Error(text(app, 'Could not connect the restored nodes.', '復元するノードを接続できませんでした。'));
        if (!applyRecipeToNodes(source, blend, result, true))
            throw new Error(text(app, 'The installed nodes do not support these recipe settings.', '現在のノードではこのレシピの設定を復元できません。'));
        const prefix = blend.widgets.find(widget => widget.name === 'filename_prefix');
        prefix.value = 'loras/' + result.file.replace(/\.safetensors$/i, '') + '_restored';
        blend.loraBlendRecipe.setImported(recipe, result);
        return structuredClone(graph.serialize());
    } finally {
        graph.clear();
    }
}

app.registerExtension({
    name: 'lora_blend_one.recipe_drop',
    setup() {
        const original = app.handleFile;
        let importing = Promise.resolve();
        app.handleFile = async function (file, ...args) {
            let recipe;
            try { recipe = await droppedRecipe(file); }
            catch (error) { report(error); return; }
            if (recipe === null) return original.call(this, file, ...args);
            const restore = async () => {
                try {
                    const response = await api.fetchApi('/lora_blend_one/recipe_from_metadata', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ filename: file.name, recipe }),
                    });
                    if (response.status === 404) throw new Error(text(app, 'Restart ComfyUI and reload the browser to enable recipe drops.', 'ドロップ復元を有効にするためComfyUIを再起動し、ブラウザーを再読み込みしてください。'));
                    const result = await response.json();
                    if (!response.ok) throw new Error(result.error || response.statusText);
                    const workflow = recipeWorkflow(recipe, result, LiteGraph);
                    // A fresh name prevents ComfyUI from treating this as a reload of an existing tab.
                    const name = `LoRA Blend - ${file.name.replace(/\.safetensors$/i, '').slice(0, 100)} - ${workflow.id}`;
                    const loaded = await app.loadGraphData(workflow, true, true, name);
                    if (loaded === false) throw new Error(text(app, 'The restored workflow could not be opened.', '復元ワークフローを開けませんでした。'));
                } catch (error) { report(error); }
            };
            importing = importing.then(restore, restore);
            return importing;
        };
    },
});

function report(error) {
    app.extensionManager.toast.add({severity: 'error',
        summary: text(app, 'Cannot restore blend recipe', 'ブレンドのレシピを復元できません'),
        detail: error.message, life: 10000});
}
