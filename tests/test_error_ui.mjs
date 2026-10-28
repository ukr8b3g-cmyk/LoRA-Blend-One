import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { loadRunTracker } from './helpers/run_tracker.mjs';

function setup(queueRequest = async (number, data) => ({prompt_id: data.testPromptId})) {
    let locale = 'en';
    const extensions = [], listeners = new Map(), localeListeners = new Set();
    const api = {
        queuePrompt: queueRequest,
        addEventListener(name, handler) {
            if (!listeners.has(name)) listeners.set(name, new Set());
            listeners.get(name).add(handler);
        },
        removeEventListener(name, handler) { listeners.get(name)?.delete(handler); },
    };
    const graph = { id: 'workflow-a', nodes: new Map(), links: {},
        getNodeById(id) { return this.nodes.get(String(id)); }, setDirtyCanvas() {},
    };
    const app = { graph, rootGraph: graph, extensionManager: { lastNodeErrors: null },
        ui: { settings: {
            getSettingValue: () => locale,
            addEventListener: (_, handler) => localeListeners.add(handler),
            removeEventListener: (_, handler) => localeListeners.delete(handler),
        } }, registerExtension: extension => extensions.push(extension),
    };
    const context = vm.createContext({ app, api });
    loadRunTracker(context);
    for (const file of ['i18n.js', 'model_status.js']) {
        const source = fs.readFileSync(new URL(`../web/${file}`, import.meta.url), 'utf8')
            .replace(/^import .*;\r?\n/gm, '').replace(/^export /gm, '');
        vm.runInContext(`{ ${source}\n${file === 'i18n.js' ? 'Object.assign(globalThis, {text, watchLocale});' : ''} }`, context);
    }
    function node(type, id, owner = graph) {
        class Node {
            constructor() {
                this.id = id; this.type = type; this.graph = owner;
                this.widgets = [{ name: 'lora_1', value: 'moss/a.safetensors' },
                    { name: 'weight_1', value: 20 }, { name: 'enabled_1', value: true }];
                this.size = [310, 100]; this.baseHeight = 100;
                owner.nodes.set(String(id), this);
                this.onNodeCreated?.();
            }
            computeSize() { return [this.size[0], this.baseHeight]; }
            setSize(value) { this.size = value; }
        }
        for (const extension of extensions) extension.beforeRegisterNodeDef(Node, { name: type });
        return new Node();
    }
    function send(name, detail) { for (const handler of listeners.get(name) || []) handler({ detail }); }
    async function queue(nodes, id) {
        await submit(nodes, id);
        send('execution_start', { prompt_id: id });
    }
    function draw(node) {
        const labels = [], boxes = [];
        const ctx = { save() {}, restore() {}, fillRect(...args) { boxes.push({ color: this.fillStyle, args }); },
            measureText(value) { return { width: Array.from(value).reduce((sum, char) => sum + (char.codePointAt(0) > 255 ? 12 : 6), 0) }; },
            fillText(value, ...position) { labels.push({ value, position, width: this.measureText(value).width }); },
        };
        node.onDrawForeground(ctx);
        return { labels, boxes, text: labels.map(label => label.value).join('') };
    }
    async function submit(nodes, id) {
        for (const value of nodes) value.widgets[0].beforeQueued();
        return api.queuePrompt(0, {workflow: {id: nodes[0].graph.id},
            output: Object.fromEntries(nodes.map(value => [value.id, {class_type: value.type}])), testPromptId: id});
    }
    return { app, api, graph, node, send, queue, submit, draw, listeners, localeListeners,
        setLocale(value) { locale = value; for (const handler of localeListeners) handler(); } };
}

function structured(name = 'DMD2/dmd2_sdxl_4step_lora.safetensors') {
    return { code: 'incompatible_family', sources: [{ slot: 7, name }], detail: 'MOSS/SDXL mismatch',
        reason: { en: 'This LoRA targets another model family.', ja: 'このLoRAは別のモデル向けです。' },
        hint: { en: 'Turn LoRA 7 OFF or choose a compatible MOSS-Image LoRA.', ja: 'LoRA 7をOFFにするか、互換性のあるMOSS-Image用LoRAを選んでください。' } };
}

for (const type of ['LoRABlendOne', 'LoRABlendOneSplit', 'LoRABlendSources']) {
    test(`${type}: two accepted jobs display their own error without workflow metadata`, async () => {
        const env = setup(), node = env.node(type, 5);
        await env.submit([node], 'first'); await env.submit([node], 'second');
        for (const id of ['first', 'second']) {
            env.send('execution_start', {prompt_id: id});
            env.send('lora_blend_one.error', {node_id: '5', prompt_id: id, error: structured(`${id}.safetensors`)});
            env.send('execution_error', {node_id: '5', node_type: type, prompt_id: id, exception_message: 'fallback'});
            assert.equal(node.loraBlendFailure.sources[0].name, `${id}.safetensors`);
            assert.equal(node.loraBlendFailure.code, 'incompatible_family');
        }
        node.onRemoved();
    });
}

test('structured failures clear stale success and wrap full English/Japanese text', async () => {
    const env = setup(), node = env.node('LoRABlendOneSplit', 5);
    await env.queue([node], 'initial');
    env.send('executed', {node: '5', prompt_id: 'initial', output: {model_detection: ['MOSS-Image']}});
    env.send('execution_success', {prompt_id: 'initial'});
    assert.match(env.draw(node).text, /Last detected model: MOSS-Image/);
    await env.queue([node], 'run-1');
    assert.equal(node.loraBlendDetectedModel, null);
    const name = `moss/${'長いファイル名_'.repeat(12)}source.safetensors`;
    env.send('lora_blend_one.error', { node_id: '5', prompt_id: 'run-1', error: structured(name) });
    let image = env.draw(node);
    assert.match(image.text, /Blend stopped/);
    assert.ok(image.text.includes(name), 'the entire filename must remain readable');
    assert.match(image.text, /another model family/);
    assert.match(image.text, /Turn LoRA 7 OFF/);
    assert.equal(image.boxes[0].color, '#4b2025');
    assert.ok(image.labels.every(line => line.width <= 270), 'long names must wrap without squeezing text');
    assert.ok(node.size[1] > 126);
    env.setLocale('ja');
    image = env.draw(node);
    assert.match(image.text, /ブレンド停止/);
    assert.match(image.text, /別のモデル向け/);
    assert.match(image.text, /LoRA 7をOFF/);
    assert.ok(image.text.includes(name));
    assert.ok(image.labels.every(line => line.width <= 270));
    env.send('execution_error', { node_id: '5', node_type: node.type, prompt_id: 'run-1', exception_message: 'generic fallback' });
    assert.equal(node.loraBlendFailure.code, 'incompatible_family', 'generic event must not overwrite structured details');
    await env.queue([node], 'run-2');
    assert.equal(node.loraBlendFailure, null);
    assert.equal(node.size[1], 126);
    env.send('lora_blend_one.error', { node_id: '5', prompt_id: 'run-1', error: structured() });
    assert.equal(node.loraBlendFailure, null, 'late events from a previous run are ignored');
    env.send('executed', {node: '5', prompt_id: 'run-2', output: {model_detection: ['MOSS-Image']}});
    env.send('execution_success', { prompt_id: 'run-2' });
    assert.match(env.draw(node).text, /前回のモデル判定：MOSS-Image/);
});

test('standard runtime and native validation failures identify files when available', async () => {
    const env = setup(), node = env.node('LoRABlendSources', 4);
    assert.equal(node.size[1], 100, 'Sources has no empty success/status strip');
    assert.equal(env.draw(node).labels.length, 0);
    await env.queue([node], 'run-1');
    env.send('execution_error', { node_id: '4', node_type: node.type, prompt_id: 'run-1',
        exception_message: 'Cannot read the adapter.', current_inputs: { lora_1: ['moss/broken.safetensors'] } });
    assert.match(env.draw(node).text, /moss\/broken.safetensors/);
    assert.match(env.draw(node).text, /Cannot read the adapter/);
    await env.queue([node], 'retry');
    assert.equal(node.loraBlendFailure, null);
    assert.equal(node.size[1], 100);
    const missing = 'LoRA 2: selected file is unavailable: old/missing.safetensors. Choose an installed LoRA or turn this slot off.';
    env.app.extensionManager.lastNodeErrors = { 4: { errors: ['lora_1', 'weight_1'].map(name => ({
        type: 'custom_validation_failed', message: 'Custom validation failed', details: `${name} - ${missing}`,
        extra_info: { input_name: name },
    })) } };
    env.setLocale('ja');
    let image = env.draw(node);
    assert.match(image.text, /LoRA 2: old\/missing.safetensors/);
    assert.match(image.text, /ファイルが見つかりません/);
    assert.match(image.text, /OFFにして再実行/);
    assert.equal(node.loraBlendFailure.sources.length, 1);
    assert.equal(node.loraBlendFailure.sources[0].name, 'old/missing.safetensors');
    assert.equal(node.loraBlendFailure.detail, missing, 'repeated validator verdicts collapse to one representative message');
    assert.ok(node.size[1] < 500, 'repeated field prefixes must not create a giant error block');
    await env.queue([node], 'run-2');
    assert.equal(node.loraBlendFailure, null);
    env.draw(node);
    assert.equal(node.loraBlendFailure, null, 'the previous validation object must not reappear on retry');
    env.app.extensionManager.lastNodeErrors = { 4: { errors: [{ message: 'Invalid number', details: 'energy_target: invalid',
        extra_info: { input_name: 'energy_target' } }] } };
    image = env.draw(node);
    assert.match(image.text, /特定できません（詳細を確認）/);
    assert.match(image.text, /energy_target の入力値が無効/);
    env.app.extensionManager.lastNodeErrors = { 4: { errors: ['lora_1', 'weight_1'].map(name => ({
        type: 'custom_validation_failed', message: 'Custom validation failed',
        details: `${name} - DELLA requires 0 < density-half_width <= density+half_width <= 1.`,
        extra_info: { input_name: name },
    })) } };
    image = env.draw(node);
    assert.equal(node.loraBlendFailure.sources.length, 0, 'a node-wide settings verdict must not blame lora_1');
    assert.match(image.text, /入力内容の検証に失敗/);
    assert.equal(image.text.split('DELLA requires').length - 1, 1);
});

test('node/prompt/graph routing and listener cleanup keep other nodes untouched', async () => {
    const env = setup(), first = env.node('LoRABlendOneSplit', 5), other = env.node('LoRABlendOneSplit', 6);
    await env.queue([first, other], 'run-1');
    env.send('lora_blend_one.error', { node_id: '5', prompt_id: 'other-run', error: structured() });
    assert.equal(first.loraBlendFailure, null);
    env.send('lora_blend_one.error', { node_id: '5', prompt_id: 'run-1', error: structured() });
    assert.ok(first.loraBlendFailure);
    assert.equal(other.loraBlendFailure, null);
    const secondGraph = { id: 'workflow-b', nodes: new Map(), getNodeById(id) { return this.nodes.get(String(id)); }, setDirtyCanvas() {} };
    env.app.graph = env.app.rootGraph = secondGraph;
    const duplicateId = env.node('LoRABlendOneSplit', 5, secondGraph);
    env.send('lora_blend_one.error', { node_id: '5', prompt_id: 'run-1', error: structured() });
    assert.equal(duplicateId.loraBlendFailure, null, 'a node with the same ID in another workflow must not receive the old run');
    for (const node of [first, other, duplicateId]) node.onRemoved();
    assert.ok([...env.listeners.values()].every(handlers => handlers.size === 0));
    assert.equal(env.localeListeners.size, 0);
    env.app.graph = env.app.rootGraph = env.graph;
    first.onAdded(env.graph);
    first.onAdded(env.graph);
    assert.ok([...env.listeners.values()].every(handlers => handlers.size === 1), 'restoration subscribes once, even if onAdded repeats');
    assert.equal(env.localeListeners.size, 1);
    await env.queue([first], 'restored-run');
    env.send('lora_blend_one.error', { node_id: '5', prompt_id: 'restored-run', error: structured() });
    assert.ok(first.loraBlendFailure, 'an undone removal must receive failures again');
    first.onRemoved();
    assert.ok([...env.listeners.values()].every(handlers => handlers.size === 0));
    assert.equal(env.localeListeners.size, 0);
});

test('unexpected failures show raw details in both locales without duplicating the reason', async () => {
    const env = setup(), node = env.node('LoRABlendOneSplit', 5);
    await env.queue([node], 'unknown-run');
    const detail = 'PermissionError: the output file is locked by another process.';
    const error = { code: 'unexpected_error', sources: [], detail,
        reason: { en: 'An unexpected error stopped blending.', ja: '予期しないエラーでブレンドを停止しました。' },
        hint: { en: 'Check the details below.', ja: '下の詳細を確認してください。' } };
    env.send('lora_blend_one.error', { node_id: '5', prompt_id: 'unknown-run', error });
    let image = env.draw(node);
    assert.match(image.text, /Not identified \(see details\)/);
    assert.match(image.text, /Details: PermissionError/);
    assert.ok(image.text.includes(detail));
    env.setLocale('ja');
    image = env.draw(node);
    assert.match(image.text, /特定できません（詳細を確認）/);
    assert.match(image.text, /詳細: PermissionError/);
    assert.ok(image.text.includes(detail));
    error.reason.ja += `\n${detail}`;
    env.send('lora_blend_one.error', { node_id: '5', prompt_id: 'unknown-run', error });
    image = env.draw(node);
    assert.equal(image.text.split(detail).length - 1, 1, 'details already in the reason must not be repeated');
});

test('accepted IDs isolate workflows and late events regardless of metadata', async () => {
    const env = setup(), node = env.node('LoRABlendOneSplit', 5);
    await env.queue([node], 'a-1');
    await env.submit([node], 'a-2');
    env.send('lora_blend_one.error', {node_id:'5', prompt_id:'a-1', error:structured()});
    const firstFailure = node.loraBlendFailure;
    const otherGraph = {id:'workflow-b', nodes:new Map(), getNodeById(id){return this.nodes.get(String(id));}, setDirtyCanvas(){}};
    const other = env.node('LoRABlendOneSplit', 5, otherGraph);
    env.app.graph = env.app.rootGraph = otherGraph;
    await env.submit([other], 'b-1');
    env.send('execution_start', {prompt_id:'b-1', workflow_id:'workflow-b'});
    env.send('executed', {node:'5', prompt_id:'b-1', output:{model_detection:['Standard LoRA']}});
    assert.equal(node.loraBlendFailure, firstFailure);
    assert.equal(other.loraBlendDetectedModel, 'Standard LoRA');
    env.send('execution_start', {prompt_id:'a-2', workflow_id:'workflow-a'});
    assert.equal(node.loraBlendFailure, null);
    env.send('lora_blend_one.error', {node_id:'5', prompt_id:'a-2', workflow_id:'workflow-b', error:structured('wrong')});
    assert.equal(node.loraBlendFailure, null);
    env.send('lora_blend_one.error', {node_id:'5', prompt_id:'a-2', workflow_id:'workflow-a', error:structured('second')});
    const secondFailure = node.loraBlendFailure;
    env.send('execution_success', {prompt_id:'a-1'});
    env.send('executed', {node:'5', prompt_id:'a-1', output:{model_detection:['stale']}});
    assert.equal(node.loraBlendFailure, secondFailure);
    assert.equal(other.loraBlendDetectedModel, 'Standard LoRA');
    env.send('execution_error', {node_id:'5', prompt_id:'a-2', exception_message:'fallback'});
    env.send('lora_blend_one.error', {node_id:'5', prompt_id:'a-2', error:structured('stale')});
    assert.equal(node.loraBlendFailure, secondFailure);
    node.onRemoved(); other.onRemoved();
});

test('workflow metadata cannot claim a foreign prompt, or a graph whose identity changed', async () => {
    const env = setup(), node = env.node('LoRABlendSources', 4);
    await env.queue([node], 'own');
    env.send('execution_start', {prompt_id:'foreign', workflow_id:'workflow-a'});
    env.send('lora_blend_one.error', {node_id:'4', prompt_id:'foreign', workflow_id:'workflow-a', error:structured()});
    assert.equal(node.loraBlendFailure, null);
    delete env.graph.id;
    env.send('lora_blend_one.error', {node_id:'4', prompt_id:'own', error:structured()});
    assert.equal(node.loraBlendFailure, null);
    node.onRemoved();
});

test('an error completed before queue acknowledgement retains structured details and cannot replay', async () => {
    let accept;
    const env = setup(() => new Promise(resolve => accept = resolve)), node = env.node('LoRABlendOneSplit', 5);
    const queued = env.submit([node], 'fast');
    env.send('execution_start', {prompt_id:'fast'});
    env.send('lora_blend_one.error', {prompt_id:'foreign', node_id:'5', error:structured('foreign')});
    env.send('lora_blend_one.error', {prompt_id:'fast', node_id:'5', error:structured('fast.safetensors')});
    env.send('execution_error', {prompt_id:'fast', node_id:'5', exception_message:'generic'});
    assert.equal(node.loraBlendFailure, null);
    accept({prompt_id:'fast'}); await queued;
    assert.equal(node.loraBlendFailure.sources[0].name, 'fast.safetensors');
    assert.equal(node.loraBlendFailure.code, 'incompatible_family');
    const saved = node.loraBlendFailure;
    env.send('lora_blend_one.error', {prompt_id:'fast', node_id:'5', error:structured('stale')});
    env.send('execution_success', {prompt_id:'fast'});
    assert.equal(node.loraBlendFailure, saved);
    node.onRemoved();
});

test('rejected, interrupted, cancelled and removed jobs cannot change the next error', async () => {
    const rejected = new Error('rejected');
    const env = setup(async (number, data) => {
        if (data.testPromptId === 'rejected') throw rejected;
        return {prompt_id:data.testPromptId};
    }), node = env.node('LoRABlendOneSplit', 5);
    await assert.rejects(env.submit([node], 'rejected'), error => error === rejected);
    await env.submit([node], 'interrupted');
    env.send('execution_interrupted', {prompt_id:'interrupted'});
    await env.submit([node], 'cancelled');
    env.send('status', {exec_info:{queue_remaining:0}});
    await env.submit([node], 'removed'); node.onRemoved(); node.onAdded();
    await env.queue([node], 'next');
    env.send('lora_blend_one.error', {prompt_id:'next', node_id:'5', error:structured('next')});
    const saved = node.loraBlendFailure;
    for (const prompt_id of ['rejected','interrupted','cancelled','removed','foreign']) {
        env.send('execution_start', {prompt_id, workflow_id:'workflow-a'});
        env.send('lora_blend_one.error', {prompt_id, node_id:'5', error:structured('stale')});
        env.send('executed', {prompt_id, node:'5', output:{model_detection:['stale']}});
        assert.equal(node.loraBlendFailure, saved);
    }
    node.onRemoved();
    assert.ok([...env.listeners.values()].every(handlers => handlers.size === 0));
    assert.equal(env.localeListeners.size, 0);
});

test('a delayed acknowledgement of a completed job cannot replace a newer error', async () => {
    const accept = new Map();
    const env = setup((_, data) => new Promise(resolve => accept.set(data.testPromptId, resolve)));
    const node = env.node('LoRABlendOneSplit', 5);
    const first = env.submit([node], 'first'), second = env.submit([node], 'second');
    for (const prompt_id of ['first', 'second']) {
        env.send('execution_start', {prompt_id});
        env.send('lora_blend_one.error', {prompt_id, node_id:'5', error:structured(prompt_id)});
        env.send('execution_error', {prompt_id, node_id:'5', exception_message:'fallback'});
    }
    accept.get('second')({prompt_id:'second'}); await second;
    accept.get('first')({prompt_id:'first'}); await first;
    assert.equal(node.loraBlendFailure.sources[0].name, 'second');
    node.onRemoved();
});
