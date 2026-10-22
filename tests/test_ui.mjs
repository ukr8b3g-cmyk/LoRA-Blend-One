// Offline checks: node --test tests/test_ui.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { execFileSync } from 'node:child_process';

function uiSource(file) {
    return process.env.LBO_TEST_HEAD
        ? execFileSync('git', ['show', `HEAD:web/${file}`], { cwd: new URL('..', import.meta.url), encoding: 'utf8' })
        : fs.readFileSync(new URL(`../web/${file}`, import.meta.url), 'utf8');
}

test('custom UI follows ComfyUI locale without changing saved settings', () => {
    let locale;
    const listeners = new Set();
    const extensions = [];
    const settings = {
        getSettingValue: () => locale,
        addEventListener: (event, callback) => {
            assert.equal(event, 'Comfy.Locale.change');
            listeners.add(callback);
        },
        removeEventListener: (_, callback) => listeners.delete(callback),
    };
    const app = { ui: { settings }, graph: { setDirtyCanvas() {} },
        registerExtension: extension => extensions.push(extension) };
    const context = vm.createContext({ app, queueMicrotask });
    for (const file of ['i18n.js', 'accordion.js', 'model_status.js']) {
        const source = uiSource(file)
            .replace(/^import .*;\r?\n/gm, '').replace(/^export /gm, '');
        vm.runInContext(`{ ${source}\n${file === 'i18n.js' ? 'globalThis.text = text; globalThis.watchLocale = watchLocale;' : ''} }`, context);
    }
    class Node {
        constructor() {
            this.widgets = [{ name: 'output_rank', value: '16' },
                { name: 'seed', value: 0 }, { name: 'control_after_generate', value: 'fixed' }];
            this.properties = {};
            this.size = [300, 100];
            this.onNodeCreated();
        }
        addWidget(type, name, value, callback) {
            const widget = { type, name, value, callback };
            this.widgets.push(widget);
            return widget;
        }
        computeSize() { return [300, 100]; }
        setSize(size) { this.size = size; }
        serialize() { return this.widgets.map(widget => widget.value); }
    }
    for (const extension of extensions) extension.beforeRegisterNodeDef(Node, {
        name: 'LoRABlendOneSplit', input: { optional: { seed: ['INT', { advanced: true }] } },
    });
    const node = new Node();
    const header = node.widgets.find(widget => widget.type === 'button');
    const labels = [];
    const ctx = { save() {}, restore() {}, fillRect() {}, fillText: label => labels.push(label) };
    const draw = () => { node.onDrawForeground(ctx); return labels.at(-1); };
    assert.equal(header.name, '▶ Advanced');
    assert.equal(draw(), 'Model detection: run to detect');
    assert.equal(node.widgets.find(widget => widget.name === 'seed').hidden, true);
    node.onExecuted({ model_detection: ['MOSS-Image'] });
    assert.equal(draw(), 'Last detected model: MOSS-Image');
    locale = 'ja';
    for (const callback of listeners) callback();
    assert.equal(header.name, '▶ 詳細設定');
    assert.equal(draw(), '前回のモデル判定：MOSS-Image');
    header.callback();
    assert.equal(header.name, '▼ 詳細設定');
    assert.equal(node.widgets.find(widget => widget.name === 'seed').hidden, false);
    assert.equal(JSON.stringify(node.serialize()), JSON.stringify(['16', 0, 'fixed']));
    locale = 'en';
    for (const callback of listeners) callback();
    assert.equal(header.name, '▼ Advanced');
    assert.equal(draw(), 'Last detected model: MOSS-Image');
    locale = 'fr';
    assert.equal(draw(), 'Last detected model: MOSS-Image');
    node.onRemoved();
    assert.equal(listeners.size, 0);
});

test('store export, reload and legacy recovery preserve each setting by name', () => {
    const extensions = [];
    const app = { graph: { setDirtyCanvas() {} }, registerExtension: e => extensions.push(e) };
    const context = vm.createContext({ app, queueMicrotask });
    for (const file of ['i18n.js', 'accordion.js']) {
        const source = uiSource(file).replace(/^import .*;\r?\n/gm, '').replace(/^export /gm, '');
        vm.runInContext(`{ ${source}\n${file === 'i18n.js' ? 'globalThis.text = text; globalThis.watchLocale = watchLocale;' : ''} }`, context);
    }
    const required = { output_rank: [['Auto', '16'], { default: '16' }],
        filename_prefix: ['STRING', { default: 'blend' }] };
    const optional = {
        merge_method: [['Auto', 'Weighted'], { default: 'Auto' }],
        weight_mode: [['normalize', 'direct'], { default: 'normalize' }],
        compression_mode: [['auto', 'none'], { default: 'auto' }],
        custom_rank: ['INT', { default: 32, min: 1, max: 2048, advanced: true }],
        energy_target: ['FLOAT', { default: 0.999, min: 0.5, max: 1, advanced: true }],
        auto_rank_cap: ['INT', { default: 256, min: 8, max: 1024, advanced: true }],
        density: ['FLOAT', { default: 0.9, min: 0.01, max: 1, advanced: true }],
        sign_consensus: [['auto', 'on', 'off'], { default: 'auto', advanced: true }],
        della_half_width: ['FLOAT', { default: 0.05, min: 0, max: 0.49, advanced: true }],
        output_scale: ['FLOAT', { default: 1, min: -4, max: 4, advanced: true }],
        output_dtype: [['auto', 'fp32', 'fp16', 'bf16'], { default: 'auto' }],
    };
    const specs = { ...required, ...optional };
    class Node {
        constructor() {
            this.widgets = Object.entries(specs).map(([name, [, opts]]) => ({ name, value: opts.default }));
            this.properties = {};
            this.onNodeCreated();
        }
        addWidget(type, name, value, callback) {
            const widget = { type, name, value, callback }; this.widgets.push(widget); return widget;
        }
        // The modern graph serializer calls this method directly.
        serializeFromStoreState() {
            const entries = this.widgets.filter(w => w.serialize !== false).map(w => [w.name, w.value]);
            return { widgets_values: entries.map(([, v]) => v), widgets_values_named: Object.fromEntries(entries) };
        }
        serialize() { return this.serializeFromStoreState(); }
        configure(info) { this.widgets.forEach((w, index) => { w.value = info.widgets_values[index]; }); }
    }
    for (const extension of extensions) extension.beforeRegisterNodeDef(Node, {
        name: 'LoRABlendOneSplit', input: { required, optional },
    });
    const node = new Node();
    const expected = Object.fromEntries(Object.entries(specs).map(([name, [, opts]]) => [name, opts.default]));
    node.widgets.find(w => w.name === 'density').value = 0.73;
    node.widgets.find(w => w.name === 'output_dtype').value = 'bf16';
    expected.density = 0.73; expected.output_dtype = 'bf16';
    const exportData = node.serializeFromStoreState();
    assert.equal(JSON.stringify(exportData.widgets_values), JSON.stringify(Object.values(expected)));
    assert.equal(JSON.stringify(exportData.widgets_values_named), JSON.stringify(expected));
    const restored = new Node(); restored.configure(exportData);
    assert.equal(restored.widgets.find(w => w.name === 'output_dtype').value, 'bf16');
    assert.equal(restored.widgets.find(w => w.name === 'density').value, 0.73);
    assert.equal(JSON.stringify(restored.serializeFromStoreState()), JSON.stringify(exportData));
    // Reproduce the reported early export, including null from the misplaced "auto".
    const legacy = { widgets_values: ['16', 'blend', 'Auto', 'normalize', 'auto',
        0.9, null, 0.05, 1, 'auto', 32, 0.999, 256] };
    const repaired = new Node(); repaired.configure(legacy);
    assert.equal(repaired.widgets.find(w => w.name === 'energy_target').value, 0.999);
    assert.equal(repaired.widgets.find(w => w.name === 'custom_rank').value, 32);
    assert.equal(repaired.widgets.find(w => w.name === 'output_dtype').value, 'auto');
    assert.equal(repaired.widgets.find(w => w.name === 'sign_consensus').value, 'auto');
    const invalid = { widgets_values: Object.values(expected).map((v, i) => i === 6 ? null : v) };
    const unchanged = new Node(); unchanged.configure(invalid);
    assert.equal(unchanged.widgets.find(w => w.name === 'energy_target').value, null);
});

test('template keeps the supplied connections and reproducible defaults', () => {
    const workflow = JSON.parse(fs.readFileSync(new URL('../example_workflows/LoRA_Blend_One.json', import.meta.url)));
    assert.deepEqual(workflow.nodes.map(node => node.type).sort(), ['LoRABlendOneSplit', 'LoRABlendSources', 'PreviewAny']);
    assert.equal(workflow.links.length, 2);
    const values = workflow.nodes.find(node => node.type === 'LoRABlendOneSplit').widgets_values_named;
    assert.equal(values.seed, 0);
    assert.equal(values.control_after_generate, 'fixed');
    assert.equal(values.write_report, false);
    assert.equal(values.custom_rank, 32);
    assert.equal(values.output_dtype, 'auto');
    const english = JSON.parse(fs.readFileSync(new URL('../locales/en/nodeDefs.json', import.meta.url)));
    const japanese = JSON.parse(fs.readFileSync(new URL('../locales/ja/nodeDefs.json', import.meta.url)));
    for (const name of Object.keys(english)) {
        assert.deepEqual(Object.keys(english[name].inputs), Object.keys(japanese[name].inputs));
    }
});
