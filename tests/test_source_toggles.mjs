import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

function setup() {
    let locale = 'en';
    const extensions = [], menus = [], listeners = new Set();
    const app = {
        canvas: { ds: { scale: 1 }, graph_mouse: [25, 45] },
        graph: { setDirtyCanvas() {} },
        ui: { settings: {
            getSettingValue: () => locale,
            addEventListener: (_, callback) => listeners.add(callback),
            removeEventListener: (_, callback) => listeners.delete(callback),
        } },
        registerExtension: extension => extensions.push(extension),
    };
    const context = vm.createContext({ app, queueMicrotask,
        LiteGraph: { ContextMenu: class { constructor(values, options) { menus.push({ values, options }); } } },
    });
    for (const file of ['i18n.js', 'source_toggles.js', 'power_loader.js', 'accordion.js']) {
        const source = fs.readFileSync(new URL(`../web/${file}`, import.meta.url), 'utf8')
            .replace(/^import .*;\r?\n/gm, '').replace(/^export /gm, '');
        const exports = file === 'i18n.js' ? 'Object.assign(globalThis, {text, watchLocale});'
            : file === 'source_toggles.js' ? 'globalThis.installSourceToggles = installSourceToggles;' : file === 'power_loader.js' ? 'globalThis.installPowerLoaderControls = installPowerLoaderControls;' : '';
        vm.runInContext(`{ ${source}\n${exports} }`, context);
    }
    const required = {}, optional = {};
    for (let slot = 1; slot <= 8; slot++) {
        const group = slot <= 2 ? required : optional;
        const advanced = slot > 2 ? { advanced: true } : {};
        group[`lora_${slot}`] = [['(none)', 'moss/A.safetensors', 'moss/B.safetensors'], { default: 'moss/A.safetensors', ...advanced }];
        group[`weight_${slot}`] = ['FLOAT', { default: slot * 10, ...advanced }];
    }
    for (let slot = 1; slot <= 8; slot++) optional[`enabled_${slot}`] = ['BOOLEAN', { default: true }];
    class Node {
        constructor() {
            this.widgets = Object.entries({ ...required, ...optional }).map(([name, [type, options]]) => ({
                name, type: Array.isArray(type) ? 'combo' : typeof options.default,
                value: options.default, options: { ...options, values: Array.isArray(type) ? type : undefined },
            }));
            this.originalWidgets = [...this.widgets];
            this.properties = {};
            this.size = [340, 100];
            this.pos = [0, 0];
            this.graph = { beforeChange() {}, afterChange() {}, incrementVersion() {}, setDirtyCanvas() {} };
            this.onNodeCreated();
        }
        addWidget(type, name, value, callback) {
            const widget = { type, name, value, callback };
            this.widgets.push(widget); return widget;
        }
        computeSize() {
            return [340, 40 + this.widgets.reduce((height, widget) => height + (widget.hidden ? 0 : 24), 0)];
        }
        setSize(value) { this.size = value; }
        serializeFromStoreState() {
            const serialized = this.widgets.filter(widget => widget.serialize !== false);
            return { widgets_values: serialized.map(widget => widget.value),
                widgets_values_named: Object.fromEntries(serialized.map(widget => [widget.name, widget.value])) };
        }
        serialize() { return this.serializeFromStoreState(); }
        configure(info) {
            info.widgets_values.forEach((value, index) => { this.widgets[index].value = value; });
            this.onConfigure?.(info);
        }
    }
    for (const extension of extensions) extension.beforeRegisterNodeDef(Node, {
        name: 'LoRABlendSources', input: { required, optional },
    });
    return { Node, app, menus, listeners, setLocale(value) { locale = value; for (const listener of listeners) listener(); } };
}

test('each source keeps two rows with an embedded switch and unchanged native combo binding', () => {
    const { Node, app, menus, setLocale } = setup();
    const node = new Node();
    const file = node.widgets.find(widget => widget.name === 'lora_1');
    const enabled = node.widgets.find(widget => widget.name === 'enabled_1');
    const weight = node.widgets.find(widget => widget.name === 'weight_1');
    assert.equal(file, node.originalWidgets[0]);
    assert.equal(file.type, 'combo');
    assert.deepEqual(Array.from(node.widgets.filter(widget => !widget.hidden), widget => widget.name),
        ['lora_1', 'weight_1', 'lora_2', 'weight_2', '▶ Advanced']);
    const height = node.size[1];
    const pointer = { eDown: { type: 'pointerdown', button: 0 } };
    assert.equal(file.onPointerDown(pointer, node, app.canvas), true);
    assert.equal(enabled.value, false);
    assert.equal(file.value, 'moss/A.safetensors');
    assert.equal(weight.value, 10);
    file.mouse({ type: 'pointerup', button: 0 }, [25, 45]);
    assert.equal(enabled.value, false, 'release must not toggle twice');
    assert.equal(node.size[1], height, 'switch must not add a row');
    assert.equal(menus.length, 0, 'switch must not open the file menu');
    file.mouse({ type: 'pointerdown', button: 0 }, [150, 45]);
    assert.deepEqual(menus[0].values, ['(none)', 'moss/A.safetensors', 'moss/B.safetensors']);
    menus[0].options.callback('moss/B.safetensors');
    assert.equal(file.value, 'moss/B.safetensors');
    assert.equal(enabled.value, false, 'choosing a file must not enable it implicitly');
    assert.match(file.tooltip, /moss\/B.safetensors/);
    setLocale('ja');
    assert.match(file.tooltip, /有効・無効/);
    const labels = [];
    const ctx = { save() {}, restore() {}, beginPath() {}, roundRect() {}, fill() {}, stroke() {}, arc() {},
        measureText: value => ({ width: value.length * 6 }), fillText: value => labels.push(value) };
    file.draw(ctx, node, 340, 40, 20);
    assert.match(labels[0], /LoRA 1: B.safetensors/);
    node.widgets.find(widget => widget.type === 'button').callback();
    assert.equal(node.widgets.filter(widget => !widget.hidden).length, 17, 'eight file/weight pairs plus Advanced');
    for (let slot = 1; slot <= 8; slot++) {
        assert.equal(node.widgets.find(widget => widget.name === `enabled_${slot}`).hidden, true);
    }
    node.onRemoved();
});

test('switch states round-trip without shifting old file/weight values', async () => {
    const { Node } = setup();
    const node = new Node();
    node.widgets.find(widget => widget.name === 'lora_7').value = 'moss/B.safetensors';
    const enabled7 = node.widgets.find(widget => widget.name === 'enabled_7');
    enabled7.value = false;
    const values = node.serializeFromStoreState();
    assert.deepEqual(Array.from(values.widgets_values.slice(0, 4)), ['moss/A.safetensors', 10, 'moss/A.safetensors', 20]);
    assert.equal(values.widgets_values[22], false);
    const restored = new Node(); restored.configure(values);
    await new Promise(resolve => queueMicrotask(resolve));
    assert.equal(restored.widgets.find(widget => widget.name === 'enabled_7').value, false);
    assert.match(restored.widgets.find(widget => widget.name === 'lora_7').tooltip, /moss\/B.safetensors/);
    assert.equal(restored.widgets.find(widget => widget.name === 'weight_7').value, 70);
    assert.deepEqual(restored.serializeFromStoreState(), values);
    const old = new Node(); old.configure({ widgets_values: values.widgets_values.slice(0, 16) });
    await new Promise(resolve => queueMicrotask(resolve));
    assert.equal(old.widgets.find(widget => widget.name === 'weight_8').value, 80);
    for (let slot = 1; slot <= 8; slot++) assert.equal(old.widgets.find(widget => widget.name === `enabled_${slot}`).value, true);
});
