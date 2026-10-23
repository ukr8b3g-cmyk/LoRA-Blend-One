import { app } from '../../scripts/app.js';
import { text, watchLocale } from './i18n.js';
import { installSourceToggles } from './source_toggles.js';
import { installRecipeControls } from './recipe.js';
import { installPowerLoaderControls } from './power_loader.js';

const NODE_NAMES = new Set(['LoRABlendSources', 'LoRABlendOneSplit']);
const STATE_KEY = 'lora_blend_advanced_open';

// Early exports used this visual order but labelled the values with schema names.
const LEGACY_VISIBLE = new Set(['output_rank', 'filename_prefix', 'merge_method',
    'weight_mode', 'compression_mode', 'density', 'sign_consensus',
    'della_half_width', 'output_scale', 'output_dtype']);

function restoreWidgetOrder(info, original, display, nodeData) {
    const widgets = original.filter(widget => widget.serialize !== false);
    const values = info?.widgets_values;
    if (!Array.isArray(values) || values.length !== widgets.length) return info;
    const specs = { ...nodeData.input?.required, ...nodeData.input?.optional };
    const named = Object.fromEntries(widgets.map((widget, index) => [widget.name, values[index]]));
    const valid = entries => Object.entries(entries).every(([name, value]) => {
        const spec = specs[name];
        if (!spec) return true;
        const [type, options = {}] = spec;
        if (Array.isArray(type)) return type.includes(value);
        if (type === 'INT' || type === 'FLOAT') return typeof value === 'number'
            && Number.isFinite(value) && (type !== 'INT' || Number.isInteger(value))
            && value >= (options.min ?? -Infinity) && value <= (options.max ?? Infinity);
        if (type === 'BOOLEAN') return typeof value === 'boolean';
        if (type === 'STRING') return typeof value === 'string';
        return true;
    });
    if (valid(named)) return info;
    // Only repair a complete, unambiguous permutation. Never reset arbitrary settings.
    const legacy = [...widgets.filter(widget => LEGACY_VISIBLE.has(widget.name)),
        ...widgets.filter(widget => !LEGACY_VISIBLE.has(widget.name))];
    const candidates = [legacy, display.filter(widget => widget.serialize !== false)]
        .map(order => Object.fromEntries(order.map((widget, index) => [widget.name, values[index]])));
    for (const candidate of candidates) {
        // Some old exports already converted the misplaced string "auto" to null.
        if (candidate.sign_consensus === null && typeof named.output_dtype === 'number'
            && !Number.isInteger(named.custom_rank) && named.auto_rank_cap < 1)
            candidate.sign_consensus = 'auto';
    }
    const matches = candidates.filter(valid);
    if (!matches.length || matches.some(candidate => widgets.some(widget => candidate[widget.name] !== matches[0][widget.name]))) return info;
    const repaired = matches[0];
    return { ...info, widgets_values: widgets.map(widget => repaired[widget.name]),
        widgets_values_named: { ...info.widgets_values_named, ...repaired } };
}

function installAccordion(node, nodeData) {
    if (node.loraBlendAccordion || !node.addWidget) return;
    if (nodeData.name === 'LoRABlendOneSplit') installRecipeControls(node);
    if (nodeData.name === 'LoRABlendSources') installPowerLoaderControls(node);
    const advancedNames = new Set(Object.entries(nodeData.input?.optional || {})
        .filter(([, spec]) => spec[1]?.advanced === true)
        .map(([name]) => name));
    if (advancedNames.has('seed')) {
        advancedNames.add('control_after_generate');
        advancedNames.add('control_before_generate');
    }
    const original = [...(node.widgets || [])];
    const advanced = [...original.filter(widget => widget.loraBlendAdvanced),
        ...original.filter(widget => advancedNames.has(widget.name))];
    if (!advanced.length) return;
    const regular = original.filter(widget => !advanced.includes(widget));
    const presentation = new Map(advanced.map(widget => [widget, {
        computeSize: widget.computeSize,
        elements: [...new Set([widget.element, widget.inputEl].filter(Boolean))]
            .map(element => [element, element.style?.display || '']),
    }]));
    const header = node.addWidget('button', `▶ ${text(app, 'Advanced', '詳細設定')}`, null, () => {
        node.graph?.beforeChange?.();
        node.properties ??= {};
        node.properties[STATE_KEY] = !node.properties[STATE_KEY];
        refresh();
        node.graph?.afterChange?.();
    }, { serialize: false });
    header.serialize = false;
    const display = [...regular, header, ...advanced];
    node.widgets = display;

    function refresh() {
        const open = node.properties?.[STATE_KEY] === true;
        void node.loraBlendRecipe?.refresh(open);
        node.loraBlendPowerLoader?.refresh();
        const krea2 = original.find(widget => widget.name === 'merge_method')?.value === 'Krea2 Gate-Aware';
        header.name = `${open ? '▼' : '▶'} ${text(app, 'Advanced', '詳細設定')}`;
        for (const widget of advanced) {
            const saved = presentation.get(widget);
            const visible = open && (!widget.name.startsWith('krea2_') || krea2);
            widget.hidden = !visible;
            widget.computeSize = visible ? saved.computeSize : () => [0, -4];
            for (const [element, priorDisplay] of saved.elements)
                if (element.style) element.style.display = visible ? priorDisplay : 'none';
        }
        if (node.computeSize && node.setSize) {
            const size = node.computeSize();
            node.setSize([Math.max(node.size?.[0] || 0, size[0]), size[1]]);
        }
        (node.graph || app.graph)?.setDirtyCanvas?.(true, true);
    }

    const methodWidget = original.find(widget => widget.name === 'merge_method');
    if (methodWidget) {
        const callback = methodWidget.callback;
        methodWidget.callback = function (...args) {
            const result = callback?.apply(this, args);
            queueMicrotask(refresh);
            return result;
        };
    }

    // Current LiteGraph also serializes directly from its store, bypassing serialize().
    // Both paths must see the schema order, including when the graph is exported.
    for (const method of ['serialize', 'serializeFromStoreState', 'configure']) {
        const previous = node[method];
        if (typeof previous !== 'function') continue;
        node[method] = function (...args) {
            this.widgets = original;
            try {
                if (method === 'configure') args[0] = restoreWidgetOrder(args[0], original, display, nodeData);
                return previous.apply(this, args);
            }
            finally {
                this.widgets = display;
                if (method === 'configure') queueMicrotask(refresh);
            }
        };
    }
    node.loraBlendAccordion = { refresh };
    watchLocale(app, node, refresh);
    refresh();
}

app.registerExtension({
    name: 'lora_blend_one.accordion',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name)) return;
        const previous = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = previous?.apply(this, args);
            if (nodeData.name === 'LoRABlendSources') installSourceToggles(this);
            installAccordion(this, nodeData);
            return result;
        };
    },
});
