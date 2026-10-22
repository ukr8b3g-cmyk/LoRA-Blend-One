import { app } from '../../scripts/app.js';

const NODE_NAMES = new Set(['LoRABlendSources', 'LoRABlendOneSplit']);
const STATE_KEY = 'lora_blend_advanced_open';

function installAccordion(node, nodeData) {
    if (node.loraBlendAccordion || !node.addWidget) return;
    const advancedNames = new Set(Object.entries(nodeData.input?.optional || {})
        .filter(([, spec]) => spec[1]?.advanced === true)
        .map(([name]) => name));
    if (advancedNames.has('seed')) {
        advancedNames.add('control_after_generate');
        advancedNames.add('control_before_generate');
    }
    const original = [...(node.widgets || [])];
    const advanced = original.filter(widget => advancedNames.has(widget.name));
    if (!advanced.length) return;
    const regular = original.filter(widget => !advancedNames.has(widget.name));
    const presentation = new Map(advanced.map(widget => [widget, {
        computeSize: widget.computeSize,
        elements: [...new Set([widget.element, widget.inputEl].filter(Boolean))]
            .map(element => [element, element.style?.display || '']),
    }]));
    const header = node.addWidget('button', '▶ 詳細設定 / Advanced', null, () => {
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
        const krea2 = original.find(widget => widget.name === 'merge_method')?.value === 'Krea2 Gate-Aware';
        header.name = `${open ? '▼' : '▶'} 詳細設定 / Advanced`;
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

    for (const method of ['serialize', 'configure']) {
        const previous = node[method];
        if (typeof previous !== 'function') continue;
        node[method] = function (...args) {
            this.widgets = original;
            try { return previous.apply(this, args); }
            finally {
                this.widgets = display;
                if (method === 'configure') queueMicrotask(refresh);
            }
        };
    }
    node.loraBlendAccordion = { refresh };
    refresh();
}

app.registerExtension({
    name: 'lora_blend_one.accordion',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name)) return;
        const previous = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = previous?.apply(this, args);
            installAccordion(this, nodeData);
            return result;
        };
    },
});
