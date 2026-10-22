import { app } from '../../scripts/app.js';
import { text, watchLocale } from './i18n.js';

const SOURCE_NODES = new Set(['LoRABlendSources', 'LoRABlendOne']);

function fitLabel(ctx, value, width) {
    if (ctx.measureText(value).width <= width) return value;
    let start = 0, end = value.length;
    while (start < end) {
        const middle = Math.ceil((start + end) / 2);
        if (ctx.measureText(`${value.slice(0, middle)}…`).width <= width) start = middle;
        else end = middle - 1;
    }
    return `${value.slice(0, start)}…`;
}

export function installSourceToggles(node) {
    if (node.loraBlendSourceToggles) return;
    const rows = [];
    for (let slot = 1; slot <= 8; slot++) {
        const file = node.widgets?.find(widget => widget.name === `lora_${slot}`);
        const enabled = node.widgets?.find(widget => widget.name === `enabled_${slot}`);
        if (!file || !enabled) continue;
        // Keep the boolean in schema order for both API prompts and workflow saves.
        enabled.hidden = true;
        enabled.computeSize = () => [0, -4];

        function change(widget, value, event, pos) {
            const oldValue = widget.value;
            if (oldValue === value) return;
            node.graph?.beforeChange?.();
            widget.value = value;
            widget.callback?.(value, app.canvas, node, pos, event);
            node.onWidgetChanged?.(widget.name, value, oldValue, widget);
            node.graph?.incrementVersion?.();
            node.graph?.afterChange?.();
            refresh();
        }

        // Preserve the native combo, its value binding and file-list refresh support.
        Object.assign(file, {
            draw(ctx, owner, width, y, height) {
                const on = enabled.value !== false;
                ctx.save();
                ctx.fillStyle = '#222';
                ctx.strokeStyle = '#666';
                ctx.beginPath();
                ctx.roundRect(10, y, width - 20, height, height / 2);
                ctx.fill();
                ctx.stroke();
                ctx.fillStyle = on ? '#287d54' : '#4b5058';
                ctx.beginPath();
                ctx.roundRect(18, y + (height - 14) / 2, 32, 14, 7);
                ctx.fill();
                ctx.fillStyle = on ? '#d8ffe7' : '#aeb3ba';
                ctx.beginPath();
                ctx.arc(on ? 43 : 25, y + height / 2, 5, 0, Math.PI * 2);
                ctx.fill();
                ctx.font = '12px sans-serif';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = on ? '#ddd' : '#999';
                const label = `LoRA ${slot}: ${String(file.value ?? '(none)').split(/[\\/]/).at(-1)}`;
                ctx.fillText(fitLabel(ctx, label, Math.max(0, width - 91)), 58, y + height / 2);
                ctx.fillText('▾', width - 28, y + height / 2);
                ctx.restore();
            },
            mouse(event, pos) {
                if (event.type !== 'pointerdown' || (event.button != null && event.button !== 0)) return false;
                if (pos[0] < 10 || pos[0] > node.size[0] - 10) return false;
                if (pos[0] <= 53) {
                    change(enabled, enabled.value === false, event, pos);
                    return true;
                }
                const options = file.options || {};
                const values = typeof options.values === 'function' ? options.values(file, node) : options.values;
                new LiteGraph.ContextMenu(values || [], {
                    event,
                    scale: Math.max(1, app.canvas?.ds?.scale || 1),
                    className: 'dark',
                    callback: value => {
                        if (typeof value === 'string') change(file, value, event, pos);
                    },
                });
                return true;
            },
            onPointerDown(pointer, owner, canvas) {
                const event = pointer.eDown;
                if (!event) return false;
                return file.mouse(event, [canvas.graph_mouse[0] - owner.pos[0],
                    canvas.graph_mouse[1] - owner.pos[1]]);
            },
        });
        rows.push([file, enabled]);
    }
    if (!rows.length) return;

    function refresh() {
        for (const [file, enabled] of rows) {
            file.tooltip = text(app,
                `LoRA ${file.name.slice(5)}: click the switch to enable/disable. OFF preserves the file and weight and skips this source.\n${file.value}`,
                `LoRA ${file.name.slice(5)}：スイッチで有効・無効を切り替えます。OFFでもファイルと重みを保持し、合成対象から除外します。\n${file.value}`);
            // Never expose a separate toggle row, including after Advanced is opened.
            enabled.hidden = true;
        }
        (node.graph || app.graph)?.setDirtyCanvas?.(true, true);
    }
    node.loraBlendSourceToggles = { refresh };
    const configured = node.onConfigure;
    node.onConfigure = function (...args) {
        const result = configured?.apply(this, args);
        refresh();
        return result;
    };
    watchLocale(app, node, refresh);
    refresh();
    if (node.computeSize && node.setSize) node.setSize(node.computeSize());
}

app.registerExtension({
    name: 'lora_blend_one.source_toggles',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!SOURCE_NODES.has(nodeData.name)) return;
        const previous = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = previous?.apply(this, args);
            installSourceToggles(this);
            return result;
        };
    },
});
