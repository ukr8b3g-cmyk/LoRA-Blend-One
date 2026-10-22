import { app } from '../../scripts/app.js';
import { text, watchLocale } from './i18n.js';

const MERGE_NODES = new Set(['LoRABlendOne', 'LoRABlendOneSplit']);
const STATUS_HEIGHT = 26;

app.registerExtension({
    name: 'lora_blend_one.model_status',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!MERGE_NODES.has(nodeData.name)) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            this.loraBlendDetectedModel = null;
            const computeSize = this.computeSize;
            this.computeSize = function (...args) {
                const size = computeSize.apply(this, args);
                return [size[0], size[1] + STATUS_HEIGHT];
            };
            this.setSize(this.computeSize());
            watchLocale(app, this, () => (this.graph || app.graph)?.setDirtyCanvas?.(true, true));
            return result;
        };

        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = executed?.apply(this, arguments);
            this.loraBlendDetectedModel = message?.model_detection?.[0] || null;
            (this.graph || app.graph)?.setDirtyCanvas?.(true, true);
            return result;
        };

        const draw = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx, ...args) {
            const result = draw?.call(this, ctx, ...args);
            if (this.flags?.collapsed) return result;
            const model = this.loraBlendDetectedModel;
            const label = model
                ? text(app, `Last detected model: ${model}`, `前回のモデル判定：${model}`)
                : text(app, 'Model detection: run to detect', 'モデル判定：実行後に表示');
            const y = this.size[1] - STATUS_HEIGHT + 3;
            ctx.save();
            ctx.fillStyle = model === 'MOSS-Image' ? '#163d2d' : '#252a34';
            ctx.fillRect(8, y, this.size[0] - 16, 20);
            ctx.font = '12px sans-serif';
            ctx.fillStyle = model === 'MOSS-Image' ? '#a7f3c5' : '#ccd2dc';
            ctx.fillText(label, 14, y + 14, this.size[0] - 28);
            ctx.restore();
            return result;
        };
    },
});
