import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';
import { text, watchLocale } from './i18n.js';

const names = new Set(['LoRABlendModelInspect', 'LoRABlendModelExtract']);
app.registerExtension({
    name: 'LoRABlendOne.Extractor',
    beforeRegisterNodeDef(type, data) {
        if (!names.has(data.name)) return;
        const created = type.prototype.onNodeCreated;
        type.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            const node = this;
            const originals = [...(node.widgets || [])];
            const advanced = new Set(Object.entries(data.input?.optional || {}).filter(([,spec]) => spec[1]?.advanced).map(([key]) => key));
            const sizes = new Map(originals.map(w => [w, {type:w.type, computeSize:w.computeSize}]));
            node.properties ??= {};
            const button = node.addWidget('button', '', null, () => {
                node.properties.extract_advanced = !node.properties.extract_advanced;
                refresh();
            }, { serialize:false });
            button.serialize = false;
            node.widgets.splice(node.widgets.indexOf(button), 1);
            const firstAdvanced = originals.findIndex(w => advanced.has(w.name));
            node.widgets.splice(firstAdvanced < 0 ? originals.length : firstAdvanced, 0, button);
            const status = document.createElement('textarea');
            status.readOnly = true;
            status.style.cssText = 'width:100%;height:100%;background:#202020;color:#ddd;resize:none;font-size:12px;';
            const statusWidget = node.addDOMWidget?.('extract_status', 'text', status, {serialize:false});
            if (statusWidget) { statusWidget.serialize = false; statusWidget.computeSize = () => [320,100]; }
            node.lboExtractStatus = status;
            function refresh() {
                const open = !!node.properties.extract_advanced;
                button.name = `${open ? '▼' : '▶'} ${text(app, 'Advanced', '詳細設定')}`;
                const mode = originals.find(w => w.name === 'mode')?.value;
                for (const w of originals) {
                    const hidden = (advanced.has(w.name) && !open) || (w.name === 'rank' && mode !== 'Fixed Rank')
                        || (w.name === 'energy' && mode !== 'Auto Rank');
                    w.type = hidden ? 'lbo_hidden' : sizes.get(w).type;
                    w.computeSize = hidden ? () => [0,-4] : sizes.get(w).computeSize;
                }
                node.setSize?.([Math.max(node.size?.[0] || 340, 340), node.computeSize()[1]]);
                node.setDirtyCanvas?.(true,true);
            }
            const mode = originals.find(w => w.name === 'mode');
            if (mode) {
                const callback = mode.callback;
                mode.callback = function (...args) { const r = callback?.apply(this,args); refresh(); return r; };
            }
            const configured = node.onConfigure;
            node.onConfigure = function (...args) { const r = configured?.apply(this,args); refresh(); return r; };
            const executed = node.onExecuted;
            node.onExecuted = function (message) {
                executed?.call(this,message);
                this.title = data.display_name || data.name;
                if (message?.text) status.value = message.text.join('\n');
            };
            watchLocale(app, node, refresh);
            refresh();
            return result;
        };
    },
});
api.addEventListener('lora_blend_one.extract_status', event => {
    const node = app.graph?.getNodeById(event.detail.node_id);
    if (names.has(node?.comfyClass) && node.lboExtractStatus) {
        let value = event.detail.text;
        if (value.startsWith('Stopped')) value = text(app, value,
            value.replace('Stopped', '停止') + '\n入力ファイル・対象キー・保存精度を確認してください。');
        node.lboExtractStatus.value = value;
    }
});
