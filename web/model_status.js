import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';
import { text } from './i18n.js';

const MERGE_NODES = new Set(['LoRABlendOne', 'LoRABlendOneSplit']);
const ERROR_NODES = new Set([...MERGE_NODES, 'LoRABlendSources']);
const STATUS_HEIGHT = 26;
const LINE_HEIGHT = 17;

function currentNode(node) {
    return node.graph && (node.graph === app.graph || node.graph === app.rootGraph)
        && node.graph.getNodeById?.(node.id) === node;
}

function validationFor(node) {
    return (app.extensionManager?.lastNodeErrors ?? app.lastNodeErrors)?.[String(node.id)];
}

function sourceNames(node, inputs = {}) {
    const socket = node.inputs?.find(input => input.name === 'sources');
    const links = node.graph?.links;
    const link = links?.get?.(socket?.link) ?? links?.[socket?.link];
    const owner = link ? node.graph.getNodeById(link.origin_id) : node;
    const sources = [];
    for (let slot = 1; slot <= 8; slot++) {
        const enabled = owner?.widgets?.find(widget => widget.name === `enabled_${slot}`);
        if (enabled?.value === false) continue;
        let value = inputs[`lora_${slot}`] ?? owner?.widgets?.find(widget => widget.name === `lora_${slot}`)?.value;
        if (Array.isArray(value) && value.length === 1) value = value[0];
        if (typeof value === 'string' && value && value !== '(none)') sources.push({ slot, name: value });
    }
    return sources;
}

function validationFailure(node, entry) {
    const errors = entry.errors || [];
    const customValidation = error => error.type === 'custom_validation_failed' || error.message === 'Custom validation failed';
    const messages = errors.map(error => customValidation(error)
        ? String(error.details || error.message || '').replace(/^[A-Za-z_]\w*\s*-\s*/, '')
        : [error.message, error.details].filter(Boolean).join(': '));
    const detail = [...new Set(messages)].join('\n');
    const missing = detail.match(/LoRA (\d+): selected file is unavailable: ([^\r\n]+?)\. Choose an installed LoRA or turn this slot off\./);
    if (missing) return {
        code: 'missing_file', sources: [{ slot: Number(missing[1]), name: missing[2] }], detail: missing[0],
        reason: { en: 'The selected LoRA file is unavailable.', ja: '選択したLoRAファイルが見つかりません。' },
        hint: { en: 'Choose an installed LoRA or turn this slot OFF, then retry.', ja: 'インストール済みのLoRAを選ぶか、このLoRAをOFFにして再実行してください。' },
    };
    // Custom validators return a node-wide verdict; repeated input names do not identify its cause.
    const field = customValidation(errors[0] || {}) ? undefined : errors[0]?.extra_info?.input_name;
    const slot = field?.match(/^lora_(\d+)$/)?.[1];
    const sources = slot ? sourceNames(node).filter(source => source.slot === Number(slot)) : [];
    return {
        code: 'invalid_settings', sources, detail,
        reason: {
            en: `${field ? `Invalid input: ${field}.` : 'Input validation failed.'}\n${detail}`,
            ja: `${field ? `${field} の入力値が無効です。` : '入力内容の検証に失敗しました。'}\n${detail}`,
        },
        hint: { en: 'Correct the indicated input and retry.', ja: '表示された入力項目を修正して再実行してください。' },
    };
}

function executionFailure(node, detail) {
    const message = String(detail.exception_message || detail.exception_type || 'Execution failed.');
    return {
        code: 'execution_error', sources: sourceNames(node, detail.current_inputs), detail: message,
        reason: { en: message, ja: `実行中にエラーが発生しました。\n${message}` },
        hint: {
            en: 'Check the listed LoRAs and the error above. Correct the setting or turn the incompatible LoRA OFF, then retry.',
            ja: '表示されたLoRAとエラー内容を確認し、設定を修正するか互換性のないLoRAをOFFにして再実行してください。',
        },
    };
}

function wrapText(ctx, value, width) {
    const lines = [];
    for (const paragraph of String(value).split('\n')) {
        let line = '';
        for (const character of paragraph) {
            if (line && ctx.measureText(line + character).width > width) {
                lines.push(line); line = '';
            }
            line += character;
        }
        lines.push(line);
    }
    return lines;
}

function failureLines(ctx, node) {
    const error = node.loraBlendFailure;
    const local = value => text(app, value?.en || value?.ja || '', value?.ja || value?.en || '');
    const names = error.sources?.filter(source => typeof source.name === 'string')
        .map(source => `${source.slot ? `LoRA ${source.slot}: ` : ''}${source.name}`) || [];
    const reason = local(error.reason) || error.detail || '';
    const sections = [text(app, 'Blend stopped', 'ブレンド停止'),
        `${text(app, 'LoRA', '対象LoRA')}: ${names.length ? names.join('\n') : text(app, 'Not identified (see details).', '特定できません（詳細を確認）。')}`,
        `${text(app, 'Reason', '理由')}: ${reason}`,
        `${text(app, 'How to fix', '対処')}: ${local(error.hint)}`];
    if (error.detail && !reason.includes(error.detail)) sections.push(`${text(app, 'Details', '詳細')}: ${error.detail}`);
    const width = Math.max(40, node.size[0] - 40);
    return sections.flatMap((section, index) => [...(index ? [''] : []), ...wrapText(ctx, section, width)]);
}

function resizeStatus(node) {
    if (node.computeSize && node.setSize) {
        const size = node.computeSize();
        node.setSize([Math.max(node.size?.[0] || 0, size[0]), size[1]]);
    }
    (node.graph || app.graph)?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: 'lora_blend_one.model_status',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!ERROR_NODES.has(nodeData.name)) return;
        const baseHeight = MERGE_NODES.has(nodeData.name) ? STATUS_HEIGHT : 0;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            const node = this;
            this.loraBlendDetectedModel = null;
            this.loraBlendFailure = null;
            this.loraBlendStatusHeight = baseHeight;
            const computeSize = this.computeSize;
            this.computeSize = function (...args) {
                const size = computeSize.apply(this, args);
                return [size[0], size[1] + this.loraBlendStatusHeight];
            };

            function clear(clearModel = false) {
                node.loraBlendFailure = null;
                node.loraBlendStructuredFailure = false;
                node.loraBlendStatusHeight = baseHeight;
                node.loraBlendSeenValidation = validationFor(node);
                if (clearModel) node.loraBlendDetectedModel = null;
                resizeStatus(node);
            }
            function show(error, structured = false) {
                node.loraBlendFailure = error;
                node.loraBlendStructuredFailure = structured;
                node.loraBlendDetectedModel = null;
                node.loraBlendStatusHeight = 160;
                resizeStatus(node);
            }
            this.loraBlendClearFailure = clear;
            this.loraBlendShowFailure = show;

            // Widget lifecycle hooks associate an accepted run with its submitting graph.
            const first = this.widgets?.[0];
            if (first) {
                const beforeQueued = first.beforeQueued;
                first.beforeQueued = function (...args) {
                    const value = beforeQueued?.apply(this, args);
                    node.loraBlendQueuedGraph = node.graph;
                    clear(true);
                    return value;
                };
            }
            function sameWorkflow(detail) {
                const graphId = node.graph?.rootGraph?.id ?? node.graph?.id;
                return graphId != null && String(detail.workflow_id) === String(graphId);
            }
            function sameRun(detail) {
                return currentNode(node) && node.loraBlendRunGraph === node.graph
                    && detail.prompt_id && detail.prompt_id === node.loraBlendPromptId
                    && (detail.workflow_id == null || sameWorkflow(detail));
            }
            const handlers = {
                execution_start: ({ detail }) => {
                    node.loraBlendObservedPromptId = detail.prompt_id;
                    const queuedGraph = node.loraBlendQueuedGraph;
                    node.loraBlendQueuedGraph = null;
                    if (!currentNode(node)) return;
                    if (detail.workflow_id != null) {
                        if (!sameWorkflow(detail)) return;
                    } else if (queuedGraph !== node.graph) return;
                    node.loraBlendPromptId = detail.prompt_id;
                    node.loraBlendRunGraph = node.graph;
                    clear(true);
                },
                executing: ({ detail }) => {
                    if (currentNode(node) && node.loraBlendRunGraph === node.graph
                        && node.loraBlendObservedPromptId === node.loraBlendPromptId && String(detail) === String(node.id)) clear(true);
                },
                'lora_blend_one.error': ({ detail }) => {
                    if (sameRun(detail) && String(detail.node_id) === String(node.id) && detail.error) show(detail.error, true);
                },
                execution_error: ({ detail }) => {
                    if (!sameRun(detail) || String(detail.node_id) !== String(node.id)
                        || (detail.node_type && detail.node_type !== nodeData.name) || node.loraBlendStructuredFailure) return;
                    show(executionFailure(node, detail));
                },
                execution_success: ({ detail }) => {
                    if (!sameRun(detail)) return;
                    node.loraBlendQueuedGraph = null;
                    clear();
                },
            };
            const settings = app.ui?.settings;
            const localeChanged = () => resizeStatus(node);
            let subscribed = false;
            function subscribe() {
                if (subscribed) return;
                subscribed = true;
                for (const [name, handler] of Object.entries(handlers)) api.addEventListener(name, handler);
                settings?.addEventListener?.('Comfy.Locale.change', localeChanged);
            }
            function unsubscribe() {
                if (!subscribed) return;
                subscribed = false;
                for (const [name, handler] of Object.entries(handlers)) api.removeEventListener(name, handler);
                settings?.removeEventListener?.('Comfy.Locale.change', localeChanged);
            }
            const added = this.onAdded;
            this.onAdded = function (...args) {
                const result = added?.apply(this, args);
                subscribe();
                return result;
            };
            const removed = this.onRemoved;
            this.onRemoved = function (...args) {
                unsubscribe();
                node.loraBlendQueuedGraph = null;
                node.loraBlendPromptId = null;
                node.loraBlendRunGraph = null;
                node.loraBlendObservedPromptId = null;
                return removed?.apply(this, args);
            };
            subscribe();
            resizeStatus(this);
            return result;
        };

        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = executed?.apply(this, arguments);
            if (this.loraBlendObservedPromptId && this.loraBlendObservedPromptId !== this.loraBlendPromptId) return result;
            this.loraBlendClearFailure?.();
            this.loraBlendDetectedModel = message?.model_detection?.[0] || null;
            (this.graph || app.graph)?.setDirtyCanvas?.(true, true);
            return result;
        };

        const draw = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx, ...args) {
            const result = draw?.call(this, ctx, ...args);
            if (this.flags?.collapsed) return result;
            if (currentNode(this)) {
                const validation = validationFor(this);
                if (validation !== this.loraBlendSeenValidation) {
                    this.loraBlendSeenValidation = validation;
                    if (validation?.errors?.length && !this.loraBlendStructuredFailure)
                        this.loraBlendShowFailure(validationFailure(this, validation));
                }
            }
            if (this.loraBlendFailure) {
                ctx.save();
                ctx.font = '12px sans-serif';
                const lines = failureLines(ctx, this);
                const height = lines.length * LINE_HEIGHT + 20;
                if (height !== this.loraBlendStatusHeight) {
                    this.loraBlendStatusHeight = height;
                    resizeStatus(this);
                }
                const y = this.size[1] - height + 3;
                ctx.fillStyle = '#4b2025';
                ctx.fillRect(8, y, this.size[0] - 16, height - 6);
                ctx.fillStyle = '#ffd5d9';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                for (let index = 0; index < lines.length; index++) ctx.fillText(lines[index], 18, y + 8 + index * LINE_HEIGHT);
                ctx.restore();
                return result;
            }
            if (!MERGE_NODES.has(nodeData.name)) return result;
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
