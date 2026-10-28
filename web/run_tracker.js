import { api } from '../../scripts/api.js';

const queuedNodes = new Set();
const queuePrompt = api.queuePrompt;
api.queuePrompt = async function (number, data, ...args) {
    const queued = [...queuedNodes];
    queuedNodes.clear();
    const accepted = queued.map(capture => capture(data)).filter(Boolean);
    let response;
    try {
        response = await queuePrompt.call(this, number, data, ...args);
        return response;
    } finally {
        for (const finish of accepted) finish(response?.prompt_id);
    }
};

// Each accepted prompt belongs to the submitted node object, even after a tab switch.
export function trackNodeRuns(node, nodeType, handlers) {
    const runs = new Map(), pending = new Set();
    const terminal = new Set(['execution_success', 'execution_error', 'execution_interrupted']);
    let queuedGraph, activePrompt, subscribed = false;
    let received = 0, displayed = 0;
    const workflowId = () => node.graph?.rootGraph?.id ?? node.graph?.id;
    const current = run => node.graph === run.graph && String(node.id) === run.nodeId
        && node.graph?.getNodeById?.(node.id) === node && workflowId() === run.workflowId;
    const sameWorkflow = (detail, run) => current(run)
        && (detail.workflow_id == null || String(detail.workflow_id) === String(run.workflowId));
    const sameNode = detail => String(detail.node_id ?? detail.display_node ?? detail.node) === String(node.id);

    function deliver(name, detail, run, sequence) {
        if (!sameWorkflow(detail, run)) return;
        const id = detail.prompt_id;
        if (name === 'execution_start') run.started = true;
        // A delayed HTTP acknowledgement must not replay older UI over a newer job.
        if (sequence < displayed) {
            if (terminal.has(name)) runs.delete(id);
            return;
        }
        if (name === 'execution_start') {
            activePrompt = id;
            displayed = sequence;
            handlers.execution_start?.(detail);
        } else if (sameNode(detail) && (!run.started || activePrompt === id)) {
            displayed = sequence;
            if (activePrompt !== id) {
                activePrompt = id;
                handlers.execution_start?.(detail);
            }
            handlers[name]?.(detail);
        } else if (name === 'execution_success' && activePrompt === id) {
            displayed = sequence;
            handlers.execution_success?.(detail);
        }
        if (terminal.has(name)) runs.delete(id);
    }

    function capture(prompt) {
        const run = {graph: queuedGraph, workflowId: workflowId(), nodeId: String(node.id), early: new Map()};
        queuedGraph = null;
        if (!run.graph || !current(run) || prompt.output?.[node.id]?.class_type !== nodeType
            || (prompt.workflow?.id != null && String(prompt.workflow.id) !== String(run.workflowId))) return;
        pending.add(run);
        return id => {
            if (!pending.delete(run) || !id || !current(run)) return;
            runs.set(id, run);
            // A quick job may finish before /prompt returns. Replay its result, then retire it.
            for (const [name, event] of run.early.get(id) || []) deliver(name, event.detail, run, event.sequence);
            run.early.clear();
        };
    }

    const listeners = new Map();
    for (const name of new Set(['execution_start', ...terminal, ...Object.keys(handlers)])) {
        listeners.set(name, ({detail}) => {
            if (!detail?.prompt_id) return;
            const sequence = ++received;
            const run = runs.get(detail.prompt_id);
            if (run) {
                deliver(name, detail, run, sequence);
                return;
            }
            for (const request of pending) {
                if (!sameWorkflow(detail, request) || (!terminal.has(name) && name !== 'execution_start' && !sameNode(detail))) continue;
                let events = request.early.get(detail.prompt_id);
                if (!events) request.early.set(detail.prompt_id, events = new Map());
                // Keep only the latest progress of each kind, in arrival order.
                if ([...events.keys()].some(key => terminal.has(key))) continue;
                events.delete(name);
                events.set(name, {detail, sequence});
            }
        });
    }
    listeners.set('status', ({detail}) => {
        // Deleted waiting jobs have no terminal event on older servers.
        if (detail?.exec_info?.queue_remaining === 0) runs.clear();
    });
    function subscribe() {
        if (subscribed) return;
        subscribed = true;
        for (const [name, listener] of listeners) api.addEventListener(name, listener);
    }
    const first = node.widgets?.[0], beforeQueued = first?.beforeQueued;
    if (first) first.beforeQueued = function (...args) {
        const result = beforeQueued?.apply(this, args);
        queuedGraph = node.graph;
        queuedNodes.add(capture);
        return result;
    };
    const added = node.onAdded;
    node.onAdded = function (...args) {
        const result = added?.apply(this, args);
        subscribe();
        return result;
    };
    const removed = node.onRemoved;
    node.onRemoved = function (...args) {
        for (const [name, listener] of listeners) api.removeEventListener(name, listener);
        subscribed = false;
        queuedNodes.delete(capture);
        runs.clear(); pending.clear(); queuedGraph = null;
        activePrompt = null;
        return removed?.apply(this, args);
    };
    subscribe();
}
