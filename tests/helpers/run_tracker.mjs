import fs from 'node:fs';
import vm from 'node:vm';

export function loadRunTracker(context) {
    const source = fs.readFileSync(new URL('../../web/run_tracker.js', import.meta.url), 'utf8')
        .replace(/^import .*;\r?\n/gm, '').replace(/^export /gm, '');
    vm.runInContext(`{ ${source}\nglobalThis.trackNodeRuns = trackNodeRuns; }`, context);
}
