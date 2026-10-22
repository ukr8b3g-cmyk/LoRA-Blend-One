// Follow ComfyUI's language setting; use English for other locales.
export function language(app) {
    const locale = app?.extensionManager?.setting?.get?.('Comfy.Locale')
        ?? app?.ui?.settings?.getSettingValue?.('Comfy.Locale') ?? 'en';
    return /^ja(?:[-_]|$)/i.test(locale) ? 'ja' : 'en';
}

export function text(app, english, japanese) {
    return language(app) === 'ja' ? japanese : english;
}

export function watchLocale(app, node, refresh) {
    const settings = app?.ui?.settings;
    if (!settings?.addEventListener) return;
    const handler = () => refresh();
    settings.addEventListener('Comfy.Locale.change', handler);
    const removed = node.onRemoved;
    node.onRemoved = function (...args) {
        settings.removeEventListener('Comfy.Locale.change', handler);
        return removed?.apply(this, args);
    };
}
