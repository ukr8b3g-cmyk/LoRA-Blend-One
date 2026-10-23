import fs from 'node:fs';
import vm from 'node:vm';

export function setup() {
    let locale = 'en', payload = {
        file: 'moss/blend.safetensors', sources: [
            {name: 'a.safetensors', selected: 'moss/a.safetensors', weight: .5, status: 'matched'},
            {name: 'b.safetensors', selected: 'moss/b.safetensors', weight: -.25, status: 'matched'},
        ], settings: {merge_method: 'Weighted', weight_mode: 'direct', output_rank: 'Custom', custom_rank: 7,
            compression_mode: 'auto', output_dtype: 'bf16', output_scale: 1.2, seed: 17}, ready: true, issues: [],
    };
    const extensions = [], listeners = new Set(), requests = [];
    let pending;
    const api = { async fetchApi(url, options) {
        requests.push({url, options});
        if (url.endsWith('recipe_files')) return {ok: true, json: async () => ['moss/blend.safetensors', 'moss/a.safetensors', 'moss/b.safetensors']};
        const response = pending ? await pending : structuredClone(payload);
        return {ok: !response.error, status: response.error ? 400 : 200, json: async () => response};
    }};
    const graph = {nodes: new Map(), links: new Map([[1, {origin_id: 4}]]), before: 0, after: 0,
        get _nodes() {return [...this.nodes.values()];},
        beforeChange() { this.before++; }, afterChange() { this.after++; },
        incrementVersion() {}, setDirtyCanvas() {}, getNodeById(id) {return this.nodes.get(id);},
    };
    const app = { graph, canvas: {}, registerExtension(e) {extensions.push(e);}, ui: {settings: {
        getSettingValue: () => locale, addEventListener: (_, fn) => listeners.add(fn), removeEventListener: (_, fn) => listeners.delete(fn),
    }}};
    const document = {createElement() {return {style: {}, value: '', setAttribute() {}};}};
    const context = vm.createContext({app, api, document, queueMicrotask});
    for (const file of ['i18n.js', 'source_toggles.js', 'recipe.js', 'power_loader.js', 'accordion.js']) {
        const source = fs.readFileSync(new URL(`../../web/${file}`, import.meta.url), 'utf8')
            .replace(/^import .*;\r?\n/gm, '').replace(/^export /gm, '');
        const exported = file === 'i18n.js' ? 'Object.assign(globalThis,{text,watchLocale});'
            : file === 'source_toggles.js' ? 'globalThis.installSourceToggles=installSourceToggles;'
            : file === 'recipe.js' ? 'Object.assign(globalThis,{installRecipeControls,applyRecipeToNodes});'
            : file === 'power_loader.js' ? 'globalThis.installPowerLoaderControls=installPowerLoaderControls;' : '';
        vm.runInContext(`{ ${source}\n${exported} }`,context);
    }
    const sourceSpec = {required: {},optional: {}}, blendSpec = {required: {
        output_rank: [['Auto','16','Custom'],{default: '16'}],filename_prefix: ['STRING',{default: 'keep/my_output'}],
    }, optional: {}};
    for (let i=1;i<=8;i++) {
        const group = i<=2 ? sourceSpec.required : sourceSpec.optional;
        group[`lora_${i}`] = [['(none)','old.safetensors'],{default: 'old.safetensors',advanced:i>2}];
        group[`weight_${i}`] = ['FLOAT',{default: i*10,advanced:i>2}];
        sourceSpec.optional[`enabled_${i}`] = ['BOOLEAN',{default:false}];
    }
    for (const [name,value] of Object.entries(payload.settings)) {
        if (name in blendSpec.required) continue;
        blendSpec.optional[name] = [typeof value === 'string' ? [value] : 'FLOAT',{default: value,advanced: true}];
    }
    blendSpec.optional.control_after_generate = [['fixed','randomize'],{default:'randomize',advanced:true}];
    class Base {
        constructor(type,id,spec,owner=graph) {
            this.id=id;this.type=type;this.graph=owner;this.properties={};this.size=[340,120];
            this.widgets=Object.entries({...spec.required,...spec.optional}).map(([name,[kind,options]])=>({name,
                value:options.default,options:{...options,values:Array.isArray(kind)?kind:undefined}}));
            this.inputs=type==='LoRABlendOneSplit'?[{name:'sources',link:1}]:[];
            owner?.nodes.set(id,this);this.onNodeCreated();
        }
        addWidget(type,name,value,callback,options) {const w={type,name,value,callback,options};this.widgets.push(w);return w;}
        addDOMWidget(name,type,element,options) {const w={name,type,element,options};this.widgets.push(w);return w;}
        computeSize() {return [340,40+this.widgets.reduce((h,w)=>h+(w.hidden?0:24),0)];}
        setSize(size) {this.size=size;}
        serializeFromStoreState() {return {properties:{...this.properties},widgets_values:this.widgets.filter(w=>w.serialize!==false).map(w=>w.value)};}
        serialize() {return this.serializeFromStoreState();}
        configure(info) {
            this.properties={...info.properties};
            this.widgets.filter(w=>w.serialize!==false).forEach((w,i)=>w.value=info.widgets_values[i]);
            this.onConfigure?.(info);
        }
    }
    class Source extends Base {constructor(owner=graph) {super('LoRABlendSources',4,sourceSpec,owner);}}
    class Blend extends Base {constructor(owner=graph) {super('LoRABlendOneSplit',5,blendSpec,owner);}}
    for (const e of extensions) {
        e.beforeRegisterNodeDef(Source,{name:'LoRABlendSources',input:sourceSpec});
        e.beforeRegisterNodeDef(Blend,{name:'LoRABlendOneSplit',input:blendSpec});
    }
    const source=new Source(), blend=new Blend();
    const value=(node,name)=>node.widgets.find(w=>w.name===name)?.value;
    const controls=blend.widgets.filter(w=>w.loraBlendAdvanced);
    const [select,read,details,restore]=controls;
    const header=blend.widgets.find(w=>w.type==='button'&&w.serialize===false&&!w.loraBlendAdvanced);
    return {source,blend,graph,controls,select,read,details,restore,header,value,requests,app,api,context,extensions,Source,Blend,
        async choose() {select.value='moss/blend.safetensors';await select.callback(select.value);},
        setLocale(l) {locale=l;for(const listener of listeners)listener();},
        setPayload(p) {payload=p;}, getPayload() {return structuredClone(payload);},
        setPending(p) {pending=p;}, newBlend:()=>new Blend(),
    };
}
