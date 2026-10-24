import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

test('extraction progress stays with its submitted graph and prompt', () => {
    const listeners=new Map();let extension;
    const api={addEventListener(k,fn){if(!listeners.has(k))listeners.set(k,new Set());listeners.get(k).add(fn);},
        removeEventListener(k,fn){listeners.get(k)?.delete(fn);}};
    const app={ui:{settings:{getSettingValue:()=> 'ja',addEventListener(){},removeEventListener(){}}},registerExtension:e=>extension=e};
    const box={app,api,document:{createElement:()=>({style:{}})}};vm.createContext(box);
    vm.runInContext(fs.readFileSync(new URL('../web/i18n.js',import.meta.url),'utf8').replaceAll('export ',''),box);
    vm.runInContext(fs.readFileSync(new URL('../web/extractor.js',import.meta.url),'utf8').replace(/^import .*$/gm,''),box);
    class Node {
        constructor(){this.id=1;this.size=[340,200];this.widgets=[{name:'mode',type:'combo',value:'Fixed Rank'}];}
        addWidget(type,name,value,callback,options){const w={type,name,value,callback,options};this.widgets.push(w);return w;}
        addDOMWidget(name,type,element,options){const w={name,type,element,options};this.widgets.push(w);return w;}
        computeSize(){return [340,200];}setSize(s){this.size=s;}
    }
    extension.beforeRegisterNodeDef(Node,{name:'LoRABlendModelExtract',input:{optional:{}}});
    const a=new Node(),b=new Node();
    a.graph={id:'a',getNodeById:()=>a};b.graph={id:'b',getNodeById:()=>b};a.onNodeCreated();b.onNodeCreated();
    const send=(k,detail)=>{for(const fn of listeners.get(k)||[])fn({detail});};
    const progress=d=>send('lora_blend_one.extract_status',{node_id:'1',prompt_id:'p-a',workflow_id:'a',text:'Extract A',...d});
    a.widgets[0].beforeQueued();send('execution_start',{prompt_id:'p-a',workflow_id:'a'});
    const bBefore=b.lboExtractStatus.value;app.graph=b.graph;
    progress();assert.equal(a.lboExtractStatus.value,'Extract A');assert.equal(b.lboExtractStatus.value,bBefore);
    for(const invalid of [{prompt_id:'old'},{node_id:'99'},{workflow_id:'b'},{prompt_id:undefined}]) {
        progress({...invalid,text:'Wrong'});assert.equal(a.lboExtractStatus.value,'Extract A');
    }
    b.widgets[0].beforeQueued();send('execution_start',{prompt_id:'p-b'});
    progress({prompt_id:'p-b',workflow_id:'b',text:'Stopped reason'});
    assert.match(b.lboExtractStatus.value,/停止 reason/);
    const bResult=b.lboExtractStatus.value;
    send('execution_start',{prompt_id:'foreign'});progress({prompt_id:'foreign',workflow_id:'b',text:'Wrong'});
    assert.equal(b.lboExtractStatus.value,bResult);
    a.graph.id='replacement';progress({text:'Wrong'});assert.equal(a.lboExtractStatus.value,'Extract A');
    a.onRemoved();b.onRemoved();assert.equal(listeners.get('execution_start').size,0);
    assert.equal(listeners.get('lora_blend_one.extract_status').size,0);
});

test('extractor accordion, mode visibility, locale, status and values', () => {
    const events = {}, apiEvents = {};
    let locale = 'en', extension;
    const app = {ui:{settings:{getSettingValue:()=>locale, addEventListener:(key,fn)=>events[key]=fn,
        removeEventListener:()=>{}}}, registerExtension: value=>extension=value};
    const sandbox={app,api:{addEventListener:(key,fn)=>apiEvents[key]=fn},document:{createElement:()=>({style:{}})}};
    vm.createContext(sandbox);
    const i18n=fs.readFileSync(new URL('../web/i18n.js',import.meta.url),'utf8').replaceAll('export ','');
    vm.runInContext(i18n,sandbox);
    const code=fs.readFileSync(new URL('../web/extractor.js',import.meta.url),'utf8').replace(/^import .*$/gm,'');
    vm.runInContext(code,sandbox);
    class Node {
        constructor(){ this.widgets=['mode','rank','energy','seed'].map(name=>({name,type:'number',value:name==='mode'?'Fixed Rank':16,computeSize:()=>[100,20]})); this.size=[340,200]; }
        addWidget(type,name,value,callback,options){const w={type,name,value,callback,options};this.widgets.push(w);return w;}
        addDOMWidget(name,type,element,options){const w={name,type,element,options};this.widgets.push(w);return w;}
        computeSize(){return [340,200];} setSize(size){this.size=size;}
    }
    extension.beforeRegisterNodeDef(Node,{name:'LoRABlendModelExtract',input:{optional:{seed:['INT',{advanced:true}]}}});
    const n=new Node();n.id=1;n.graph={id:'workflow-a',getNodeById:()=>n};n.comfyClass='LoRABlendModelExtract';n.onNodeCreated();
    const find=name=>n.widgets.find(w=>w.name===name);
    assert.equal(find('rank').type,'number');assert.equal(find('energy').type,'lbo_hidden');
    assert.equal(find('seed').type,'lbo_hidden');
    const button=n.widgets.find(w=>w.type==='button');button.callback();assert.equal(find('seed').type,'number');
    find('mode').value='Auto Rank';find('mode').callback();assert.equal(find('rank').type,'lbo_hidden');assert.equal(find('energy').type,'number');
    locale='ja';events['Comfy.Locale.change']();assert.match(button.name,/Advanced/);assert.match(button.tooltip,/表示・非表示/);
    assert.equal(find('seed').value,16);assert.equal(button.serialize,false);
    n.widgets[0].beforeQueued();apiEvents.execution_start({detail:{prompt_id:'prompt-a'}});
    app.graph=n.graph;apiEvents['lora_blend_one.extract_status']({detail:{node_id:'1',prompt_id:'prompt-a',text:'Saved'}});
    assert.equal(n.lboExtractStatus.value,'Saved');n.onExecuted({text:['result']});assert.equal(n.lboExtractStatus.value,'result');
});


test('all extractor controls keep English labels and switch hover help without changing saved values', () => {
    const localeListeners = new Set();
    let locale = 'en', extension;
    const app = {ui:{settings:{getSettingValue:()=>locale,
        addEventListener:(_,fn)=>localeListeners.add(fn), removeEventListener:(_,fn)=>localeListeners.delete(fn)}},
        registerExtension: value=>extension=value};
    const sandbox={app,api:{addEventListener(){},removeEventListener(){}},document:{createElement:()=>({style:{}})}};
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync(new URL('../web/i18n.js',import.meta.url),'utf8').replaceAll('export ',''),sandbox);
    vm.runInContext(fs.readFileSync(new URL('../web/extractor.js',import.meta.url),'utf8').replace(/^import .*$/gm,''),sandbox);
    const en=JSON.parse(fs.readFileSync(new URL('../locales/en/nodeDefs.json',import.meta.url)));
    const ja=JSON.parse(fs.readFileSync(new URL('../locales/ja/nodeDefs.json',import.meta.url)));
    const setLocale = value => {locale=value; for(const listener of localeListeners) listener();};
    for(const id of ['LoRABlendModelInspect','LoRABlendModelExtract']) {
        setLocale('en');
        const fields=id.endsWith('Inspect')?['base_model','modified_model','scan']:Object.keys(en[id].inputs).filter(k=>k!=='scan');
        class Node {
            constructor() {
                this.widgets=fields.map(name=>({name,label:'old localized label',type:'number',value:name==='mode'?'Fixed Rank':17,computeSize:()=>[100,20]}));
                if(fields.includes('seed')) {
                    const control={name:'control_after_generate',label:'old control',type:'combo',value:'fixed',computeSize:()=>[100,20]};
                    this.widgets.find(w=>w.name==='seed').linkedWidgets=[control];
                    this.widgets.push(control);
                }
                this.size=[340,200];
            }
            addWidget(type,name,value,callback,options){const w={type,name,value,callback,options};this.widgets.push(w);return w;}
            addDOMWidget(name,type,element,options){const w={name,type,element,options};this.widgets.push(w);return w;}
            computeSize(){return [340,200];} setSize(size){this.size=size;}
        }
        const optional=Object.fromEntries(fields.filter(k=>!['base_model','modified_model','mode','rank','energy','dtype','filename_prefix'].includes(k)).map(k=>[k,['INT',{advanced:true}]]));
        extension.beforeRegisterNodeDef(Node,{name:id,display_name:en[id].display_name,input:{optional}});
        const n=new Node();
        const original=n.widgets.map(w=>({name:w.name,value:w.value}));
        n.onNodeCreated();
        assert.equal(n.title,en[id].display_name);
        const button=n.widgets.find(w=>w.type==='button');
        const find=name=>n.widgets.find(w=>w.name===name);
        for(const field of fields) {
            assert.equal(find(field).label,en[id].inputs[field].name);
            assert.equal(find(field).tooltip,en[id].inputs[field].tooltip);
            assert.ok(find(field).tooltip.length>30,field);
            assert.equal(ja[id].inputs[field].name,en[id].inputs[field].name);
        }
        assert.equal(ja[id].display_name,en[id].display_name);
        setLocale('ja-JP');
        for(const field of fields) {
            assert.equal(find(field).label,en[id].inputs[field].name);
            assert.equal(find(field).tooltip,ja[id].inputs[field].tooltip);
            assert.match(find(field).tooltip,/[ぁ-んァ-ン一-龯]/,field);
        }
        assert.equal(button.name,'▶ Advanced');
        assert.match(button.tooltip,/表示・非表示/);
        assert.match(n.lboExtractStatus.title,/画像品質/);
        if(fields.includes('seed')) {
            assert.equal(find('control_after_generate').type,'lbo_hidden');
            assert.equal(find('control_after_generate').label,'Seed Behavior');
            assert.match(find('control_after_generate').tooltip,/実行後/);
            button.callback();
            assert.equal(find('control_after_generate').type,'combo');
            assert.equal(find('seed').type,'number');
            assert.match(find('rank').tooltip,/17や24も有効/);
            assert.match(find('rank').tooltip,/8／16／32／64／128／256/);
        }
        const status=n.widgets.find(w=>w.name==='extract_status');
        assert.equal(status.computeSize,undefined);
        assert.equal(status.options.getMinHeight(),100);
        n.size=[700,800];
        n.onConfigure({size:[700,800]});
        assert.deepEqual(Array.from(n.size),[700,800]);
        button.callback();
        assert.equal(n.size[1],800);
        if(id.endsWith('Inspect')) {
            const info={architecture:'MOSS-Image',parameters:462,ignored_buffers:[[],['rope.bands']],scan:'Header',values_checked:false,
                largest_fp32_tensor_bytes:226492416,full_delta_bf16_bytes_upper_bound:10164617216};
            const message={text:[JSON.stringify(info)]};
            const before=JSON.stringify(message);
            n.onExecuted(message);
            assert.match(n.lboExtractStatus.value,/重みの値は未比較/);
            assert.match(n.lboExtractStatus.value,/216.00 MiB/);
            assert.match(n.lboExtractStatus.value,/10.16 GB/);
            assert.match(n.lboExtractStatus.value,/Modified Model: rope.bands/);
            assert.match(n.lboExtractStatus.value,/LoRAを保存しません/);
            assert.equal(JSON.stringify(message),before);
            assert.equal(n.properties.extract_inspection_result,message.text[0]);
            // ComfyUI can reconstruct nodes when changing locale; keep the last inspection.
            n.lboExtractMessage=undefined;
            n.lboExtractStatus.value='';
            n.onConfigure({properties:n.properties});
            assert.match(n.lboExtractStatus.value,/重みの値は未比較/);
            setLocale('en');
            assert.match(n.lboExtractStatus.value,/weight values NOT compared/);
            assert.equal(n.size[1],800);
            n.onExecuted({text:['not JSON']});
            assert.equal(n.lboExtractStatus.value,'not JSON');
            n.onExecuted({text:[JSON.stringify({...info,scan:'Full Scan',values_checked:true,changed:20,changed_non_matrix:3})]});
            assert.match(n.lboExtractStatus.value,/Changed entries: 20; nonmatrix changes: 3/);
            assert.match(n.lboExtractStatus.value,/LoRA matrix changes: 17; unchanged: 442/);
            assert.match(n.lboExtractStatus.value,/Full Scan completed on CPU/);
            assert.doesNotMatch(n.lboExtractStatus.value,/Next: Full Scan checks/);
            find('base_model').callback();
            assert.equal(n.properties.extract_inspection_result,undefined);
            assert.match(n.lboExtractStatus.value,/Run Inspect again/);
        }
        setLocale('en');
        for(const field of fields) assert.equal(find(field).tooltip,en[id].inputs[field].tooltip);
        assert.match(n.lboExtractStatus.title,/image-quality/);
        assert.equal(JSON.stringify(n.widgets.filter(w=>w.serialize!==false).map(w=>({name:w.name,value:w.value}))),JSON.stringify(original));
        n.title='My inspection';
        setLocale('fr');
        assert.equal(n.title,'My inspection');
        assert.equal(find('base_model').tooltip,en[id].inputs.base_model.tooltip);
        n.onRemoved();
        assert.equal(localeListeners.size,0);
    }
});

test('plan connection preserves manual settings and renders localized effective settings', () => {
    let extension, locale='en'; const listeners={};
    const app={ui:{settings:{getSettingValue:()=>locale,addEventListener:(k,fn)=>listeners[k]=fn,removeEventListener(){}}},registerExtension:e=>extension=e};
    const sandbox={app,api:{addEventListener(){}},document:{createElement:()=>({style:{}})}};
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync(new URL('../web/i18n.js',import.meta.url),'utf8').replaceAll('export ',''),sandbox);
    vm.runInContext(fs.readFileSync(new URL('../web/extractor.js',import.meta.url),'utf8').replace(/^import .*$/gm,''),sandbox);
    class Node {
        constructor() {
            this.inputs=[{name:'extraction_plan',link:null}];this.size=[600,600];
            this.widgets=Object.entries({base_model:'manual base',modified_model:'manual modified',mode:'Fixed Rank',rank:17,
                energy:.8,dtype:'fp32',filename_prefix:'manual-name',seed:123,plan_action:'Review Only',
                max_rank:128,device:'cpu',svd:'exact',non_lora:'Stop',key_filter:'*',control_after_generate:'randomize'})
                .map(([name,value])=>({name,value,type:'combo',computeSize:()=>[100,20]}));
        }
        addWidget(type,name,value,callback,options){const w={type,name,value,callback,options};this.widgets.push(w);return w;}
        addDOMWidget(name,type,element,options){const w={name,type,element,options};this.widgets.push(w);return w;}
        computeSize(){return [340,200];}setSize(s){this.size=s;}
    }
    extension.beforeRegisterNodeDef(Node,{name:'LoRABlendModelExtract',input:{optional:{seed:['INT',{advanced:true}]}}});
    const n=new Node();const before=JSON.stringify(n.widgets.map(w=>[w.name,w.value]));n.onNodeCreated();
    const find=k=>n.widgets.find(w=>w.name===k);
    assert.equal(find('rank').type,'combo');assert.equal(find('plan_action').type,'lbo_hidden');
    n.inputs[0].link=42;n.onConnectionsChange();
    for(const k of ['base_model','modified_model','mode','rank','dtype']) assert.equal(find(k).type,'combo');
    assert.equal(find('seed').type,'lbo_hidden');
    assert.equal(find('filename_prefix').type,'combo');assert.equal(find('plan_action').type,'combo');
    const action=find('plan_action'), labels=[], fills=[];
    const ctx={save(){},restore(){},beginPath(){},roundRect(){},fill(){fills.push(this.fillStyle);},stroke(){},fillText(label){labels.push(label);}};
    action.draw(ctx,n,600,100);
    assert.equal(action.computeSize()[1],56);
    assert.ok(labels.includes('✓ Review Only'));assert.ok(labels.includes('Next Run: review without saving'));
    assert.equal(action.mouse({type:'pointerdown',button:0},[450,115]),true);
    assert.equal(action.value,'Extract and Save');
    labels.length=0;fills.length=0;action.draw(ctx,n,600,100);
    assert.ok(labels.includes('✓ Extract and Save'));assert.ok(fills.includes('#236c48'));
    assert.ok(labels.includes('Next Run: extract and save'));
    n.onConfigure({});assert.equal(action.value,'Extract and Save');
    assert.equal(action.mouse({type:'pointerdown',button:2},[50,115]),false);
    assert.equal(action.mouse({type:'pointerdown',button:0},[300,115]),false); // between buttons
    assert.equal(action.mouse({type:'pointerdown',button:0},[50,150]),false); // caption
    n.inputs.push({name:'plan_action',link:99});
    assert.equal(action.mouse({type:'pointerdown',button:0},[50,115]),false);
    n.inputs.pop();
    n.pos=[20,30];
    assert.equal(action.onPointerDown({eDown:{type:'pointerdown',button:0}},n,{graph_mouse:[70,145]}),true);
    assert.equal(action.value,'Review Only');
    const plan={version:2,base_model:'analyzed base',modified_model:'analyzed modified',purpose:'ComfyUI Loading',ready:true,
        settings:{mode:'Fixed Rank',rank:32,energy:.99,max_rank:64,dtype:'bf16',device:'auto',svd:'auto',seed:0,non_lora:'Store Diff',key_filter:'*'},
        changed_non_matrix:1,non_matrix_examples:['final_norm_video.weight']};
    n.onExecuted({text:[JSON.stringify({extraction_plan:plan,effective_settings:{mode:'Fixed Rank',rank:17},review_only:true})]});
    assert.match(n.lboExtractStatus.value,/manual values/);
    assert.match(n.lboExtractStatus.value,/analyzed base/);assert.match(n.lboExtractStatus.value,/no file saved/);
    locale='ja';listeners['Comfy.Locale.change']();
    labels.length=0;action.draw(ctx,n,600,100);
    assert.ok(labels.includes('次のRun：保存せず確認'));assert.ok(labels.includes('✓ Review Only'));
    assert.match(n.lboExtractStatus.value,/推奨抽出設定/);assert.match(n.lboExtractStatus.value,/保存していません/);
    assert.match(n.lboExtractStatus.value,/未測定/);
    n.inputs[0].link=null;n.onConnectionsChange();
    assert.equal(find('rank').type,'combo');assert.equal(find('plan_action').type,'lbo_hidden');
    assert.match(n.lboExtractStatus.value,/手動抽出/);
    assert.equal(JSON.stringify(n.widgets.filter(w=>w.serialize!==false).map(w=>[w.name,w.value])),before);
    // A completed Inspect is advisory until the explicit button is pressed.
    const source={properties:{},widgets:[{name:'base_model',value:plan.base_model},{name:'modified_model',value:plan.modified_model}]};
    const setPlan=(values=true, version=2)=>source.properties.extract_inspection_result=JSON.stringify({values_checked:values,recommendation:{...plan,version}});
    n.graph={links:{42:{origin_id:3}},getNodeById:()=>source,change(){}};
    n.inputs[0].link=42;n.onConnectionsChange();
    const apply=find('Apply Recommendations');
    setPlan(false);apply.callback();assert.equal(find('rank').value,17);assert.match(n.lboExtractStatus.value,/Full Scan/);
    setPlan(true,1);apply.callback();assert.equal(find('rank').value,17);
    setPlan();
    n.inputs.push({name:'dtype',link:99});apply.callback();assert.equal(find('base_model').value,'manual base');
    n.inputs.pop();
    source.widgets[0].value='changed';apply.callback();assert.equal(find('base_model').value,'manual base');
    source.widgets[0].value=plan.base_model;
    apply.callback();
    assert.equal(find('rank').value,32);assert.equal(find('dtype').value,'bf16');assert.equal(find('max_rank').value,64);
    assert.equal(find('base_model').value,plan.base_model);assert.equal(find('filename_prefix').value,'manual-name');
    assert.equal(find('control_after_generate').value,'fixed');
    const preset=find('Rank Preset');
    for(const rank of [16,32,64,256]) {
        preset.callback(String(rank));assert.equal(find('rank').value,rank);
        n.onConfigure({});setPlan();n.onConnectionsChange();assert.equal(find('rank').value,rank);
    }
    find('rank').value=17;find('rank').callback();assert.equal(preset.value,'Custom');
    n.inputs.push({name:'rank',link:100});preset.callback('64');assert.equal(find('rank').value,17);n.inputs.pop();
    find('mode').value='Auto Rank';find('mode').callback();assert.equal(preset.type,'lbo_hidden');
    assert.equal(JSON.stringify(n.widgets.filter(w=>w.serialize!==false).map(w=>w.name)),JSON.stringify(JSON.parse(before).map(([name])=>name)));
});
