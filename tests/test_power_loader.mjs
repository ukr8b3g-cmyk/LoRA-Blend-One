import test from 'node:test';
import assert from 'node:assert/strict';
import { setup } from './helpers/power_loader_ui.mjs';

function fixture(rows = [
    {on:false,lora:'missing/off.safetensors',strength:.7},
    {on:true,lora:'moss/a.safetensors',strength:.8},
    {on:true,lora:'moss/b.safetensors',strength:.2},
]) {
    const s=setup();
    for (let i=1;i<=8;i++) {
        s.source.widgets.find(w=>w.name===`lora_${i}`).options.values=['(none)','moss/a.safetensors','moss/b.safetensors'];
        Object.assign(s.source.widgets.find(w=>w.name===`weight_${i}`).options,{min:-400,max:400});
    }
    const loader={id:8,type:'Power Lora Loader (rgthree)',title:'My LoRAs',properties:{'Show Strengths':'Single Strength'},
        inputs:[{name:'model',link:null},{name:'clip',link:null}],
        widgets:rows.map((value,i)=>({name:`lora_${i+1}`,value})),
    };
    s.graph.nodes.set(loader.id,loader);
    const [select,button,details]=s.source.widgets.filter(w=>w.loraBlendAdvanced);
    const mode=s.blend.widgets.find(w=>w.name==='weight_mode');
    mode.value='normalize';mode.options.values=['normalize','direct'];
    const header=s.source.widgets.find(w=>w.type==='button'&&!w.loraBlendAdvanced);
    const choose=()=>select.callback(select.options.values()[1]);
    return {...s,loader,select,button,details,header,choose};
}

test('import controls stay in Advanced, follow locale and never shift serialized source values',()=>{
    const s=fixture(), before=s.source.serializeFromStoreState();
    assert.ok([s.select,s.button,s.details].every(w=>w.hidden&&w.serialize===false));
    assert.equal(s.details.element.style.display,'none');
    s.header.callback();
    assert.ok([s.select,s.button,s.details].every(w=>!w.hidden));
    assert.equal(s.button.name,'Import Enabled LoRAs');
    assert.deepEqual([...s.select.options.values()],['(none)','#8 My LoRAs']);
    s.choose();
    assert.equal(s.source.properties.lora_blend_power_loader_id,8);
    assert.deepEqual(s.source.serialize().widgets_values,before.widgets_values);
    s.setLocale('ja');
    assert.equal(s.button.name,'ONのLoRA設定を取り込む');
    assert.match(s.details.element.value,/最大8本/);
    const copy=new s.Source();copy.configure(s.source.serialize());
    assert.equal(copy.properties.lora_blend_power_loader_id,8);
    assert.equal(copy.widgets.find(w=>w.loraBlendAdvanced).value,'#8 My LoRAs');
    assert.deepEqual(copy.serialize().widgets_values,before.widgets_values);
});

test('ON rows copy in displayed order with direct percentages, clear unused slots and leave the loader unchanged',()=>{
    const s=fixture(), loaderBefore=JSON.stringify(s.loader), blendBefore=s.blend.serialize().widgets_values;
    s.choose();s.button.callback();
    assert.equal(s.value(s.source,'lora_1'),'moss/a.safetensors');
    assert.equal(s.value(s.source,'lora_2'),'moss/b.safetensors');
    assert.equal(s.value(s.source,'weight_1'),80);assert.equal(s.value(s.source,'weight_2'),20);
    assert.equal(s.value(s.source,'enabled_1'),true);assert.equal(s.value(s.source,'enabled_2'),true);
    for(let i=3;i<=8;i++) {
        assert.equal(s.value(s.source,`lora_${i}`),'(none)');
        assert.equal(s.value(s.source,`weight_${i}`),0);assert.equal(s.value(s.source,`enabled_${i}`),false);
    }
    assert.equal(s.value(s.blend,'weight_mode'),'direct');
    const expected=[...blendBefore];expected[3]='direct'; // output rank, filename, method, weight mode.
    assert.deepEqual([...s.blend.serialize().widgets_values],expected);
    assert.equal(JSON.stringify(s.loader),loaderBefore);
    assert.match(s.details.element.value,/Imported 2 ON LoRAs/);
    assert.match(s.details.element.value,/1 OFF rows skipped/);
    s.loader.widgets[1].value.strength=.4;
    assert.equal(s.value(s.source,'weight_1'),80);
    s.button.callback();assert.equal(s.value(s.source,'weight_1'),40);
    s.setLocale('ja');assert.match(s.details.element.value,/ONの2本/);
    const values=s.source.serialize().widgets_values, copy=new s.Source();copy.configure(s.source.serialize());
    assert.deepEqual(copy.serializeFromStoreState().widgets_values,values);
});

test('eight enabled rows among ten fit, and zero/negative strengths retain their values',()=>{
    const strengths=[.8,.2,.2,.05,.1,.2,0,-.25];
    const rows=strengths.map(strength=>({on:true,lora:'moss/a.safetensors',strength}));
    rows.splice(1,0,{on:false,lora:'off-a.safetensors',strength:1});
    rows.push({on:false,lora:'off-b.safetensors',strength:1});
    const s=fixture(rows);s.choose();s.button.callback();
    assert.match(s.details.element.value,/Imported 8 ON LoRAs/);
    assert.match(s.details.element.value,/2 OFF rows skipped/);
    strengths.forEach((strength,i)=>{
        assert.equal(s.value(s.source,`weight_${i+1}`),strength*100);
        assert.equal(s.value(s.source,`enabled_${i+1}`),true);
    });
});

test('too many or no enabled rows fail before changing source or blend settings',()=>{
    for(const rows of [Array.from({length:9},()=>({on:true,lora:'moss/a.safetensors',strength:.2})),
        [{on:false,lora:'missing.safetensors',strength:1}]]) {
        const s=fixture(rows);s.choose();const a=s.source.serialize(),b=s.blend.serialize(),changes=s.graph.before;
        s.button.callback();
        assert.deepEqual(s.source.serialize(),a);assert.deepEqual(s.blend.serialize(),b);
        assert.equal(s.graph.before,changes);assert.match(s.details.element.value,/exceed|No ON LoRAs/);
    }
});

test('full paths match across slash styles; unique basenames resolve; ambiguous or missing names do not import partially',()=>{
    const s=fixture([{on:true,lora:'moss\\a.safetensors',strength:.5},{on:true,lora:'b.safetensors',strength:.1}]);
    s.choose();s.button.callback();
    assert.equal(s.value(s.source,'lora_1'),'moss/a.safetensors');
    assert.equal(s.value(s.source,'lora_2'),'moss/b.safetensors');
    for(const name of ['missing.safetensors','a.safetensors']) {
        s.loader.widgets[1].value.lora=name;
        s.source.widgets.find(w=>w.name==='lora_2').options.values.push('other/a.safetensors');
        const before=s.source.serialize(),changes=s.graph.before;s.button.callback();
        assert.deepEqual(s.source.serialize(),before);assert.equal(s.graph.before,changes);
        assert.match(s.details.element.value,/file not found|ambiguous filename/);
    }
});

test('invalid strengths and unequal connected CLIP strengths fail atomically; single/model-only strengths import',()=>{
    for(const strength of [NaN,Infinity,'0.8',5]) {
        const s=fixture([{on:true,lora:'moss/a.safetensors',strength}]);s.choose();
        const before=s.source.serialize(),changes=s.graph.before;s.button.callback();
        assert.deepEqual(s.source.serialize(),before);assert.equal(s.graph.before,changes);
        assert.match(s.details.element.value,/invalid strength|outside this Sources/);
    }
    const s=fixture([{on:true,lora:'moss/a.safetensors',strength:.8,strengthTwo:.2}]);
    s.loader.properties['Show Strengths']='Separate Model & Clip';s.loader.inputs[1].link=9;s.choose();
    const before=s.source.serialize(),changes=s.graph.before;s.button.callback();
    assert.deepEqual(s.source.serialize(),before);assert.equal(s.graph.before,changes);
    assert.match(s.details.element.value,/different Model\/CLIP strengths/);
    s.setLocale('ja');assert.match(s.details.element.value,/異なる強度/);
    s.loader.inputs[1].link=null;s.button.callback();assert.equal(s.value(s.source,'weight_1'),80);
    s.loader.inputs[1].link=9;s.loader.properties['Show Strengths']='Single Strength';
    s.button.callback();assert.match(s.details.element.value,/ONの1本/);
});

test('linked or missing controls and disconnected consumers reject the import before any values change',()=>{
    const cases=[s=>s.source.inputs.push({name:'weight_2',link:6}),
        s=>s.blend.inputs.push({name:'weight_mode',link:6}),s=>s.graph.links.clear(),
        s=>s.source.widgets.splice(s.source.widgets.findIndex(w=>w.name==='enabled_7'),1)];
    for(const change of cases) {
        const s=fixture();s.choose();change(s);const a=s.source.serialize(),b=s.blend.serialize(),changes=s.graph.before;
        s.button.callback();assert.deepEqual(s.source.serialize(),a);assert.deepEqual(s.blend.serialize(),b);
        assert.equal(s.graph.before,changes);assert.match(s.details.element.value,/connected input|Connect Sources/);
    }
});

test('multiple connected blends all use direct, while stale or other-workflow loaders cannot be imported',()=>{
    const s=fixture();s.choose();
    const other={id:11,type:'LoRABlendOneSplit',inputs:[{name:'sources',link:1}],widgets:[{name:'weight_mode',value:'normalize'}]};
    s.graph.nodes.set(other.id,other);s.button.callback();
    assert.equal(other.widgets[0].value,'direct');assert.match(s.details.element.value,/2 connected Blend One/);
    s.graph.nodes.delete(s.loader.id);const before=s.source.serialize(),changes=s.graph.before;s.button.callback();
    assert.deepEqual(s.source.serialize(),before);assert.equal(s.graph.before,changes);
    assert.match(s.details.element.value,/Select a Power LoRA Loader/);
    assert.deepEqual([...s.select.options.values()],['(none)']);
    s.graph.nodes.set(s.loader.id,s.loader);s.app.graph={};s.button.callback();
    assert.equal(s.graph.before,changes);assert.match(s.details.element.value,/current workflow/);
});
