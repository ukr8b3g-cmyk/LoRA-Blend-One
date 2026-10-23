import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { randomUUID } from 'node:crypto';
import { setup } from './helpers/recipe_ui.mjs';

function fileFromHeader(header, name='blend.safetensors') {
    const bytes=new TextEncoder().encode(JSON.stringify(header)), prefix=new ArrayBuffer(8);
    new DataView(prefix).setBigUint64(0,BigInt(bytes.length),true);
    const blob=new Blob([prefix,bytes,new Uint8Array(4096)]), ranges=[];
    return {name,size:blob.size,ranges,slice(start,end) {ranges.push([start,end]);return blob.slice(start,end);}};
}

function fixture() {
    const s=setup(), notices=[],delegated=[],loads=[],temporary=[];
    const raw={recipe_version:'1.0',settings:{output_rank:7},sources:[{name:'a.safetensors',weight:.5},{name:'b.safetensors',weight:-.25}]};
    s.setPayload({...s.getPayload(),file:'blend.safetensors'});
    const originalSnapshot=JSON.stringify({source:s.source.serialize(),blend:s.blend.serialize()});
    const tabs=[{id:'original',data:originalSnapshot}];
    s.app.extensionManager={toast:{add:notice=>notices.push(notice)}};
    s.app.handleFile=function(file,...args) {delegated.push({file,args,receiver:this});return 'native handler';};
    s.app.loadGraphData=async function(data,clean,view,name) {
        loads.push({data,clean,view,name});tabs.push({id:data.id,data:structuredClone(data)});return true;
    };
    class Graph {
        constructor() {this.id=randomUUID();this.nodes=new Map();this.links=new Map();temporary.push(this);}
        add(node) {node.id=this.nodes.size+1;node.graph=this;this.nodes.set(node.id,node);}
        beforeChange() {} afterChange() {} incrementVersion() {} setDirtyCanvas() {}
        getNodeById(id) {return this.nodes.get(id);}
        serialize() {return {id:this.id,version:.4,last_node_id:2,last_link_id:1,
            nodes:[...this.nodes.values()].map(node=>({...node.serialize(),id:node.id,type:node.type,pos:node.pos,size:node.size,inputs:node.inputs,
                widgets_values_named:Object.fromEntries(node.widgets.filter(w=>w.serialize!==false).map(w=>[w.name,w.value]))})),
            links:[...this.links.values()].map(link=>[link.id,link.origin_id,0,link.target_id,0,'LORA_BLEND_SOURCES'])};}
        clear() {for(const node of this.nodes.values()) {node.onRemoved?.();node.inputs.forEach(input=>input.link=null);node.graph=null;}this.nodes.clear();this.links.clear();this.cleared=true;}
    }
    const liteGraph={LGraph:Graph,createNode(type) {
        const node=type==='LoRABlendSources'?new s.Source(null):new s.Blend(null);
        node.connect=function(output,target,input) {
            if(this.graph!==target.graph||input<0)return null;
            const link={id:1,origin_id:this.id,target_id:target.id};this.graph.links.set(1,link);target.inputs[input].link=1;return link;
        };
        return node;
    }};
    Object.assign(s.context,{LiteGraph:liteGraph,crypto:{randomUUID},structuredClone,TextEncoder});
    const code=fs.readFileSync(new URL('../web/recipe_drop.js',import.meta.url),'utf8').replace(/^import .*;\r?\n/gm,'').replace(/^export /gm,'');
    vm.runInContext(`{${code}\nObject.assign(globalThis,{droppedRecipe,recipeWorkflow});}`,s.context);
    s.extensions.find(e=>e.name==='lora_blend_one.recipe_drop').setup();
    return {...s,notices,delegated,loads,tabs,temporary,raw,originalSnapshot,liteGraph,
        file:()=>fileFromHeader({__metadata__:{lbo_recipe_json:JSON.stringify(raw)}}),
        assertOriginal() {assert.equal(JSON.stringify({source:s.source.serialize(),blend:s.blend.serialize()}),originalSnapshot);assert.equal(s.graph.before,0);assert.equal(tabs[0].data,originalSnapshot);},
    };
}

test('reads only the dropped header and sends only recipe metadata to the local API',async()=>{
    const s=fixture(),file=s.file();await s.app.handleFile(file);
    assert.equal(file.ranges.length,2);assert.deepEqual(file.ranges[0],[0,8]);
    assert.ok(file.ranges[1][1]<file.size);
    const request=s.requests.find(r=>r.url.endsWith('recipe_from_metadata'));
    assert.deepEqual(JSON.parse(request.options.body),{filename:file.name,recipe:s.raw});
    assert.equal(s.delegated.length,0);s.assertOriginal();
});

test('opens connected Sources and Blend One in a fresh tab with restored coefficients, settings and fixed seed',async()=>{
    const s=fixture();await s.app.handleFile(s.file());
    assert.equal(s.notices.length,0);assert.equal(s.loads.length,1);assert.equal(s.tabs.length,2);
    const {data,clean,view,name}=s.loads[0];assert.equal(clean,true);assert.equal(view,true);assert.ok(name.includes(data.id));
    assert.deepEqual(data.nodes.map(n=>n.type),['LoRABlendSources','LoRABlendOneSplit']);
    const [source,blend]=data.nodes,weights=source.widgets_values_named,settings=blend.widgets_values_named;
    assert.equal(weights.lora_1,'moss/a.safetensors');assert.equal(weights.weight_1,50);assert.equal(weights.weight_2,-25);
    assert.equal(weights.enabled_1,true);assert.equal(weights.enabled_3,false);
    assert.equal(settings.output_rank,'Custom');assert.equal(settings.custom_rank,7);assert.equal(settings.weight_mode,'direct');
    assert.equal(settings.seed,17);assert.equal(settings.control_after_generate,'fixed');
    assert.equal(settings.output_dtype,'bf16');assert.equal(settings.output_scale,1.2);
    assert.equal(blend.properties.lora_blend_advanced_open,true);
    assert.equal(data.links.length,1);assert.equal(blend.inputs[0].link,1);
    assert.deepEqual(blend.properties.lora_blend_recipe_import.recipe,s.raw);
    assert.ok(source.pos[0]<blend.pos[0]);assert.ok(s.temporary.every(g=>g.cleared));s.assertOriginal();
});

test('dropping the same LoRA again creates another unique tab',async()=>{
    const s=fixture();await s.app.handleFile(s.file());await s.app.handleFile(s.file());
    assert.equal(s.tabs.length,3);assert.notEqual(s.loads[0].data.id,s.loads[1].data.id);assert.notEqual(s.loads[0].name,s.loads[1].name);s.assertOriginal();
});

test('missing or hash-mismatched sources retain recipe names and weights with every slot OFF',async()=>{
    const s=fixture(),payload=s.getPayload();payload.ready=false;payload.sources[1].selected=null;payload.sources[1].status='hash_mismatch';s.setPayload(payload);
    await s.app.handleFile(s.file());assert.equal(s.loads.length,1);
    const source=s.loads[0].data.nodes[0].widgets_values_named;
    assert.equal(source.lora_2,'b.safetensors');assert.equal(source.weight_2,-25);
    for(let i=1;i<=8;i++)assert.equal(source[`enabled_${i}`],false);
    s.assertOriginal();
});

test('non-recipe LoRAs, images and workflows delegate to the existing handler with the original arguments',async()=>{
    const s=fixture(),file=fileFromHeader({__metadata__:{ss_network_dim:'16'}});
    const image={name:'sample.png'},workflow={name:'workflow.json'};
    for(const f of [file,image,workflow])assert.equal(await s.app.handleFile(f,'drop',{test:true}),'native handler');
    assert.equal(s.loads.length,0);assert.equal(s.requests.length,0);assert.equal(s.notices.length,0);assert.equal(s.delegated.length,3);
    assert.equal(s.delegated[0].receiver,s.app);assert.deepEqual(s.delegated[0].args,['drop',{test:true}]);s.assertOriginal();
});

test('invalid settings or unavailable API leave the existing workflow intact and show an actionable error',async()=>{
    const s=fixture(),payload=s.getPayload();payload.issues=['seed: -1'];payload.ready=false;s.setPayload(payload);
    await s.app.handleFile(s.file());assert.equal(s.loads.length,0);assert.match(s.notices[0].detail,/seed: -1/);
    s.setLocale('ja');s.api.fetchApi=async()=>({ok:false,status:404});await s.app.handleFile(s.file());
    assert.equal(s.loads.length,0);assert.match(s.notices.at(-1).detail,/再起動/);assert.match(s.notices.at(-1).summary,/復元できません/);s.assertOriginal();
});

test('missing node types clean up detached nodes without opening or replacing a tab',async()=>{
    const s=fixture(),create=s.liteGraph.createNode;s.liteGraph.createNode=type=>type==='LoRABlendOneSplit'?null:create(type);
    await s.app.handleFile(s.file());assert.equal(s.loads.length,0);assert.ok(s.temporary.every(g=>g.cleared));assert.match(s.notices[0].detail,/nodes are unavailable/);s.assertOriginal();
});

test('invalid safetensors lengths do not cause oversized reads, and malformed recipe JSON does not replace a workflow',async()=>{
    const s=fixture(),prefix=new ArrayBuffer(8);new DataView(prefix).setBigUint64(0,2n**60n,true);
    const blob=new Blob([prefix,new Uint8Array(10)]),ranges=[];
    const invalid={name:'invalid.safetensors',size:blob.size,slice(a,b){ranges.push([a,b]);return blob.slice(a,b);}};
    await s.app.handleFile(invalid);assert.deepEqual(ranges,[[0,8]]);assert.equal(s.delegated.length,1);
    await s.app.handleFile(fileFromHeader({__metadata__:{lbo_recipe_json:'{bad json'}}));
    assert.equal(s.loads.length,0);assert.equal(s.notices.length,1);s.assertOriginal();
});
