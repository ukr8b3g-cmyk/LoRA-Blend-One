import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

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
        addDOMWidget(name,type,element){const w={name,type,element};this.widgets.push(w);return w;}
        computeSize(){return [340,200];} setSize(size){this.size=size;}
    }
    extension.beforeRegisterNodeDef(Node,{name:'LoRABlendModelExtract',input:{optional:{seed:['INT',{advanced:true}]}}});
    const n=new Node();n.comfyClass='LoRABlendModelExtract';n.onNodeCreated();
    const find=name=>n.widgets.find(w=>w.name===name);
    assert.equal(find('rank').type,'number');assert.equal(find('energy').type,'lbo_hidden');
    assert.equal(find('seed').type,'lbo_hidden');
    const button=n.widgets.find(w=>w.type==='button');button.callback();assert.equal(find('seed').type,'number');
    find('mode').value='Auto Rank';find('mode').callback();assert.equal(find('rank').type,'lbo_hidden');assert.equal(find('energy').type,'number');
    locale='ja';events['Comfy.Locale.change']();assert.match(button.name,/詳細設定/);
    assert.equal(find('seed').value,16);assert.equal(button.serialize,false);
    app.graph={getNodeById:()=>n};apiEvents['lora_blend_one.extract_status']({detail:{node_id:'1',text:'Saved'}});
    assert.equal(n.lboExtractStatus.value,'Saved');n.onExecuted({text:['result']});assert.equal(n.lboExtractStatus.value,'result');
});
