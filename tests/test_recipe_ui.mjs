import test from 'node:test';
import assert from 'node:assert/strict';
import { setup } from './helpers/recipe_ui.mjs';

test('recipe controls stay inside Advanced, display both languages, and do not shift saved values',async()=>{
    const s=setup(), before=s.blend.serialize().widgets_values;
    assert.equal(s.controls.length,4);
    assert.ok(s.controls.every(w=>w.hidden&&w.serialize===false));
    assert.equal(s.details.element.style.display,'none');
    s.header.callback();
    assert.ok(s.controls.every(w=>!w.hidden));
    await s.choose();
    assert.match(s.details.element.value,/a.safetensors/);
    assert.match(s.details.element.value,/Coefficient: 0.5/);
    assert.match(s.details.element.value,/SHA256 matched/);
    s.setLocale('ja');
    assert.equal(s.select.name,'保存済みブレンドLoRA');
    assert.match(s.details.element.value,/合成係数: -0.25/);
    assert.match(s.details.element.value,/SHA256一致/);
    assert.deepEqual(s.blend.serializeFromStoreState().widgets_values,before);
    const info=s.blend.serialize(), restored=s.newBlend();restored.configure(info);
    assert.deepEqual(restored.serialize().widgets_values,before);
    assert.equal(restored.properties.lora_blend_recipe_file,'moss/blend.safetensors');
    s.header.callback();
    assert.ok(s.controls.every(w=>w.hidden));
});

test('restore applies direct coefficients and settings to the connected Sources, clears unused slots, fixes seed and preserves filename',async()=>{
    const s=setup();await s.choose();await s.restore.callback();
    assert.equal(s.value(s.source,'lora_1'),'moss/a.safetensors');
    assert.equal(s.value(s.source,'weight_1'),50);
    assert.equal(s.value(s.source,'weight_2'),-25);
    assert.equal(s.value(s.source,'enabled_1'),true);
    for(let i=3;i<=8;i++) {
        assert.equal(s.value(s.source,`lora_${i}`),'(none)');assert.equal(s.value(s.source,`enabled_${i}`),false);
    }
    assert.equal(s.value(s.blend,'weight_mode'),'direct');
    assert.equal(s.value(s.blend,'output_rank'),'Custom');
    assert.equal(s.value(s.blend,'custom_rank'),7);
    assert.equal(s.value(s.blend,'output_scale'),1.2);
    assert.equal(s.value(s.blend,'seed'),17);
    assert.equal(s.value(s.blend,'control_after_generate'),'fixed');
    assert.equal(s.value(s.blend,'filename_prefix'),'keep/my_output');
    assert.equal(s.graph.before,1);assert.equal(s.graph.after,1);
    assert.match(s.details.element.value,/Restored Sources/);
    const info=s.blend.serialize(), next=s.newBlend();next.configure(info);
    assert.deepEqual(next.serialize().widgets_values,info.widgets_values);
});

test('missing/hash-mismatched sources are shown and restore rechecks files before modifying either node',async()=>{
    const s=setup(), before=s.source.serialize().widgets_values;
    await s.choose();
    const payload=s.getPayload();payload.ready=false;payload.sources[0].selected=null;payload.sources[0].status='hash_mismatch';s.setPayload(payload);
    await s.restore.callback();
    assert.deepEqual(s.source.serialize().widgets_values,before);
    assert.equal(s.graph.before,0);
    assert.match(s.details.element.value,/SHA256 mismatch/);
    assert.match(s.details.element.value,/Cannot restore/);
});

test('restore does not overwrite linked settings or unrelated source providers',async()=>{
    const s=setup();await s.choose();
    s.blend.inputs.push({name:'seed',link:10});await s.restore.callback();
    assert.equal(s.graph.before,0);assert.match(s.details.element.value,/connected or unavailable/);
    s.blend.inputs.pop();s.source.type='AnotherSourceProvider';await s.restore.callback();
    assert.equal(s.graph.before,0);assert.match(s.details.element.value,/Connect a LoRA Blend Sources/);
});

test('an earlier inspection cannot replace a newer selection',async()=>{
    const s=setup();let resolve;
    s.setPending(new Promise(r=>resolve=r));
    s.select.value='moss/blend.safetensors';const earlier=s.select.callback(s.select.value);
    await new Promise(setImmediate);
    s.select.value='(none)';await s.select.callback(s.select.value);
    resolve(s.getPayload());await earlier;
    assert.match(s.details.element.value,/Select a saved blend/);
    await s.restore.callback();assert.equal(s.graph.before,0);
});

test('a file without recipe has actionable English and Japanese messages and cannot restore',async()=>{
    const s=setup();s.setPayload({error:'No recipe',code:'no_recipe'});await s.choose();
    assert.match(s.details.element.value,/no LoRA Blend One recipe/);
    s.setLocale('ja');assert.match(s.details.element.value,/レシピはありません/);
    await s.restore.callback();assert.equal(s.graph.before,0);
});

test('a dropped recipe remains inspectable and refreshable without installing the blend file',async()=>{
    const s=setup(), raw={recipe_version:'1.0',sources:[],settings:{}};
    s.blend.loraBlendRecipe.setImported(raw,{...s.getPayload(),file:'from_downloads.safetensors'});
    assert.ok(s.select.options.values.includes(s.select.value));
    const info=s.blend.serialize();
    assert.equal(info.properties.lora_blend_recipe_import.recipe,raw);
    await s.read.callback();
    const request=s.requests.find(r=>r.url.endsWith('recipe_from_metadata'));
    assert.deepEqual(JSON.parse(request.options.body),{filename:'from_downloads.safetensors',recipe:raw});
    assert.match(s.details.element.value,/a.safetensors/);
    const copy=s.newBlend();copy.configure(info);
    await new Promise(setImmediate);
    assert.equal(copy.properties.lora_blend_recipe_import.filename,'from_downloads.safetensors');
    assert.equal(s.graph.before,0);
});
