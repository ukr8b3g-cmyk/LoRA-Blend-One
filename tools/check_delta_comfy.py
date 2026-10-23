"""Check an extracted file against native MOSS loader and actual base weights on CPU."""
import importlib
import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from safetensors import safe_open
from safetensors.torch import load_file

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file',type=Path,required=True)
    parser.add_argument('--comfy-root',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args()
    output,root,base=args.file,args.comfy_root,args.base
    sys.path.insert(0,str(root)); sys.argv = ['check-delta','--cpu']
    importlib.import_module('comfy.options').enable_args_parsing()
    lora = importlib.import_module('comfy.lora')
    model_base = importlib.import_module('comfy.model_base')
    with safe_open(str(base), framework='pt', device='cpu') as f:
        header = {'diffusion_model.'+k: torch.empty(tuple(f.get_slice(k).get_shape()),device='meta')
                  for k in f.keys() if k!='rope.bands'}
    class HeaderModel(model_base.MOSSImage):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.model_config=SimpleNamespace(unet_config={'image_model':'moss_image'})
        def state_dict(self): return header
    key_map=lora.model_lora_keys_unet(HeaderModel(),{})
    tensors=load_file(str(output))
    missing=[]
    class Capture(logging.Handler):
        def emit(self,record):
            if 'lora key not loaded' in record.getMessage(): missing.append(record.getMessage())
    handler=Capture(); logging.getLogger().addHandler(handler)
    try: patches=lora.load_lora(tensors,key_map)
    finally: logging.getLogger().removeHandler(handler)
    assert not missing, missing
    report=json.loads(output.with_suffix('.safetensors.report.json').read_text(encoding='utf-8'))
    assert len(patches)==len(report['layers']),(len(patches),len(report['layers']))
    applied=[]
    with safe_open(str(base),framework='pt',device='cpu') as f:
        for target in ('diffusion_model.layers.0.proj.q','diffusion_model.layers.0.proj.o','diffusion_model.layers.0.norm1'):
            if target+'.lora_A.weight' in tensors:
                adapter=patches[key_map[target]]
                destination=key_map[target]
                key,offset=(destination,None) if isinstance(destination,str) else destination
                weight=f.get_tensor(key.removeprefix('diffusion_model.')).float()
                if offset: weight=weight[offset[1]:offset[1]+offset[2]]
                expected=weight + tensors[target+'.lora_B.weight'].float() @ tensors[target+'.lora_A.weight'].float()
                actual=adapter.calculate_weight(weight.clone(),key,1.,1.,None,lambda x:x)
                torch.testing.assert_close(actual,expected)
                applied.append(target)
            elif target+'.diff' in tensors:
                patch=patches[key_map[target]]
                assert patch[0]=='diff'
                weight=f.get_tensor(target.removeprefix('diffusion_model.')+'.weight').float()
                actual=lora.calculate_weight([(1.,patch,1.,None,None)],weight.clone(),target+'.weight')
                torch.testing.assert_close(actual,weight+tensors[target+'.diff'].float())
                applied.append(target)
    result={'file':str(output),'bytes':output.stat().st_size,'loaded_layers':len(patches),'unloaded_keys':missing,
            'native_lora_weight_apply':applied,'artifact_type':report['artifact_type'],'gpu_generation':False}
    print(json.dumps(result,indent=2))
    output.with_suffix('.comfy-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
