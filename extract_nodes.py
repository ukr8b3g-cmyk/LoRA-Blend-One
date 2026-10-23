import json
import math
import time
from pathlib import Path
import folder_paths
import comfy.model_management
import comfy.utils
from safetensors import safe_open
from .extractor.engine import extract
from .extractor.model_files import make_plan, read_entry, identity
from .extractor.writer import finite


def models():
    return [f'{category}: {name}' for category in ('diffusion_models', 'checkpoints')
            for name in folder_paths.get_filename_list(category) if name.lower().endswith('.safetensors')]


def resolve(value):
    category, name = value.split(': ', 1)
    if category not in ('diffusion_models', 'checkpoints') or not name.lower().endswith('.safetensors'):
        raise ValueError('Select a registered, non-quantized Safetensors model')
    return folder_paths.get_full_path_or_raise(category, name)


class LoRABlendModelInspect:
    CATEGORY = 'model/merging/LoRA Blend One'
    RETURN_TYPES = ('STRING',)
    FUNCTION = 'run'
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        choices = models()
        return {'required': {'base_model': (choices,), 'modified_model': (choices,)},
                'optional': {'scan': (['Header', 'Full Scan'], {'default': 'Header', 'advanced': True})}}

    @classmethod
    def IS_CHANGED(cls, base_model, modified_model, **kwargs):
        return str((identity(resolve(base_model)), identity(resolve(modified_model))))

    def run(self, base_model, modified_model, scan='Header'):
        a, b = make_plan(resolve(base_model), resolve(modified_model))
        result = {'architecture': 'MOSS-Image', 'parameters': len(a['entries']), 'ignored_buffers': [a['ignored'], b['ignored']],
                  'scan': scan, 'values_checked': scan == 'Full Scan'}
        result['largest_fp32_tensor_bytes'] = max(math.prod(e['shape'])*4 for e in a['entries'].values())
        result['full_delta_bf16_bytes_upper_bound'] = sum(math.prod(e['shape'])*2 for e in a['entries'].values())
        if scan == 'Full Scan':
            changed = non_matrix = 0
            with safe_open(str(a['path']), framework='pt', device='cpu') as ha, safe_open(str(b['path']), framework='pt', device='cpu') as hb:
                for key, entry in a['entries'].items():
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    x, y = read_entry(ha, entry), read_entry(hb, b['entries'][key])
                    finite(x, key); finite(y, key)
                    if not x.equal(y):
                        changed += 1
                        non_matrix += not (key.endswith('.weight') and x.ndim == 2)
            result.update(changed=changed, changed_non_matrix=non_matrix)
        text = json.dumps(result, ensure_ascii=False, indent=2)
        return {'ui': {'text': [text]}, 'result': (text,)}


class LoRABlendModelExtract(LoRABlendModelInspect):
    @classmethod
    def INPUT_TYPES(cls):
        choices = models()
        return {'required': {'base_model': (choices,), 'modified_model': (choices,),
                'mode': (['Fixed Rank', 'Auto Rank', 'Full Delta'],), 'rank': ('INT', {'default': 16, 'min': 1, 'max': 4096}),
                'energy': ('FLOAT', {'default': .99, 'min': .01, 'max': 1., 'step': .001}),
                'dtype': (['bf16', 'fp16', 'fp32'],), 'filename_prefix': ('STRING', {'default': 'loras/LoRA_Extract'})},
                'optional': {'max_rank': ('INT', {'default': 256, 'min': 1, 'max': 4096, 'advanced': True}),
                'device': (['auto', 'cpu', 'cuda'], {'advanced': True}), 'svd': (['auto', 'exact', 'randomized'], {'advanced': True}),
                'non_lora': (['Stop', 'Store Diff', 'Exclude With Report'], {'advanced': True}),
                'key_filter': ('STRING', {'default': '*', 'advanced': True}),
                'seed': ('INT', {'default': 0, 'min': 0, 'max': 0x7fffffffffffffff, 'advanced': True}),
                'hash_inputs': ('BOOLEAN', {'default': True, 'advanced': True}),
                'write_report': ('BOOLEAN', {'default': True, 'advanced': True})}}

    def run(self, base_model, modified_model, mode, rank, energy, dtype, filename_prefix, **kwargs):
        root = Path(folder_paths.get_output_directory()).resolve()
        prefix = (root / filename_prefix).resolve()
        if not prefix.is_relative_to(root) or prefix == root:
            raise ValueError('Filename Prefix must stay inside ComfyUI/output')
        bar = comfy.utils.ProgressBar(1)
        started = time.monotonic()
        def progress(stage, key, done, total):
            bar.update_absolute(done, max(1, total))
            try:
                from server import PromptServer
                from comfy_execution.utils import get_executing_context
            except ImportError:
                return
            context = get_executing_context()
            if context is not None and PromptServer.instance.client_id is not None:
                PromptServer.instance.send_sync('lora_blend_one.extract_status',
                    {'node_id': str(context.node_id), 'text': f'{stage} {done}/{total} | {time.monotonic()-started:.1f}s\n{key}'}, PromptServer.instance.client_id)
        try:
            output, report = extract(resolve(base_model), resolve(modified_model), prefix,
                                     mode=mode, rank=rank, energy=energy, dtype=dtype,
                                     cancel=comfy.model_management.throw_exception_if_processing_interrupted,
                                     progress=progress, **kwargs)
        except Exception as error:
            progress('Stopped', f'{base_model} -> {modified_model}\n{error}', 0, 1)
            raise
        text = f'{output or report.get("status")} | {report["artifact_type"]} | layers={len(report["layers"])}'
        text += f' | weight_error={100*report["relative_error_all_parameters"]:.2f}%'
        if report['artifact_type'] != 'standard_lora':
            text += ' | ComfyUI extension: not a Blend One source'
        if report.get('partial'):
            text += f' | PARTIAL: excluded={len(report["excluded"])}'
        if report['warnings']:
            text += '\n' + '\n'.join(report['warnings'])
        return {'ui': {'text': [text]}, 'result': (text,)}


EXTRACT_NODES = {cls.__name__: cls for cls in (LoRABlendModelInspect, LoRABlendModelExtract)}
EXTRACT_NAMES = {'LoRABlendModelInspect': 'LoRA Blend Model Inspect', 'LoRABlendModelExtract': 'LoRA Blend Model Extract'}
