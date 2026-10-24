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
    RETURN_TYPES = ('STRING', 'LBO_EXTRACTION_PLAN')
    RETURN_NAMES = ('info', 'extraction_plan')
    FUNCTION = 'run'
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        choices = models()
        return {'required': {'base_model': (choices,), 'modified_model': (choices,)},
                'optional': {'scan': (['Header', 'Full Scan'], {'default': 'Full Scan', 'advanced': True}),
                             'purpose': (['ComfyUI Loading', 'Blend One Source'], {'advanced': True}),
                             'allow_partial': ('BOOLEAN', {'default': False, 'advanced': True})}}

    @classmethod
    def IS_CHANGED(cls, base_model, modified_model, **kwargs):
        return str((identity(resolve(base_model)), identity(resolve(modified_model))))

    def run(self, base_model, modified_model, scan='Header', purpose='ComfyUI Loading', allow_partial=False):
        if scan not in ('Header', 'Full Scan') or purpose not in ('ComfyUI Loading', 'Blend One Source'):
            raise ValueError('Unsupported inspection depth/purpose')
        a, b = make_plan(resolve(base_model), resolve(modified_model))
        result = {'architecture': 'MOSS-Image', 'parameters': len(a['entries']), 'ignored_buffers': [a['ignored'], b['ignored']],
                  'scan': scan, 'values_checked': scan == 'Full Scan'}
        result['largest_fp32_tensor_bytes'] = max(math.prod(e['shape'])*4 for e in a['entries'].values())
        result['full_delta_bf16_bytes_upper_bound'] = sum(math.prod(e['shape'])*2 for e in a['entries'].values())
        if scan == 'Full Scan':
            changed = non_matrix = 0
            examples = []
            bar = comfy.utils.ProgressBar(len(a['entries']))
            with safe_open(str(a['path']), framework='pt', device='cpu') as ha, safe_open(str(b['path']), framework='pt', device='cpu') as hb:
                for index, (key, entry) in enumerate(a['entries'].items()):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    x, y = read_entry(ha, entry), read_entry(hb, b['entries'][key])
                    finite(x, key); finite(y, key)
                    if not x.equal(y):
                        changed += 1
                        if not (key.endswith('.weight') and x.ndim == 2):
                            non_matrix += 1
                            if len(examples) < 8:
                                examples.append(key)
                    bar.update_absolute(index + 1, len(a['entries']))
            result.update(changed=changed, changed_non_matrix=non_matrix, non_matrix_examples=examples)
        for source in (a, b):
            if identity(source['path']) != source['identity']:
                raise ValueError('Input changed during inspection. Run Full Scan again.')
        plan = recommendation(result, base_model, modified_model, purpose, allow_partial)
        plan['identities'] = [a['identity'], b['identity']]
        result['recommendation'] = plan
        text = json.dumps(result, ensure_ascii=False, indent=2)
        return {'ui': {'text': [text]}, 'result': (text, plan)}


def recommendation(info, base_model, modified_model, purpose, allow_partial):
    non_matrix = info.get('changed_non_matrix', 0)
    partial = bool(non_matrix and purpose == 'Blend One Source' and allow_partial)
    policy = 'Exclude With Report' if partial else 'Store Diff' if non_matrix and purpose == 'ComfyUI Loading' else 'Stop'
    blocked = ('full_scan_required' if not info['values_checked'] else
               'no_difference' if not info['changed'] else
               'partial_consent_required' if non_matrix and purpose == 'Blend One Source' and not allow_partial else
               'no_matrix_difference' if partial and info['changed'] == non_matrix else None)
    return {'version': 2, 'base_model': base_model, 'modified_model': modified_model,
            'purpose': purpose, 'ready': blocked is None, 'blocked_reason': blocked,
            'partial': partial, 'changed_non_matrix': non_matrix,
            'non_matrix_examples': info.get('non_matrix_examples', []),
            'artifact_type': 'unconfirmed' if not info['values_checked'] else 'hybrid_delta' if policy == 'Store Diff' else 'standard_lora',
            'required_rank': None, 'output_bytes': None,
            'settings': {'mode': 'Fixed Rank', 'rank': 32, 'energy': .99, 'max_rank': 64,
                         'dtype': 'bf16', 'device': 'auto', 'svd': 'auto', 'non_lora': policy,
                         'key_filter': '*', 'seed': 0}}


def checked_plan(plan, base_model, modified_model):
    if not isinstance(plan, dict) or plan.get('version') != 2:
        raise ValueError('Invalid inspection result. Re-run Inspect or disconnect for manual extraction.')
    if (base_model, modified_model) != (plan['base_model'], plan['modified_model']):
        raise ValueError('Models differ from Inspect. Apply Recommendations, re-analyze this pair, or disconnect Inspect.')
    paths = [resolve(plan['base_model']), resolve(plan['modified_model'])]
    if [list(identity(p)) for p in paths] != [list(i) for i in plan['identities']]:
        raise ValueError('Model files changed after inspection. Run Full Scan again.')


class LoRABlendModelExtract(LoRABlendModelInspect):
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('info',)

    @classmethod
    def IS_CHANGED(cls, base_model, modified_model, extraction_plan=None, **kwargs):
        return super().IS_CHANGED(base_model, modified_model)

    @classmethod
    def INPUT_TYPES(cls):
        choices = models()
        return {'required': {'base_model': (choices,), 'modified_model': (choices,),
                'mode': (['Fixed Rank', 'Auto Rank', 'Full Delta'],), 'rank': ('INT', {'default': 32, 'min': 1, 'max': 4096}),
                'energy': ('FLOAT', {'default': .99, 'min': .01, 'max': 1., 'step': .001}),
                'dtype': (['bf16', 'fp16', 'fp32'],), 'filename_prefix': ('STRING', {'default': 'loras/LoRA_Extract'})},
                'optional': {'max_rank': ('INT', {'default': 64, 'min': 1, 'max': 4096, 'advanced': True}),
                'device': (['auto', 'cpu', 'cuda'], {'advanced': True}), 'svd': (['auto', 'exact', 'randomized'], {'advanced': True}),
                'non_lora': (['Stop', 'Store Diff', 'Exclude With Report'], {'advanced': True}),
                'key_filter': ('STRING', {'default': '*', 'advanced': True}),
                'seed': ('INT', {'default': 0, 'min': 0, 'max': 0x7fffffffffffffff, 'advanced': True}),
                'hash_inputs': ('BOOLEAN', {'default': True, 'advanced': True}),
                'write_report': ('BOOLEAN', {'default': True, 'advanced': True}),
                'extraction_plan': ('LBO_EXTRACTION_PLAN',),
                'plan_action': (['Review Only', 'Extract and Save'], {'default': 'Review Only'})},
                'hidden': {'extra_pnginfo': 'EXTRA_PNGINFO'}}

    def run(self, base_model, modified_model, mode, rank, energy, dtype, filename_prefix,
            extraction_plan=None, plan_action='Review Only', extra_pnginfo=None, **kwargs):
        effective = {'base_model': base_model, 'modified_model': modified_model, 'mode': mode, 'rank': rank,
                     'energy': energy, 'dtype': dtype, 'max_rank': kwargs.get('max_rank', 64),
                     'non_lora': kwargs.get('non_lora', 'Stop'), 'device': kwargs.get('device', 'auto'),
                     'svd': kwargs.get('svd', 'auto'), 'seed': kwargs.get('seed', 0), 'key_filter': kwargs.get('key_filter', '*')}
        if extraction_plan is not None:
            if plan_action not in ('Review Only', 'Extract and Save'):
                raise ValueError('Unsupported plan action')
            # Review also shows blocked plans; it must never invoke extraction or write files.
            if plan_action == 'Review Only':
                text = json.dumps({'extraction_plan': extraction_plan, 'effective_settings': effective,
                                   'review_only': True}, ensure_ascii=False, indent=2)
                return {'ui': {'text': [text]}, 'result': (text,)}
            checked_plan(extraction_plan, base_model, modified_model)
        root = Path(folder_paths.get_output_directory()).resolve()
        prefix = (root / filename_prefix).resolve()
        if not prefix.is_relative_to(root) or prefix == root:
            raise ValueError('Filename Prefix must stay inside ComfyUI/output')
        bar = comfy.utils.ProgressBar(1)
        started = time.monotonic()
        try:
            from server import PromptServer
            from comfy_execution.utils import get_executing_context
        except ImportError:
            context = None
        else:
            context = get_executing_context()
            if context is not None:
                server = PromptServer.instance
                client_id = server.client_id
        workflow = (extra_pnginfo or {}).get('workflow') or {}
        def progress(stage, key, done, total):
            bar.update_absolute(done, max(1, total))
            if context is not None and client_id is not None:
                server.send_sync('lora_blend_one.extract_status',
                    {'node_id': str(context.node_id), 'prompt_id': context.prompt_id,
                     'workflow_id': workflow.get('id'),
                     'text': f'{stage} {done}/{total} | {time.monotonic()-started:.1f}s\n{key}'}, client_id)
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
        display = json.dumps({'extraction_plan': extraction_plan, 'effective_settings': effective,
                              'result_summary': text}, ensure_ascii=False) if extraction_plan is not None else text
        return {'ui': {'text': [display]}, 'result': (text,)}


EXTRACT_NODES = {cls.__name__: cls for cls in (LoRABlendModelInspect, LoRABlendModelExtract)}
EXTRACT_NAMES = {'LoRABlendModelInspect': 'LoRA Blend Model Inspect', 'LoRABlendModelExtract': 'LoRA Blend Model Extract'}
