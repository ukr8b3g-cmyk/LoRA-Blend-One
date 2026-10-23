from pathlib import Path
import re
from safetensors import safe_open


def identity(path):
    stat = Path(path).stat()
    return (stat.st_size, stat.st_mtime_ns)


def inspect_model(path):
    path = Path(path)
    if path.suffix.lower() != '.safetensors':
        raise ValueError('Only single, non-quantized Safetensors files are supported')
    entries = {}
    ignored = []
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        for raw in handle.keys():
            key = raw
            for prefix in ('model.diffusion_model.', 'diffusion_model.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            if key == 'rope.bands':
                ignored.append(raw)
                continue
            shape = tuple(handle.get_slice(raw).get_shape())
            dtype = handle.get_slice(raw).get_dtype()
            if dtype not in ('F32', 'F16', 'BF16') or 'scale_weight' in key or 'weight_scale' in key:
                raise ValueError(f'{path.name}: quantized/unsupported tensor {raw}: {dtype}')
            module = key.removesuffix('.weight').removesuffix('.bias')
            allowed = (re.fullmatch(r'layers\.\d+\.(proj\.(qkv|q|k|v|o|ffn_up_gate|ffn_down)|k_norm|q_norm|norm1|norm2)', module)
                       or re.fullmatch(r'text_embedder\.blocks\.\d+\.(qkv|proj|ffn_gate|ffn_up|ffn_down|norm1|norm2|q_norm|k_norm)', module)
                       or module in {'video_embedder', 'text_embedder.proj', 'time_embedding.0', 'time_embedding.2',
                                     'shared_block_modulation', 'video_head_modulation', 'final_linear_video', 'final_norm_video'})
            if not allowed:
                raise ValueError(f'{path.name}: unsupported MOSS target {raw}')
            if re.fullmatch(r'layers\.\d+\.proj\.qkv\.weight', key):
                if len(shape) != 2 or shape[0] % 3:
                    raise ValueError(f'{path.name}: invalid QKV shape: {raw}')
                rows = shape[0] // 3
                parts = [(key.replace('.qkv.weight', f'.{part}.weight'), (rows, shape[1]), (i*rows, (i+1)*rows))
                         for i, part in enumerate(('q', 'k', 'v'))]
            else:
                parts = [(key, shape, None)]
            for name, dims, row_slice in parts:
                canonical = 'diffusion_model.' + name
                if canonical in entries:
                    raise ValueError(f'{path.name}: ambiguous alias {canonical}')
                entries[canonical] = {'raw': raw, 'shape': dims, 'dtype': dtype, 'rows': row_slice}
    if not any(re.fullmatch(r'diffusion_model.layers\.\d+\.proj\.q\.weight', k) for k in entries):
        raise ValueError(f'{path.name}: not a supported MOSS-Image model')
    return {'path': path, 'identity': identity(path), 'entries': entries, 'ignored': ignored}


def make_plan(base, modified):
    a, b = inspect_model(base), inspect_model(modified)
    if set(a['entries']) != set(b['entries']):
        raise ValueError(f'Model keys differ: {sorted(set(a["entries"]) ^ set(b["entries"]))[:8]}')
    for key, entry in a['entries'].items():
        if entry['shape'] != b['entries'][key]['shape']:
            raise ValueError(f'{key}: incompatible shapes')
    return a, b


def read_entry(handle, entry):
    rows = entry['rows']
    return (handle.get_slice(entry['raw'])[rows[0]:rows[1], :] if rows else handle.get_tensor(entry['raw']))
