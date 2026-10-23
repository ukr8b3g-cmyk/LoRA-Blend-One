import fnmatch
import hashlib
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
import torch
from safetensors import safe_open
from .model_files import make_plan, read_entry, identity
from .writer import Spool, finite, publish


def extract(base, modified, output_prefix, *, mode='Fixed Rank', rank=16, energy=.99, max_rank=256,
            dtype='bf16', device='auto', svd='auto', non_lora='Stop', key_filter='*', seed=0,
            hash_inputs=True, write_report=True, cancel=lambda: None, progress=lambda *args: None):
    if mode not in ('Fixed Rank', 'Auto Rank', 'Full Delta') or dtype not in ('bf16', 'fp16', 'fp32'):
        raise ValueError('Unsupported extraction mode/storage type')
    if device not in ('auto', 'cpu', 'cuda') or svd not in ('auto', 'exact', 'randomized'):
        raise ValueError('Unsupported device/SVD method')
    if non_lora not in ('Stop', 'Store Diff', 'Exclude With Report') or not 0 < energy <= 1 or min(rank, max_rank) < 1:
        raise ValueError('Invalid rank/energy/non-LoRA policy')
    started = time.monotonic()
    a, b = make_plan(base, modified)
    storage = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[dtype]
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report = {'version': 1, 'architecture': 'moss_image', 'base': Path(base).name, 'modified': Path(modified).name,
              'mode': mode, 'rank': rank, 'energy_target': energy, 'max_rank': max_rank, 'dtype': dtype,
              'svd': svd, 'seed': seed, 'non_lora': non_lora, 'key_filter': key_filter, 'layers': [],
              'ignored_buffers': [a['ignored'], b['ignored']], 'warnings': [], 'excluded': [], 'zero': []}
    keys = sorted(a['entries'])
    estimated = sum(math.prod(e['shape']) * torch.empty((), dtype=storage).element_size() for e in a['entries'].values())
    if shutil.disk_usage(prefix.parent).free < 2*estimated + 64*1024*1024:
        raise OSError('Insufficient working disk space (conservative full-delta upper bound)')
    if hash_inputs:
        report['sha256'] = {}
        for label, path in (('base', Path(base)), ('modified', Path(modified))):
            digest = hashlib.sha256()
            with path.open('rb') as f:
                while block := f.read(8*1024*1024):
                    cancel(); progress('Hash', path.name, 0, len(keys)); digest.update(block)
            report['sha256'][label] = digest.hexdigest()
    mixed = False
    total_energy = total_error = 0.
    with tempfile.TemporaryDirectory(prefix='.lbo_extract_', dir=prefix.parent) as folder:
        spool = Spool(folder, cancel)
        with safe_open(str(base), framework='pt', device='cpu') as ha, safe_open(str(modified), framework='pt', device='cpu') as hb:
            for index, key in enumerate(keys):
                cancel(); progress('Extract', key, index, len(keys))
                x, y = read_entry(ha, a['entries'][key]), read_entry(hb, b['entries'][key])
                finite(x, f'{Path(base).name}: {key}'); finite(y, f'{Path(modified).name}: {key}')
                delta = y.float() - x.float()
                del x, y
                finite(delta, key)
                norm2 = float(delta.double().square().sum())
                total_energy += norm2
                if not norm2:
                    report['zero'].append(key); continue
                if not fnmatch.fnmatchcase(key, key_filter):
                    report['excluded'].append(key); total_error += norm2; continue
                linear = key.endswith('.weight') and delta.ndim == 2
                if not linear and mode != 'Full Delta' and non_lora == 'Stop':
                    raise ValueError(f'{key}: changed non-LoRA parameter; select Store Diff, Full Delta, or explicit partial extraction')
                if not linear and mode != 'Full Delta' and non_lora == 'Exclude With Report':
                    report['excluded'].append(key); total_error += norm2; continue
                target = key[:-7] if key.endswith('.weight') else key
                if mode == 'Full Delta' or not linear:
                    saved = delta.to(storage).contiguous()
                    finite(saved, key)
                    tensors = {target + '.diff': saved}
                    errors = delta - saved.float()
                    err2 = float(errors.double().square().sum())
                    max_error = float(errors.abs().max()); selected_rank = None; captured = None
                    mixed = True
                else:
                    scale = float(delta.abs().max())
                    matrix = delta / scale
                    cap = min(rank if mode == 'Fixed Rank' else max_rank, *matrix.shape)
                    use_exact = svd == 'exact' or (svd == 'auto' and min(matrix.shape) <= 512 and matrix.numel() <= 4_000_000)
                    work_device = torch.device('cpu')
                    if device == 'cuda':
                        work_device = torch.device('cuda')
                    elif device == 'auto' and torch.cuda.is_available():
                        free, _ = torch.cuda.mem_get_info()
                        if matrix.numel()*4*12 < free*.5:
                            work_device = torch.device('cuda')

                    def decompose(where):
                        mat = matrix.to(where)
                        if use_exact:
                            return torch.linalg.svd(mat, full_matrices=False)
                        q = min(min(mat.shape), cap+8)
                        generator = torch.Generator(device=where).manual_seed(seed_for(seed, key))
                        omega = torch.randn((mat.shape[1], q), generator=generator, device=where)
                        y = mat @ omega
                        for _ in range(2):
                            y = mat @ (mat.T @ torch.linalg.qr(y, mode='reduced').Q)
                        basis = torch.linalg.qr(y, mode='reduced').Q
                        u, s, vh = torch.linalg.svd(basis.T @ mat, full_matrices=False)
                        return basis @ u, s, vh

                    try:
                        u, s, vh = decompose(work_device)
                    except torch.cuda.OutOfMemoryError:
                        if work_device.type != 'cuda':
                            raise
                        report['warnings'].append(f'{key}: CUDA OOM, same-settings CPU retry')
                        u, s, vh = decompose(torch.device('cpu'))
                    energies = s.double().square().cpu() / (norm2/(scale*scale))
                    selected_rank = cap
                    if mode == 'Auto Rank':
                        selected_rank = min(cap, int(torch.searchsorted(energies.cumsum(0), energy)) + 1)
                    captured = float(energies[:selected_rank].sum())
                    if mode == 'Auto Rank' and captured < energy:
                        report['warnings'].append(f'{key}: energy target not reached: {captured:.6f}')
                    root = s[:selected_rank].sqrt() * math.sqrt(scale)
                    up = (u[:, :selected_rank] * root).to(device='cpu', dtype=storage).contiguous()
                    down = (root[:, None] * vh[:selected_rank]).to(device='cpu', dtype=storage).contiguous()
                    finite(up, key); finite(down, key)
                    tensors = {target+'.lora_A.weight': down, target+'.lora_B.weight': up,
                               target+'.alpha': torch.tensor(float(selected_rank))}
                    err2 = 0.; max_error = 0.
                    for start in range(0, delta.shape[0], 128):
                        cancel()
                        error = delta[start:start+128] - up[start:start+128].float() @ down.float()
                        err2 += float(error.double().square().sum())
                        max_error = max(max_error, float(error.abs().max()))
                    del u, s, vh, root, matrix, up, down
                total_error += err2
                report['layers'].append({'key': key, 'rank': selected_rank, 'captured_energy': captured,
                                         'relative_error': math.sqrt(err2/norm2), 'max_absolute_error': max_error})
                spool.add(tensors)
                progress(f'Layer saved (rank={selected_rank})', key, index+1, len(keys))
                del tensors, delta
        for source in (a, b):
            if identity(source['path']) != source['identity']:
                raise ValueError(f'{source["path"].name}: input changed during extraction')
        report['relative_error_all_parameters'] = math.sqrt(total_error/total_energy) if total_energy else 0.
        report['partial'] = bool(report['excluded'])
        report['artifact_type'] = 'full_delta' if mode == 'Full Delta' else 'hybrid_delta' if mixed else 'standard_lora'
        report['seconds'] = time.monotonic()-started
        if not spool.entries:
            report['status'] = 'No difference' if not total_energy else 'No selected difference'
            return None, report
        metadata = {'lbo_extraction_version': '1', 'lbo_artifact_type': report['artifact_type'],
                    'lbo_model_family': 'moss_image', 'lbo_extraction': json.dumps({k:v for k,v in report.items() if k != 'layers'})}
        progress('Verify', '', len(keys), len(keys))
        temp = spool.finish(metadata)
        for counter in range(1, 1000000):
            output = prefix.with_name(f'{prefix.name}_{counter:05}_.safetensors')
            sidecar = output.with_suffix(output.suffix+'.report.json')
            if output.exists() or sidecar.exists():
                continue
            try:
                publish(temp, output); break
            except FileExistsError:
                continue
        else:
            raise FileExistsError('No free output filename')
        if write_report:
            report_temp = Path(folder) / 'report.partial'
            try:
                report_temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
                publish(report_temp, sidecar)
            except (OSError, ValueError, TypeError) as error:
                report['warnings'].append(f'Saved {output.name}; report failed: {error}')
        progress('Saved', output.name, len(keys), len(keys))
        return output, report


def seed_for(seed, key):
    return int.from_bytes(hashlib.blake2b(f'{seed}:{key}'.encode(), digest_size=8).digest(), 'little') & ((1<<63)-1)
