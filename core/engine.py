from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable

import torch

from .errors import MergeError
from .formats import OpenedSources
from .math_ops import (
    dare_sparsify,
    della_sparsify,
    exact_core_svd,
    factor_fro_dot,
    factor_fro_norm,
    flatten_factors,
    gate_conflict_count,
    make_generator,
    randomized_svd_factors,
    reconstruct_delta,
    restore_factor_layout,
    stable_seed,
    ties_sparsify,
    torch_dtype_from_name,
)
from .types import LayerReport, MergeReport, MergeSettings, SourceInfo


GATE_MARKERS = (".attn.to_gate", ".attn.gate", ".ff.gate", ".mlp.gate", ".gate_proj")
KREA2_GATE_MARKERS = (".attn.to_gate", ".attn.gate", ".ff.gate", ".mlp.gate")


def is_gate_target(target: str) -> bool:
    return any(marker in target for marker in GATE_MARKERS)


def resolve_weights(raw_percentages: list[float], mode: str = "normalize") -> list[float]:
    vals = [float(x) / 100.0 for x in raw_percentages]
    if not all(math.isfinite(x) for x in vals) or not math.isfinite(sum(vals)):
        raise ValueError("Weights and their total must be finite (no NaN or Inf)")
    if mode == "direct":
        if all(abs(x) < 1e-15 for x in vals):
            raise ValueError("All direct weights are zero")
        return vals
    if any(x < 0 for x in vals):
        raise ValueError("Negative weights require weight_mode=direct")
    total = sum(vals)
    if total <= 0:
        raise ValueError("At least one positive weight is required")
    return [x / total for x in vals]


def resolve_method(name: str) -> str:
    n = name.strip().lower()
    if n == "auto":
        return "Weighted"
    table = {"weighted": "Weighted", "ties": "TIES", "dare": "DARE", "della": "DELLA", "krea2 gate-aware": "Krea2 Gate-Aware"}
    if n not in table:
        raise ValueError(f"Unsupported merge method: {name}")
    return table[n]


def resolve_sign_consensus(method: str, setting: str) -> bool:
    s = setting.strip().lower()
    if s == "on":
        return True
    if s == "off":
        return False
    return method in ("TIES", "DELLA")


def _resolved_rank_mode(settings: MergeSettings, method: str, max_input_rank: int, rank_sum: int) -> tuple[str, int | None]:
    mode = settings.compression_mode.lower()
    if mode == "auto":
        return ("energy", None) if settings.output_rank is None else ("fixed", settings.output_rank)
    if mode == "none":
        if method != "Weighted":
            raise ValueError("compression=none is supported only for Weighted merge")
        if settings.output_rank is None or settings.output_rank >= rank_sum:
            return "none", None
        return "fixed", settings.output_rank
    if mode == "max_input_rank":
        return "fixed", min(max_input_rank, settings.output_rank) if settings.output_rank is not None else max_input_rank
    if mode == "fixed_rank":
        if settings.output_rank is None:
            raise ValueError("fixed_rank compression requires a numeric output rank")
        return "fixed", settings.output_rank
    if mode == "energy":
        return "energy", None
    raise ValueError(f"Unknown compression mode: {settings.compression_mode}")


def _source_dtype_to_storage(infos: list[SourceInfo]) -> torch.dtype:
    dtypes = set()
    for info in infos:
        dtypes.update(info.factor_dtypes)
    if len(dtypes) == 1:
        only = next(iter(dtypes))
        return {
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "F32": torch.float32,
        }.get(only, torch.float32)
    return torch.float32


def resolve_storage_dtype(name: str, infos: list[SourceInfo]) -> torch.dtype:
    if name.lower() == "auto":
        return _source_dtype_to_storage(infos)
    return torch_dtype_from_name(name)


def _cuda_free_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        free, _ = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return 0


def choose_device(method: str, out_dim: int, in_flat: int, rank_sum: int, auto_rank_cap: int, requested: str) -> torch.device:
    req = requested.lower()
    if req == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if req == "cuda":
        return torch.device("cuda")
    free = _cuda_free_bytes()
    if free <= 0:
        return torch.device("cpu")
    if method == "Weighted":
        factor_bytes = (out_dim + in_flat) * max(1, rank_sum) * 4
        estimate = factor_bytes * 8 + max(1, rank_sum) ** 2 * 32
    else:
        dense_bytes = out_dim * in_flat * 4
        q = min(max(32, auto_rank_cap + 8), min(out_dim, in_flat))
        rsvd_bytes = (out_dim + in_flat) * q * 4 * 3
        estimate = dense_bytes * 3 + rsvd_bytes
    return torch.device("cuda") if estimate < int(free * 0.55) else torch.device("cpu")


def _validate_sources(infos: list[SourceInfo], compatibility_mode: str) -> list[str]:
    warnings: list[str] = []
    if any(info.architecture_hint == "moss_image" for info in infos) and any(info.architecture_hint != "moss_image" for info in infos):
        incompatible = [info for info in infos if info.architecture_hint != "moss_image"]
        names = ", ".join(f"{info.name} ({info.architecture_hint})" for info in incompatible)
        raise MergeError("moss_mixed", [info.name for info in incompatible],
                         "MOSS-Image LoRAs must be blended with other recognized MOSS-Image LoRAs. "
                         f"Incompatible sources: {names}", source_paths=[str(info.path) for info in incompatible])
    for info in infos:
        if info.unsupported:
            preview = ", ".join(info.unsupported[:5])
            extra = " ..." if len(info.unsupported) > 5 else ""
            raise MergeError("unsupported_adapter", [info.name],
                f"{info.name} contains unsupported adapter tensors: {preview}{extra}",
                source_paths=[str(info.path)],
            )
        warnings.extend(f"{info.name}: {w}" for w in info.warnings)

    if len(infos) < 2:
        return warnings

    hints = {i.architecture_hint for i in infos if i.architecture_hint != "unknown"}
    common = set(infos[0].layers)
    for info in infos[1:]:
        common &= set(info.layers)

    if not common and len(hints) > 1 and compatibility_mode != "allow_disjoint":
        names = ", ".join(f"{info.name} ({info.architecture_hint})" for info in infos)
        raise MergeError("model_mismatch", [info.name for info in infos],
            "Selected LoRAs have no matching target keys and appear to belong to different architecture families. "
            "Use compatibility_mode=allow_disjoint only when you know the adapters target the same base model. "
            f"Sources: {names}", source_paths=[str(info.path) for info in infos],
        )
    if not common:
        warnings.append("No shared target keys were found; layers will be combined as disjoint adapters.")
    if len(hints) > 1:
        warnings.append(f"Architecture hints differ: {', '.join(sorted(hints))}")
    return warnings


def _check_target_shapes(contributors, infos):
    first_index, _, first = contributors[0]
    for source_index, _, spec in contributors[1:]:
        sources = [infos[first_index].name, infos[source_index].name]
        source_paths = [str(infos[first_index].path), str(infos[source_index].path)]
        if spec.out_dim != first.out_dim or spec.in_flat != first.in_flat or spec.layout != first.layout:
            raise MergeError("shape_mismatch", sources,
                f"Target {first.target!r} has incompatible shapes/layouts: "
                f"{first.up_shape}/{first.down_shape} vs {spec.up_shape}/{spec.down_shape}. "
                f"Sources: {sources[0]} vs {sources[1]}", source_paths=source_paths,
            )
        if spec.layout == "conv" and spec.down_tail != first.down_tail:
            raise MergeError("shape_mismatch", sources,
                             f"Target {first.target!r} has incompatible convolution kernel shapes. "
                             f"Sources: {sources[0]} vs {sources[1]}", source_paths=source_paths)


def _sparsify(method: str, delta: torch.Tensor, settings: MergeSettings, seed: int) -> torch.Tensor:
    if method == "TIES":
        return ties_sparsify(delta, settings.density)
    if method == "DARE":
        return dare_sparsify(delta, settings.density, generator=make_generator(delta.device, seed))
    if method == "DELLA":
        return della_sparsify(
            delta,
            settings.density,
            settings.della_half_width,
            generator=make_generator(delta.device, seed),
        )
    return delta


def _dense_delta(opened: OpenedSources, source_idx: int, spec, device: torch.device) -> torch.Tensor:
    up, down = opened.read_factors(source_idx, spec)
    up2, down2 = flatten_factors(up, down, spec, device=device)
    return reconstruct_delta(up2, down2, spec.effective_scale)


def _merge_dense_target(opened, contributors, method, settings, device, weights, cancel_check):
    sign_consensus = resolve_sign_consensus(method, settings.sign_consensus)
    first = contributors[0][2]
    shape = (first.out_dim, first.in_flat)

    def source_sparse(src_idx, spec):
        delta = _dense_delta(opened, src_idx, spec, device)
        seed = stable_seed(settings.seed, first.target, src_idx, method)
        return _sparsify(method, delta, settings, seed)

    elected = None
    if sign_consensus:
        sign_acc = torch.zeros(shape, device=device, dtype=torch.float32)
        for src_idx, weight, spec in contributors:
            if cancel_check:
                cancel_check()
            if abs(weight) < 1e-15:
                continue
            d = source_sparse(src_idx, spec)
            sign_acc.add_(d, alpha=float(weight * settings.output_scale))
            del d
        elected = torch.sign(sign_acc)
        del sign_acc

    merged = torch.zeros(shape, device=device, dtype=torch.float32)
    for src_idx, weight, spec in contributors:
        if cancel_check:
            cancel_check()
        if abs(weight) < 1e-15:
            continue
        d = source_sparse(src_idx, spec)
        d.mul_(float(weight * settings.output_scale))
        if elected is not None:
            agree = (torch.sign(d) == elected) & (elected != 0)
            d.mul_(agree)
        merged.add_(d)
        del d
    return merged


def _merge_krea2_sources(infos, weights, settings, warnings, progress, cancel_check):
    if len(infos) != 2:
        raise ValueError("Krea2 Gate-Aware requires exactly two LoRA slots")
    requested_rank = max(1, int(settings.output_rank)) if settings.output_rank is not None else None
    targets = sorted(set(infos[0].layers) | set(infos[1].layers))
    storage_dtype = torch.float32 if settings.output_dtype == "auto" else resolve_storage_dtype(settings.output_dtype, infos)
    report = MergeReport(method="Krea2 Gate-Aware", source_count=2, warnings=warnings)
    output = {}

    with OpenedSources(infos) as opened:
        for index, target in enumerate(targets):
            if cancel_check:
                cancel_check()
            contributors = [(i, float(weights[i]), info.layers[target]) for i, info in enumerate(infos) if target in info.layers]
            _check_target_shapes(contributors, infos)
            template = contributors[0][2]
            unmatched = len(contributors) == 1
            if unmatched and settings.unmatched_policy == "error":
                present = {item[0] for item in contributors}
                missing = [info for source_index, info in enumerate(infos) if source_index not in present]
                raise MergeError("missing_targets", [info.name for info in missing],
                                 f"Target {target!r} is not present in both Krea2 LoRAs. "
                                 f"Missing sources: {', '.join(info.name for info in missing)}",
                                 source_paths=[str(info.path) for info in missing])
            if unmatched and settings.unmatched_policy == "drop":
                continue
            rank_sum = sum(item[2].rank for item in contributors)
            max_rank = max(item[2].rank for item in contributors)
            gate = any(marker in target for marker in KREA2_GATE_MARKERS)
            device = torch.device("cuda" if settings.device == "cuda" and torch.cuda.is_available() else "cpu")

            def run_on(run_device):
                factors = []
                for source_index, weight, spec in contributors:
                    up, down = opened.read_factors(source_index, spec)
                    up, down = flatten_factors(up, down, spec, device=run_device)
                    factors.append((up, down, float(spec.effective_scale * weight * settings.output_scale)))
                scales = [entry[2] for entry in factors]
                if len(factors) == 2 and settings.krea2_balance_mode == "per_layer_norm":
                    norms = [float(factor_fro_norm(up * scale, down).item()) for (up, down, scale) in factors]
                    if all(norms):
                        middle = sum(norms) / 2.0
                        scales = [scale * middle / norm for scale, norm in zip(scales, norms)]

                conflict = 0
                if gate and len(factors) == 2:
                    scales = [scale * settings.krea2_gate_strength for scale in scales]
                    norms = [float(factor_fro_norm(up * scale, down).item()) for (up, down, _), scale in zip(factors, scales)]
                    if norms[0] > 0 and norms[1] > 0:
                        dot = factor_fro_dot(factors[0][0] * scales[0], factors[0][1],
                                             factors[1][0] * scales[1], factors[1][1])
                        conflict = int(float(dot.item()) / (norms[0] * norms[1]) < -settings.krea2_opposite_threshold)
                    if conflict and settings.krea2_opposite_strategy == "zero_out":
                        rank = settings.krea2_gate_rank if settings.krea2_gate_rank > 0 else max_rank
                        if requested_rank is not None:
                            rank = min(rank, requested_rank)
                        up = torch.zeros((template.out_dim, rank), device=run_device, dtype=torch.float32)
                        down = torch.zeros((rank, template.in_flat), device=run_device, dtype=torch.float32)
                        return up, down, rank, 1.0, True, conflict
                    if conflict or norms[0] == 0 or norms[1] == 0:
                        chosen = 1 if norms[0] == 0 else 0 if norms[0] >= norms[1] else 1
                        up, down, _ = factors[chosen]
                        compress_chosen = requested_rank is not None and requested_rank < up.shape[1]
                        up, down, rank, energy, exact = exact_core_svd(
                            [up], [down], [scales[chosen]],
                            rank_mode="fixed" if compress_chosen else "none",
                            fixed_rank=requested_rank if compress_chosen else None, energy_target=1.0)
                        return up, down, rank, energy, exact, conflict

                compress = False
                rank = None
                if len(factors) == 2:
                    if gate:
                        mode = settings.krea2_gate_compression
                        compress = mode == "all_gates" or (mode == "attn_to_gate_only" and
                            any(marker in target for marker in KREA2_GATE_MARKERS[:2]))
                        rank = max(1, settings.krea2_gate_rank)
                    else:
                        compress = settings.krea2_normal_compression == "max_input_rank"
                        rank = max_rank
                if requested_rank is not None and (not compress or requested_rank < rank):
                    compress = True
                    rank = requested_rank
                if compress and rank >= rank_sum:
                    compress = False
                up, down, out_rank, energy, exact = exact_core_svd(
                    [item[0] for item in factors], [item[1] for item in factors], scales,
                    rank_mode="fixed" if compress else "none", fixed_rank=rank, energy_target=1.0)
                return up, down, out_rank, energy, exact, conflict

            try:
                up2, down2, out_rank, energy, exact, conflict = run_on(device)
            except torch.cuda.OutOfMemoryError:
                if device.type != "cuda" or settings.fallback_policy != "auto_cpu":
                    raise
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                report.warnings.append(f"{target}: CUDA OOM; retried on CPU with identical Krea2 settings")
                up2, down2, out_rank, energy, exact, conflict = run_on(device)
            up, down = restore_factor_layout(up2, down2, template)
            output[f"{target}{template.style.up_suffix}"] = _storage_factor(up, storage_dtype, target, contributors, infos)
            output[f"{target}{template.style.down_suffix}"] = _storage_factor(down, storage_dtype, target, contributors, infos)
            output[f"{target}.alpha"] = torch.tensor(float(out_rank), dtype=torch.float32)
            report.layer_reports.append(LayerReport(
                target=target, contributors=len(contributors), method=report.method, device=device.type,
                input_rank_sum=rank_sum, max_input_rank=max_rank, output_rank=out_rank,
                captured_energy=energy, exact=exact, unmatched=unmatched, gate_conflicts=conflict))
            if gate:
                report.gate_targets += 1
                report.gate_conflicts += conflict
            if progress:
                progress(index + 1, len(targets))
    return output, report


def _storage_factor(tensor, dtype, target, contributors, infos):
    tensor = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
    if not torch.isfinite(tensor).all().item():
        sources = [infos[index] for index, _, _ in contributors]
        raise MergeError("nonfinite_output", [info.name for info in sources],
                         f"{target}: output contains NaN or Inf after conversion to {dtype}; "
                         "check strengths/input values or select a storage type with a wider range",
                         source_paths=[str(info.path) for info in sources])
    return tensor


def merge_sources(
    infos: list[SourceInfo],
    weights: list[float],
    settings: MergeSettings,
    *,
    compatibility_mode: str = "safe",
    progress: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> tuple[dict[str, torch.Tensor], MergeReport]:
    if len(infos) != len(weights):
        raise ValueError("infos/weights length mismatch")
    if not infos:
        raise ValueError("No LoRAs selected")
    method = resolve_method(settings.method)
    for info, weight in zip(infos, weights):
        for spec in info.layers.values():
            coefficient = weight * settings.output_scale * spec.effective_scale
            gate_coefficient = (coefficient * settings.krea2_gate_strength
                                if method == "Krea2 Gate-Aware" and is_gate_target(spec.target) else coefficient)
            if not all(math.isfinite(value) for value in (weight, settings.output_scale, spec.effective_scale,
                                                        coefficient, gate_coefficient)):
                raise MergeError("nonfinite_input", [info.name],
                                 f"{info.name}: {spec.target}: weight/alpha/output scale produces NaN or Inf",
                                 source_paths=[str(info.path)])
    warnings = _validate_sources(infos, compatibility_mode)
    if method == "Krea2 Gate-Aware":
        if any(info.architecture_hint == "moss_image" for info in infos):
            moss = [info for info in infos if info.architecture_hint == "moss_image"]
            raise MergeError("krea2_moss", [info.name for info in moss],
                             "Krea2 Gate-Aware cannot be used with MOSS-Image LoRAs. "
                             f"Sources: {', '.join(info.name for info in moss)}",
                             source_paths=[str(info.path) for info in moss])
        return _merge_krea2_sources(infos, weights, settings, warnings, progress, cancel_check)
    storage_dtype = resolve_storage_dtype(settings.output_dtype, infos)

    targets = sorted(set().union(*(info.layers.keys() for info in infos)))
    report = MergeReport(method=method, source_count=len(infos), warnings=warnings)
    output: dict[str, torch.Tensor] = {}

    with OpenedSources(infos) as opened:
        for target_index, target in enumerate(targets):
            if cancel_check:
                cancel_check()
            contributors = []
            for src_idx, (info, weight) in enumerate(zip(infos, weights)):
                spec = info.layers.get(target)
                if spec is not None and abs(weight) > 1e-15:
                    contributors.append((src_idx, float(weight), spec))
            if not contributors:
                continue
            _check_target_shapes(contributors, infos)
            template = contributors[0][2]
            unmatched = len(contributors) < len(infos)
            if unmatched and settings.unmatched_policy == "error":
                present = {item[0] for item in contributors}
                missing = [info for source_index, info in enumerate(infos) if source_index not in present]
                raise MergeError("missing_targets", [info.name for info in missing],
                                 f"Target {target!r} is not present in every selected LoRA. "
                                 f"Missing sources: {', '.join(info.name for info in missing)}",
                                 source_paths=[str(info.path) for info in missing])
            if unmatched and settings.unmatched_policy == "drop":
                continue

            rank_sum = sum(c[2].rank for c in contributors)
            max_input_rank = max(c[2].rank for c in contributors)
            rank_mode, fixed_rank = _resolved_rank_mode(settings, method, max_input_rank, rank_sum)
            rank_cap = max(1, int(settings.output_rank)) if settings.output_rank is not None else None
            energy_rank_cap = min(settings.auto_rank_cap, rank_cap) if rank_mode == "energy" and rank_cap is not None else settings.auto_rank_cap
            device = choose_device(method, template.out_dim, template.in_flat, rank_sum, energy_rank_cap, settings.device)

            def run_on(run_device: torch.device):
                gate_conflicts = 0
                if method == "Weighted":
                    ups, downs, scales = [], [], []
                    gate_factor_pairs = []
                    for src_idx, weight, spec in contributors:
                        up, down = opened.read_factors(src_idx, spec)
                        up2, down2 = flatten_factors(up, down, spec, device=run_device)
                        weighted_scale = float(spec.effective_scale * weight * settings.output_scale)
                        ups.append(up2)
                        downs.append(down2)
                        scales.append(weighted_scale)
                        if is_gate_target(target):
                            # Bake scale into up for direction diagnostics, including negative weights.
                            gate_factor_pairs.append((up2 * weighted_scale, down2))
                    if gate_factor_pairs:
                        gate_conflicts = gate_conflict_count(gate_factor_pairs, settings.gate_conflict_threshold)
                    up2, down2, out_rank, energy, exact = exact_core_svd(
                        ups,
                        downs,
                        scales,
                        rank_mode=rank_mode,
                        fixed_rank=fixed_rank,
                        energy_target=settings.energy_target,
                        rank_cap=rank_cap if rank_mode == "energy" else None,
                    )
                else:
                    dense = _merge_dense_target(opened, contributors, method, settings, run_device, weights, cancel_check)
                    svd_seed = stable_seed(settings.seed, target, "refactor")
                    up2, down2, out_rank, energy = randomized_svd_factors(
                        dense,
                        rank_mode=rank_mode,
                        fixed_rank=fixed_rank,
                        energy_target=settings.energy_target,
                        auto_rank_cap=energy_rank_cap,
                        seed=svd_seed,
                    )
                    exact = False
                    del dense
                return up2, down2, out_rank, energy, exact, gate_conflicts

            try:
                up2, down2, out_rank, energy, exact, gate_conflicts = run_on(device)
            except torch.cuda.OutOfMemoryError:
                if device.type != "cuda" or settings.fallback_policy != "auto_cpu":
                    raise
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                report.warnings.append(f"{target}: CUDA OOM; retried on CPU with identical merge settings")
                up2, down2, out_rank, energy, exact, gate_conflicts = run_on(device)

            up, down = restore_factor_layout(up2, down2, template)
            up = _storage_factor(up, storage_dtype, target, contributors, infos)
            down = _storage_factor(down, storage_dtype, target, contributors, infos)
            alpha = torch.tensor(float(out_rank), dtype=torch.float32)
            output[f"{target}{template.style.up_suffix}"] = up
            output[f"{target}{template.style.down_suffix}"] = down
            output[f"{target}.alpha"] = alpha

            lr_warnings = []
            if rank_mode == "energy" and energy + 1e-7 < settings.energy_target:
                lr_warnings.append(
                    f"Energy target {settings.energy_target:.5f} was not reached; estimated captured energy={energy:.5f}."
                )
            report.layer_reports.append(
                LayerReport(
                    target=target,
                    contributors=len(contributors),
                    method=method,
                    device=device.type,
                    input_rank_sum=rank_sum,
                    max_input_rank=max_input_rank,
                    output_rank=out_rank,
                    captured_energy=energy,
                    exact=exact,
                    unmatched=unmatched,
                    gate_conflicts=gate_conflicts,
                    warnings=lr_warnings,
                )
            )
            if is_gate_target(target):
                report.gate_targets += 1
                report.gate_conflicts += gate_conflicts
            if progress:
                progress(target_index + 1, len(targets))

            del up2, down2
            if device.type == "cuda":
                # Do not aggressively empty cache per layer; PyTorch reuses it. Interrupt/OOM paths handle cleanup.
                pass

    if not output:
        raise ValueError("Merge produced no supported LoRA tensors")
    return output, report
