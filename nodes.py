from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import folder_paths

try:
    import comfy.model_management as model_management
except Exception:  # pragma: no cover - only relevant outside ComfyUI
    model_management = None

try:
    import comfy.utils as comfy_utils
except Exception:  # pragma: no cover
    comfy_utils = None

from .core import (
    MergeSettings,
    build_metadata,
    build_recipe,
    merge_sources,
    report_to_dict,
    resolve_weights,
    save_verified,
    scan_lora,
)
from .core.errors import MergeError
from .error_reporting import publish_failure


NONE = "(none)"
VERSION = "0.1.0"
SOURCE_OPTIONAL_KEYS = ({key for idx in range(3, 9) for key in (f"lora_{idx}", f"weight_{idx}")}
                        | {f"enabled_{idx}" for idx in range(1, 9)})


def _lora_choices():
    names = list(folder_paths.get_filename_list("loras"))
    return [NONE] + names


def _default_lora(index: int):
    names = list(folder_paths.get_filename_list("loras"))
    return names[index] if len(names) > index else NONE


def _selected_slots(values, enabled=None):
    rows = []
    for slot, (name, weight) in enumerate(values, start=1):
        if enabled is not None and not enabled.get(f"enabled_{slot}", True):
            continue
        if name and name != NONE:
            rows.append((slot, name, float(weight)))
    return rows


def _error_slots(lora_1, lora_2, values):
    names = [lora_1, lora_2] + [values.get(f"lora_{idx}", NONE) for idx in range(3, 9)]
    return [(slot, name) for slot, name in enumerate(names, 1)
            if name and name != NONE and values.get(f"enabled_{slot}", True)]


def _is_interruption(error):
    return model_management is not None and isinstance(error, getattr(model_management, "InterruptProcessingException", ()))


def _coalesce_duplicate_sources(rows):
    order = []
    weights = {}
    for _, name, weight in rows:
        if name not in weights:
            order.append(name)
            weights[name] = 0.0
        weights[name] += float(weight)
    return [(name, weights[name]) for name in order]


def _rank_from_combo(output_rank: str, custom_rank: int):
    if output_rank == "Auto":
        return None
    if output_rank == "Custom":
        return int(custom_rank)
    return int(output_rank)


class LoRABlendOne:
    """One-node, multi-LoRA file merger with Auto defaults and Advanced controls."""

    DESCRIPTION = (
        "Merge 1-8 compatible Standard LoRAs into one safetensors file. Auto uses a weighted delta merge, "
        "FP32 computation, automatic rank compression, safe output verification, and recipe metadata. "
        "Open Advanced parameters for TIES, DARE, DELLA, additional LoRAs, dtype/device controls, and diagnostics."
    )
    CATEGORY = "model/merging/LoRA Blend One"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    OUTPUT_TOOLTIPS = ("Merge result, saved path, resolved method, rank range, and warnings.",)
    FUNCTION = "merge"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        choices = _lora_choices()
        d1 = _default_lora(0)
        d2 = _default_lora(1)
        required = {
            "lora_1": (choices, {"default": d1, "tooltip": "First LoRA file. Standard LoRA factor formats supported by current ComfyUI are recognized."}),
            "weight_1": ("FLOAT", {"default": 50.0, "min": -400.0, "max": 400.0, "step": 0.1, "tooltip": "Weight in percent. In Normalize mode, selected positive weights are normalized to 100%."}),
            "lora_2": (choices, {"default": d2, "tooltip": "Second LoRA file. Choose (none) to use the node as a rank/dtype converter for one LoRA."}),
            "weight_2": ("FLOAT", {"default": 50.0, "min": -400.0, "max": 400.0, "step": 0.1, "tooltip": "Weight in percent. Example: 30 and 70 becomes a 30:70 weighted delta merge."}),
            "output_rank": (["Auto", "16", "32", "64", "128", "256", "Custom"], {"default": "Auto", "tooltip": "Auto follows Compression (or Krea2 defaults). A selected rank caps every output layer, including energy and none compression."}),
            "filename_prefix": ("STRING", {"default": "loras/LoRA_Blend_One", "tooltip": "Output filename prefix under ComfyUI/output. A unique numeric suffix is added automatically."}),
        }
        advanced_common = {"advanced": True}
        optional = {}
        for idx in range(3, 9):
            optional[f"lora_{idx}"] = (choices, {"default": NONE, "tooltip": f"Optional LoRA {idx}. Up to eight LoRAs can be merged in one node.", **advanced_common})
            optional[f"weight_{idx}"] = ("FLOAT", {"default": 0.0, "min": -400.0, "max": 400.0, "step": 0.1, "tooltip": f"Weight for optional LoRA {idx}, in percent.", **advanced_common})

        optional.update({
            "merge_method": (["Auto", "Weighted", "TIES", "DARE", "DELLA", "Krea2 Gate-Aware"], {"default": "Auto", "tooltip": "Krea2 Gate-Aware is a two-LoRA profile with direct percentages and gate conflict handling. Other methods keep their existing behavior.", **advanced_common}),
            "weight_mode": (["normalize", "direct"], {"default": "normalize", "tooltip": "Standard methods: normalize or direct. Krea2 Gate-Aware always uses percentages directly (50 = 0.5).", **advanced_common}),
            "compression_mode": (["auto", "energy", "fixed_rank", "max_input_rank", "none"], {"default": "auto", "tooltip": "Compression for standard methods. Weighted none is exact with Output Rank Auto; a lower selected rank compresses. Krea2 uses its dedicated controls.", **advanced_common}),
            "custom_rank": ("INT", {"default": 32, "min": 1, "max": 2048, "step": 1, "tooltip": "Used only when Output Rank is Custom.", **advanced_common}),
            "energy_target": ("FLOAT", {"default": 0.999, "min": 0.50, "max": 1.0, "step": 0.0001, "round": 0.0001, "tooltip": "Numerical delta energy target for Auto/Energy compression. This is not an image-quality percentage.", **advanced_common}),
            "auto_rank_cap": ("INT", {"default": 256, "min": 8, "max": 1024, "step": 8, "tooltip": "Safety cap for randomized SVD used by TIES/DARE/DELLA Auto rank. If the energy target is not reached, the report records the shortfall.", **advanced_common}),
            "density": ("FLOAT", {"default": 0.90, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "TIES: fraction kept by magnitude. DARE: Bernoulli keep probability. DELLA: center of the magnitude-ranked keep-probability range.", **advanced_common}),
            "sign_consensus": (["auto", "on", "off"], {"default": "auto", "tooltip": "auto: ON for TIES/DELLA and OFF for DARE. Sign consensus removes contributions pointing against the elected weighted sign at each element.", **advanced_common}),
            "della_half_width": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.49, "step": 0.01, "tooltip": "DELLA keep-probability half-width. density±half_width must stay inside (0,1]. Larger-magnitude elements receive higher keep probability.", **advanced_common}),
            "output_scale": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01, "tooltip": "Global multiplier applied after source weights. 1.0 preserves the requested weighted merge magnitude.", **advanced_common}),
            "output_dtype": (["auto", "fp32", "fp16", "bf16"], {"default": "auto", "tooltip": "Computation always uses FP32. auto preserves a common input factor dtype when possible; mixed or unsupported input dtypes save as FP32.", **advanced_common}),
            "device": (["auto", "cuda", "cpu"], {"default": "auto", "tooltip": "auto estimates per-layer working memory and uses CUDA only when there is comfortable free VRAM. CPU is the safe fallback.", **advanced_common}),
            "fallback_policy": (["auto_cpu", "error"], {"default": "auto_cpu", "tooltip": "On CUDA OOM, auto_cpu retries the same layer on CPU without changing merge method, density, rank, or dtype settings.", **advanced_common}),
            "unmatched_policy": (["scaled_carry", "drop", "error"], {"default": "scaled_carry", "tooltip": "scaled_carry keeps layers present in only some LoRAs with their original global source weight. drop removes them. error stops the merge.", **advanced_common}),
            "compatibility_mode": (["safe", "allow_disjoint"], {"default": "safe", "tooltip": "safe rejects obviously different architecture families when there are no shared targets. allow_disjoint is for known-compatible adapters targeting disjoint layer sets.", **advanced_common}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF, "step": 1, "tooltip": "Local deterministic seed for DARE, DELLA, and randomized SVD. It does not modify PyTorch's global RNG state.", **advanced_common}),
            "gate_conflict_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Diagnostic only. Gate-like targets with pairwise cosine below -threshold are counted as conflicts; Auto does not alter gate strength.", **advanced_common}),
            "hash_inputs": ("BOOLEAN", {"default": True, "tooltip": "Store SHA-256 hashes of source files in the merge recipe for reproducibility. Disable only when hashing very large files is undesirable.", **advanced_common}),
            "write_report": ("BOOLEAN", {"default": True, "tooltip": "Write a .report.json sidecar containing per-layer ranks, energy estimates, devices, warnings, and source recipe.", **advanced_common}),
            "krea2_gate_strength": ("FLOAT", {"default": 0.5, "min": -4.0, "max": 4.0, "step": 0.01, "tooltip": "Krea2 only: multiplier for matched gate layers.", **advanced_common}),
            "krea2_opposite_strategy": (["pick_strongest", "zero_out"], {"default": "pick_strongest", "tooltip": "Krea2 only: resolve opposing matched gate deltas.", **advanced_common}),
            "krea2_opposite_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Krea2 only: gate deltas conflict when cosine is below the negative threshold.", **advanced_common}),
            "krea2_gate_rank": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1, "tooltip": "Krea2 only: rank for compressed gate layers; zero retains the input maximum for zeroed gates.", **advanced_common}),
            "krea2_balance_mode": (["off", "per_layer_norm"], {"default": "off", "tooltip": "Krea2 only: optionally equalize matched layer delta norms after source strengths.", **advanced_common}),
            "krea2_gate_compression": (["off", "attn_to_gate_only", "all_gates"], {"default": "attn_to_gate_only", "tooltip": "Krea2 only: compress non-conflicting attention gates or all gates.", **advanced_common}),
            "krea2_normal_compression": (["off", "max_input_rank"], {"default": "off", "tooltip": "Krea2 only: optionally compress ordinary matched layers to the larger input rank.", **advanced_common}),
        })
        for idx in range(1, 9):
            optional[f"enabled_{idx}"] = ("BOOLEAN", {"default": True, "tooltip": f"Enable LoRA {idx}. Disabled slots keep their file and weight but are excluded from blending."})
        return {"required": required, "optional": optional}

    @classmethod
    def VALIDATE_INPUTS(cls, output_rank, custom_rank=32, merge_method="Auto", compression_mode="auto", density=0.90, della_half_width=0.05, **kwargs):
        if output_rank == "Custom" and int(custom_rank) < 1:
            return "Custom output rank must be at least 1."
        if merge_method in ("DARE", "DELLA") and not (0.0 < float(density) <= 1.0):
            return "DARE/DELLA density must be in (0,1]."
        if merge_method == "DELLA":
            lo, hi = float(density) - float(della_half_width), float(density) + float(della_half_width)
            if not (0.0 < lo <= hi <= 1.0):
                return "DELLA requires 0 < density-half_width <= density+half_width <= 1."
        if compression_mode == "none" and merge_method not in ("Auto", "Weighted", "Krea2 Gate-Aware"):
            return "compression=none is available only for Weighted merge."
        return True

    def merge(self, lora_1, weight_1, lora_2, weight_2, output_rank, filename_prefix, **kwargs):
        try:
            return self._merge(lora_1, weight_1, lora_2, weight_2, output_rank, filename_prefix, **kwargs)
        except Exception as error:
            if _is_interruption(error):
                raise
            publish_failure(error, _error_slots(lora_1, lora_2, kwargs))
            raise

    def _merge(
        self,
        lora_1,
        weight_1,
        lora_2,
        weight_2,
        output_rank,
        filename_prefix,
        **kwargs,
    ):
        rows = [(lora_1, weight_1), (lora_2, weight_2)]
        for idx in range(3, 9):
            rows.append((kwargs.get(f"lora_{idx}", NONE), kwargs.get(f"weight_{idx}", 0.0)))
        selected = _selected_slots(rows, kwargs)
        if not selected:
            raise MergeError("no_sources", [], "LoRA Blend One: select at least one enabled LoRA.")
        krea2 = kwargs.get("merge_method") == "Krea2 Gate-Aware"
        if krea2 and len(selected) != 2:
            raise MergeError("krea2_count", [name for _, name, _ in selected],
                             "Krea2 Gate-Aware requires exactly two selected LoRA slots.")
        coalesced = [(name, weight) for _, name, weight in selected] if krea2 else _coalesce_duplicate_sources(selected)
        names = [x[0] for x in coalesced]
        raw_weights = [x[1] for x in coalesced]
        try:
            weights = [float(weight) / 100.0 for weight in raw_weights] if krea2 else resolve_weights(raw_weights, kwargs.get("weight_mode", "normalize"))
        except ValueError as error:
            raise MergeError("invalid_weights", [], str(error)) from error

        infos = []
        hash_inputs = bool(kwargs.get("hash_inputs", True))
        for name in names:
            try:
                path = folder_paths.get_full_path_or_raise("loras", name)
                infos.append(scan_lora(path, compute_hash=hash_inputs))
            except Exception as error:
                if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)) or _is_interruption(error):
                    raise
                if isinstance(error, MergeError):
                    raise MergeError(error.code, [name], error.detail) from error
                code = "missing_file" if isinstance(error, FileNotFoundError) else "read_failed"
                raise MergeError(code, [name], f"{name}: {error}") from error

        rank_value = _rank_from_combo(output_rank, kwargs.get("custom_rank", 32))
        settings = MergeSettings(
            method=kwargs.get("merge_method", "Auto"),
            compression_mode=kwargs.get("compression_mode", "auto"),
            output_rank=rank_value,
            energy_target=float(kwargs.get("energy_target", 0.999)),
            auto_rank_cap=int(kwargs.get("auto_rank_cap", 256)),
            output_dtype=kwargs.get("output_dtype", "auto"),
            device=kwargs.get("device", "auto"),
            fallback_policy=kwargs.get("fallback_policy", "auto_cpu"),
            unmatched_policy=kwargs.get("unmatched_policy", "scaled_carry"),
            density=float(kwargs.get("density", 0.90)),
            sign_consensus=kwargs.get("sign_consensus", "auto"),
            della_half_width=float(kwargs.get("della_half_width", 0.05)),
            seed=int(kwargs.get("seed", 0)),
            output_scale=float(kwargs.get("output_scale", 1.0)),
            gate_conflict_threshold=float(kwargs.get("gate_conflict_threshold", 0.05)),
            krea2_gate_strength=float(kwargs.get("krea2_gate_strength", 0.5)),
            krea2_opposite_strategy=kwargs.get("krea2_opposite_strategy", "pick_strongest"),
            krea2_opposite_threshold=float(kwargs.get("krea2_opposite_threshold", 0.05)),
            krea2_gate_rank=int(kwargs.get("krea2_gate_rank", 32)),
            krea2_balance_mode=kwargs.get("krea2_balance_mode", "off"),
            krea2_gate_compression=kwargs.get("krea2_gate_compression", "attn_to_gate_only"),
            krea2_normal_compression=kwargs.get("krea2_normal_compression", "off"),
        )

        union_target_count = len(set().union(*(set(i.layers) for i in infos)))
        progress_bar = comfy_utils.ProgressBar(union_target_count) if comfy_utils else None
        progress_state = {"done": 0, "last": 0}

        def cancel_check():
            if model_management is not None:
                model_management.throw_exception_if_processing_interrupted()

        def progress(done, total):
            # Engine reports union-target progress. Map it onto a monotonic Comfy progress bar.
            if progress_bar is None:
                return
            if done > progress_state["last"]:
                progress_bar.update(done - progress_state["last"])
                progress_state["last"] = done

        try:
            tensors, merge_report = merge_sources(
                infos,
                weights,
                settings,
                compatibility_mode=kwargs.get("compatibility_mode", "safe"),
                progress=progress,
                cancel_check=cancel_check,
            )
        except MergeError as error:
            # File paths identify the right slot even when several files share a basename.
            source_names = [name for name, info in zip(names, infos) if str(info.path) in (error.source_paths or ())]
            if source_names:
                raise MergeError(error.code, source_names, error.detail) from error
            raise
        recipe = build_recipe(infos, weights, settings, merge_report)
        metadata = build_metadata(recipe)
        report_dict = report_to_dict(merge_report)
        report_dict["recipe"] = recipe
        warnings_before_save = len(merge_report.warnings)

        try:
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory()
            )
            os.makedirs(full_output_folder, exist_ok=True)
            output_name = f"{filename}_{counter:05}_.safetensors"
            output_path = Path(full_output_folder) / output_name
            save_verified(
                tensors,
                output_path,
                metadata=metadata,
                report=report_dict,
                write_report=bool(kwargs.get("write_report", True)),
            )
        except Exception as error:
            if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)) or _is_interruption(error):
                raise
            raise MergeError("save_failed", [], str(error)) from error

        rel_path = str(Path(subfolder) / output_name) if subfolder else output_name
        min_rank, max_rank = merge_report.rank_range()
        warning_count = len(merge_report.warnings) + sum(len(x.warnings) for x in merge_report.layer_reports)
        model_label = ("MOSS-Image" if all(info.architecture_hint == "moss_image" for info in infos)
                       else "Krea2 (Gate-Aware)" if merge_report.method == "Krea2 Gate-Aware" else "Standard LoRA")
        info = (
            f"Saved {rel_path} | model={model_label} | method={merge_report.method} | sources={len(infos)} | "
            f"layers={len(merge_report.layer_reports)} | rank={min_rank}-{max_rank} | "
            f"gate_conflicts={merge_report.gate_conflicts} | warnings={warning_count}"
        )
        save_warnings = merge_report.warnings[warnings_before_save:]
        if save_warnings:
            info += "\n" + "\n".join(save_warnings)
        return {"ui": {"text": [info], "model_detection": [model_label]}, "result": (info,)}


class LoRABlendSources:
    """Select LoRA files and percentages for a Blend One merge."""

    DESCRIPTION = "Select 1-8 LoRAs and their percentages. Open Advanced to add LoRAs 3-8."
    CATEGORY = LoRABlendOne.CATEGORY
    RETURN_TYPES = ("LORA_BLEND_SOURCES",)
    RETURN_NAMES = ("sources",)
    OUTPUT_TOOLTIPS = ("Selected LoRA files and percentages for Blend One.",)
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        schema = LoRABlendOne.INPUT_TYPES()
        return {
            "required": {key: schema["required"][key] for key in ("lora_1", "weight_1", "lora_2", "weight_2")},
            "optional": {key: value for key, value in schema["optional"].items() if key in SOURCE_OPTIONAL_KEYS},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, lora_1=NONE, lora_2=NONE, **kwargs):
        names = [lora_1, lora_2] + [kwargs.get(f"lora_{idx}", NONE) for idx in range(3, 9)]
        choices = _lora_choices()
        for idx, name in enumerate(names, start=1):
            if not kwargs.get(f"enabled_{idx}", True) or not name or name == NONE:
                continue
            if name not in choices:
                return f"LoRA {idx}: selected file is unavailable: {name}. Choose an installed LoRA or turn this slot off."
            try:
                folder_paths.get_full_path_or_raise("loras", name)
            except FileNotFoundError:
                return f"LoRA {idx}: selected file is unavailable: {name}. Choose an installed LoRA or turn this slot off."
        return True

    def select(self, lora_1, weight_1, lora_2, weight_2, **kwargs):
        try:
            rows = [(lora_1, weight_1), (lora_2, weight_2)]
            for idx in range(3, 9):
                rows.append((kwargs.get(f"lora_{idx}", NONE), kwargs.get(f"weight_{idx}", 0.0)))
            selected = _selected_slots(rows, kwargs)
            if not selected:
                raise MergeError("no_sources", [], "LoRA Blend Sources: select at least one enabled LoRA.")
            return (tuple(selected),)
        except Exception as error:
            publish_failure(error, _error_slots(lora_1, lora_2, kwargs))
            raise


class LoRABlendOneSplit(LoRABlendOne):
    """Merge the LoRAs selected by LoRABlendSources."""

    DESCRIPTION = "Merge selected LoRAs into one verified safetensors file. Open Advanced for less-used numerical and diagnostic controls."

    @classmethod
    def INPUT_TYPES(cls):
        schema = LoRABlendOne.INPUT_TYPES()
        visible_settings = {"merge_method", "weight_mode", "compression_mode", "output_dtype"}
        optional = {}
        for key, (kind, options) in schema["optional"].items():
            if key in SOURCE_OPTIONAL_KEYS:
                continue
            options = dict(options)
            if key in visible_settings:
                options.pop("advanced", None)
            if key == "write_report":
                options["default"] = False
            optional[key] = (kind, options)
        return {
            "required": {
                "sources": ("LORA_BLEND_SOURCES",),
                "output_rank": schema["required"]["output_rank"],
                "filename_prefix": schema["required"]["filename_prefix"],
            },
            "optional": optional,
        }

    def merge(self, sources, output_rank, filename_prefix, **kwargs):
        slots = {slot: (name, weight) for slot, name, weight in sources}
        extra = dict(kwargs)
        extra.setdefault("write_report", False)
        for idx in range(3, 9):
            extra[f"lora_{idx}"], extra[f"weight_{idx}"] = slots.get(idx, (NONE, 0.0))
        lora_1, weight_1 = slots.get(1, (NONE, 0.0))
        lora_2, weight_2 = slots.get(2, (NONE, 0.0))
        return super().merge(lora_1, weight_1, lora_2, weight_2, output_rank, filename_prefix, **extra)


NODE_CLASS_MAPPINGS = {
    "LoRABlendOne": LoRABlendOne,
    "LoRABlendSources": LoRABlendSources,
    "LoRABlendOneSplit": LoRABlendOneSplit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoRABlendOne": "LoRA Blend One (Legacy)",
    "LoRABlendSources": "LoRA Blend Sources",
    "LoRABlendOneSplit": "LoRA Blend One",
}
