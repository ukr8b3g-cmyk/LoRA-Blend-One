from __future__ import annotations

import re

from .types import FactorComponent, FactorStyle, LayerSpec


OUTPUT_STYLE = FactorStyle("peft", ".lora_A.weight", ".lora_B.weight")
JOINT = re.compile(r"layers\.\d+\.proj\.(qkv|q|k|v|o|ffn_up_gate|ffn_down)")
TEXT = re.compile(r"text_embedder\.blocks\.\d+\.(qkv|proj|ffn_gate|ffn_up|ffn_down)")
LINEARS = {"video_embedder", "text_embedder.proj", "time_embedding.0", "time_embedding.2",
           "shared_block_modulation", "video_head_modulation", "final_linear_video"}


def module_path(target: str) -> str:
    prefixes = ("base_model.model.", "transformer.", "diffusion_model.")
    while target.startswith(prefixes):
        target = next(target[len(p):] for p in prefixes if target.startswith(p))
    for prefix in ("lora_unet_", "lycoris_", "lora_transformer_"):
        if target.startswith(prefix):
            target = target[len(prefix):]
            match = re.fullmatch(r"layers_(\d+)_proj_(qkv|q|k|v|o|ffn_up_gate|ffn_down)", target)
            if match:
                return f"layers.{match[1]}.proj.{match[2]}"
            match = re.fullmatch(r"text_embedder_blocks_(\d+)_(qkv|proj|ffn_gate|ffn_up|ffn_down)", target)
            if match:
                return f"text_embedder.blocks.{match[1]}.{match[2]}"
            return next((name for name in LINEARS if name.replace(".", "_") == target), target)
    return target


def normalize_moss_layers(layers: dict[str, LayerSpec], handle, metadata: dict, adapter_config: dict) -> dict[str, LayerSpec] | None:
    paths = {target: module_path(target) for target in layers}
    declared = (str(metadata.get("ss_base_model_version", "")).lower().replace("-", "_").startswith("moss_image")
                or str(metadata.get("modelspec.architecture", "")).lower().startswith("moss/")
                or metadata.get("lbo_model_family") == "moss_image"
                or "moss-image" in str(adapter_config.get("base_model_name_or_path", "")).lower())
    signature = any(JOINT.fullmatch(paths[target]) and spec.layout == "linear" and
                    (paths[target].rsplit(".", 1)[-1], spec.out_dim, spec.in_flat) in {
                        ("qkv", 9216, 3072), ("ffn_up_gate", 16384, 3072), ("ffn_down", 3072, 8192)}
                    for target, spec in layers.items())
    if not declared and not signature:
        return None

    consumed = {key for spec in layers.values() for key in (spec.down_key, spec.up_key, spec.alpha_key) if key is not None}
    unused = sorted(set(handle.keys()) - consumed)
    if unused:
        raise ValueError(f"MOSS-Image contains unsupported or incomplete adapter tensors: {', '.join(unused[:5])}")

    groups = {}
    seen = {}
    for target, spec in layers.items():
        path = paths[target]
        if not (JOINT.fullmatch(path) or TEXT.fullmatch(path) or path in LINEARS):
            raise ValueError(f"Unsupported MOSS-Image target: {target}")
        if spec.layout != "linear":
            raise ValueError(f"MOSS-Image requires linear LoRA factors: {target}")
        if any(handle.get_slice(key).get_dtype() not in ("F32", "F16", "BF16") for key in (spec.up_key, spec.down_key)):
            raise ValueError(f"MOSS-Image supports FP32, FP16 or BF16 LoRA factors: {target}")
        if path in seen:
            raise ValueError(f"Duplicate MOSS-Image aliases: {seen[path]} and {target}")
        seen[path] = target
        fused = bool(JOINT.fullmatch(path) and path.endswith(".qkv"))
        if fused and spec.out_dim % 3:
            raise ValueError(f"MOSS-Image QKV rows are not divisible by three: {target}")
        parts = ("q", "k", "v") if fused else (None,)
        rows = spec.out_dim // 3 if fused else spec.out_dim
        for index, part in enumerate(parts):
            output_path = path[:-3] + part if fused else path
            canonical = f"diffusion_model.{output_path}"
            columns = None
            rank = spec.rank
            # Remove only exactly zero up columns; a block-diagonal QKV export then
            # has the same per-projection rank as its separate q/k/v source.
            if fused or (JOINT.fullmatch(path) and path.rsplit(".", 1)[-1] in ("q", "k", "v")):
                up = handle.get_slice(spec.up_key)[index * rows:(index + 1) * rows, :]
                used = up.ne(0).any(dim=0).nonzero().flatten().tolist()
                if len(used) != spec.rank:
                    columns = tuple(used) if used else (0,)
                    rank = len(columns)
            component = FactorComponent(spec.down_key, spec.up_key, spec.effective_scale,
                                        index * rows, rows, columns)
            groups.setdefault(canonical, []).append((spec, component, rank, rows))

    output = {}
    for target, terms in sorted(groups.items()):
        template = terms[0][0]
        rows = terms[0][3]
        if any(spec.in_flat != template.in_flat or out_rows != rows for spec, _, _, out_rows in terms):
            raise ValueError(f"Incompatible MOSS-Image factors for {target}")
        rank = sum(term[2] for term in terms)
        output[target] = LayerSpec(
            target=target, style=OUTPUT_STYLE,
            down_key=target + OUTPUT_STYLE.down_suffix, up_key=target + OUTPUT_STYLE.up_suffix,
            alpha_key=None, rank=rank, down_shape=(rank, template.in_flat), up_shape=(rows, rank),
            dtype_name=template.dtype_name, effective_scale=1.0, layout="linear", out_dim=rows,
            in_flat=template.in_flat, down_tail=(template.in_flat,), up_tail=(),
            components=tuple(term[1] for term in terms))
    return output
