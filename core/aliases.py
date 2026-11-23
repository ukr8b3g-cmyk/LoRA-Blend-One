"""Align equivalent adapter targets before counting shared layers. Raw tensor keys stay on LayerSpec."""
from __future__ import annotations

import re
from dataclasses import replace

from .errors import MergeError
from .types import SourceInfo


# Projection aliases from ComfyUI's krea2_to_diffusers. Limit these translations
# to Krea2: e.g. a Qwen transformer_blocks.0.attn.to_q is a different module.
_KREA_BLOCKS = {
    "transformer_blocks": "blocks",
    "text_fusion.layerwise_blocks": "txtfusion.layerwise_blocks",
    "text_fusion.refiner_blocks": "txtfusion.refiner_blocks",
}
_KREA_PROJECTIONS = {
    "attn.to_q": "attn.wq", "attn.to_k": "attn.wk", "attn.to_v": "attn.wv",
    "attn.to_gate": "attn.gate", "attn.to_out": "attn.wo", "attn.to_out.0": "attn.wo",
    "ff.gate": "mlp.gate", "ff.up": "mlp.up", "ff.down": "mlp.down",
}
_KREA_BASIC = {
    "img_in": "first", "time_embed.linear_1": "tmlp.0", "time_embed.linear_2": "tmlp.2",
    "time_mod_proj": "tproj.1", "txt_in.linear_1": "txtmlp.1", "txt_in.linear_2": "txtmlp.3",
    "text_fusion.projector": "txtfusion.projector", "final_layer.linear": "last.linear",
}
_CLIP_MODULES = ("mlp.fc1", "mlp.fc2", "self_attn.q_proj", "self_attn.k_proj",
                 "self_attn.v_proj", "self_attn.out_proj")


def _family(info):
    metadata = info.metadata.get("ss_base_model_version", "").lower().replace("_", "-")
    base = str(info.adapter_config.get("base_model_name_or_path", "")).rsplit("/", 1)[-1].lower()
    if metadata == "krea2" or base.startswith("krea2"):
        return "krea2"
    if metadata.startswith("qwen-image") or base.startswith("qwen-image"):
        return "qwen"
    if any(re.search(r"(?:txtfusion|text_fusion)[._](?:layerwise|refiner)_blocks[._]", t) for t in info.layers):
        return "krea2"
    if info.architecture_hint == "qwen_image":
        return "qwen"
    if info.architecture_hint == "sd_family" and all(t.startswith("lora_unet_") for t in info.layers):
        # Comfy exposes this prefix for every diffusion model, including Krea2.
        return None
    return None if info.architecture_hint in ("unknown", "gated_transformer") else info.architecture_hint


def _clip_groups(observed):
    blocks = {m[1] for t in observed if (m := re.search(r"text_model[._]encoder[._]layers[._](\d+)[._]", t))}
    for i in blocks:
        for module in _CLIP_MODULES:
            path = f"text_model.encoder.layers.{i}.{module}"
            flat = path.replace(".", "_")
            for prefix, kohya in (("text_encoder.", ("lora_te_", "lora_te1_")),
                                  ("text_encoder_2.", ("lora_te2_",))):
                name = prefix + path
                yield name, [name, *(p + flat for p in kohya)]


def _krea_groups(observed):
    blocks = {m[1] for t in observed if (m := re.search(r"blocks[._](\d+)[._]", t))}
    paths = list(_KREA_BASIC.items())
    for i in blocks:
        for group, native_group in _KREA_BLOCKS.items():
            for proj, native_proj in _KREA_PROJECTIONS.items():
                paths.append((f"{group}.{i}.{proj}", f"{native_group}.{i}.{native_proj}"))
    for path, native in paths:
        name = "diffusion_model." + native
        yield name, [name, "lora_unet_" + native.replace(".", "_"), path, "transformer." + path,
                     "diffusion_model." + path, "lycoris_" + path.replace(".", "_")]


def _qwen_groups(observed):
    # Flattening alone is not reversible. Generate flattened spellings from
    # observed dotted paths, never infer dot positions from an underscore.
    for target in observed:
        prefix, _, suffix = target.partition(".")
        path = suffix if prefix in ("transformer", "diffusion_model") else target
        if path.startswith("transformer_blocks."):
            yield path, [path, "transformer." + path, "diffusion_model." + path,
                         "lycoris_" + path.replace(".", "_")]


def _matching_names(groups, observed):
    present = {}
    for physical, aliases in groups:
        matches = observed.intersection(aliases)
        if matches:
            present.setdefault(physical, set()).update(matches)
    return present


def align_target_aliases(infos: list[SourceInfo], method: str) -> list[SourceInfo]:
    if any(info.architecture_hint == "moss_image" for info in infos):
        return infos
    observed = {t for info in infos for t in info.layers}
    families = {_family(info) for info in infos} - {None}
    if method == "Krea2 Gate-Aware":
        families.add("krea2")
    groups = _matching_names(_clip_groups(observed), observed)
    sources = [i.name for i in infos]
    paths = [str(i.path) for i in infos]
    for family, make_groups in (("krea2", _krea_groups), ("qwen", _qwen_groups)):
        if families and families != {family}:
            continue
        for physical, keys in _matching_names(make_groups(observed), observed).items():
            if len(keys) > 1 and not families:
                raise MergeError("ambiguous_target", sources,
                                 f"Cannot identify the model for possible aliases {', '.join(sorted(keys))}. "
                                 "Use matching layer names or supply model-family metadata.", source_paths=paths)
            if families:
                groups[physical] = keys

    names = {}
    for physical, keys in groups.items():
        if len(keys) == 1:
            continue
        for key in keys:
            previous = names.setdefault(key, physical)
            if previous != physical:
                raise MergeError("ambiguous_target", sources, f"{key!r} could mean {previous!r} or {physical!r}",
                                 source_paths=paths)
    if not names:
        return infos
    aligned = []
    for info in infos:
        layers = {}
        originals = {}
        for original, spec in info.layers.items():
            output = names.get(original, original)
            if output in layers:
                raise MergeError("duplicate_target", [info.name],
                                 f"{info.name}: {originals[output]!r} and {original!r} address the same layer",
                                 source_paths=[str(info.path)])
            layers[output] = replace(spec, target=output) if original != output else spec
            originals[output] = original
        aligned.append(replace(info, layers=layers))
    return aligned
