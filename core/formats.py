from __future__ import annotations

import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open

from .types import FactorStyle, LayerSpec, SourceInfo
from .moss import normalize_moss_layers
from .errors import MergeError


# Ordered from most specific to least specific to avoid suffix ambiguity.
FACTOR_STYLES: tuple[FactorStyle, ...] = (
    FactorStyle("peft_named_default", ".lora_A.default.weight", ".lora_B.default.weight"),
    FactorStyle("transformers", ".lora_linear_layer.down.weight", ".lora_linear_layer.up.weight"),
    FactorStyle("peft", ".lora_A.weight", ".lora_B.weight"),
    FactorStyle("regular", ".lora_down.weight", ".lora_up.weight"),
    FactorStyle("dot", ".lora.down.weight", ".lora.up.weight"),
    FactorStyle("diffusers_underscore", "_lora.down.weight", "_lora.up.weight"),
    FactorStyle("mochi", ".lora_A", ".lora_B"),
)

UNSUPPORTED_SUFFIXES = (
    ".dora_scale",
    ".lora_mid.weight",
    ".hada_w1_a",
    ".hada_w1_b",
    ".hada_w2_a",
    ".hada_w2_b",
    ".diff",
    ".diff_b",
    ".set_weight",
)
UNSUPPORTED_SUBSTRINGS = (".lokr_", "lora_magnitude_vector")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_adapter_config(path: Path) -> dict:
    cfg = path.parent / "adapter_config.json"
    if not cfg.is_file():
        return {}
    try:
        with cfg.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _find_style(key: str):
    for style in FACTOR_STYLES:
        if key.endswith(style.down_suffix):
            return style, key[: -len(style.down_suffix)], "down"
        if key.endswith(style.up_suffix):
            return style, key[: -len(style.up_suffix)], "up"
    return None


def _adapter_pattern_value(mapping, target: str):
    if not isinstance(mapping, dict):
        return None
    if target in mapping:
        return mapping[target]
    # PEFT pattern keys are commonly module suffixes. Prefer the longest match.
    matches = [(len(k), v) for k, v in mapping.items() if isinstance(k, str) and target.endswith(k)]
    if matches:
        matches.sort(reverse=True, key=lambda x: x[0])
        return matches[0][1]
    return None


def _effective_scale(handle, keys: set[str], target: str, rank: int, adapter_config: dict) -> tuple[float, str | None]:
    alpha_key = f"{target}.alpha"
    if alpha_key in keys:
        alpha = float(handle.get_tensor(alpha_key).item())
        return alpha / float(rank), alpha_key

    alpha = _adapter_pattern_value(adapter_config.get("alpha_pattern"), target)
    if alpha is None:
        alpha = adapter_config.get("lora_alpha")

    rank_cfg = _adapter_pattern_value(adapter_config.get("rank_pattern"), target)
    if rank_cfg is None:
        rank_cfg = adapter_config.get("r", rank)

    if alpha is not None:
        alpha = float(alpha)
        use_rslora = bool(adapter_config.get("use_rslora", False))
        denom = math.sqrt(float(rank_cfg or rank)) if use_rslora else float(rank_cfg or rank)
        if denom <= 0:
            raise ValueError(f"Invalid rank/scale configuration for {target!r}")
        return alpha / denom, None

    # This matches current ComfyUI LoRA loading semantics: no alpha tensor => scale 1.0.
    return 1.0, None


def _validate_shapes(target: str, down_shape: tuple[int, ...], up_shape: tuple[int, ...]) -> tuple[int, int, str, tuple[int, ...], tuple[int, ...]]:
    if len(down_shape) < 2 or len(up_shape) < 2:
        raise ValueError(f"{target}: LoRA factors must be at least 2D, got {down_shape} / {up_shape}")
    rank = down_shape[0]
    if up_shape[1] != rank:
        raise ValueError(f"{target}: rank mismatch, down={down_shape}, up={up_shape}")
    if len(down_shape) != len(up_shape):
        # 2D up + convolutional down is not a standard ComfyUI factor pair.
        raise ValueError(f"{target}: factor dimensionality mismatch, down={down_shape}, up={up_shape}")

    if len(down_shape) == 2:
        out_dim = up_shape[0]
        in_flat = down_shape[1]
        return out_dim, in_flat, "linear", down_shape[1:], up_shape[2:]

    if any(int(x) != 1 for x in up_shape[2:]):
        raise ValueError(
            f"{target}: convolutional LoRA is supported only when the up factor has 1x1 spatial kernel; up={up_shape}"
        )
    out_dim = up_shape[0]
    in_flat = math.prod(down_shape[1:])
    return out_dim, in_flat, "conv", down_shape[1:], up_shape[2:]


def architecture_hint(targets: Iterable[str]) -> str:
    joined = "\n".join(targets)
    if "double_blocks" in joined or "single_blocks" in joined:
        return "flux"
    if "transformer_blocks" in joined and ("img_mlp" in joined or "txt_mlp" in joined or "add_q_proj" in joined):
        return "qwen_image"
    if ".self_attn." in joined and ".cross_attn." in joined and ("diffusion_model.blocks" in joined or ".blocks." in joined):
        return "wan"
    if any(x in joined for x in (".attn.to_gate", ".attn.gate", ".ff.gate", ".mlp.gate")):
        return "gated_transformer"
    if "lora_unet_" in joined or "down_blocks" in joined or "up_blocks" in joined:
        return "sd_family"
    if "diffusion_model.layers" in joined:
        return "dit"
    return "unknown"


def scan_lora(path: str | Path, *, compute_hash: bool = True) -> SourceInfo:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    adapter_config = _load_adapter_config(path)
    unsupported: list[str] = []
    warnings: list[str] = []
    layers: dict[str, LayerSpec] = {}
    factor_dtypes: set[str] = set()

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        keyset = set(keys)
        metadata = handle.metadata() or {}

        for key in keys:
            if key.endswith(UNSUPPORTED_SUFFIXES) or any(marker in key for marker in UNSUPPORTED_SUBSTRINGS):
                unsupported.append(key)

        factor_candidates: dict[tuple[str, str], dict[str, str]] = {}
        for key in keys:
            parsed = _find_style(key)
            if parsed is None:
                continue
            style, target, side = parsed
            factor_candidates.setdefault((style.name, target), {})[side] = key

        for (style_name, target), pair in factor_candidates.items():
            style = next(s for s in FACTOR_STYLES if s.name == style_name)
            if "down" not in pair or "up" not in pair:
                warnings.append(f"Incomplete factor pair ignored: {target} ({style_name})")
                continue
            down_key, up_key = pair["down"], pair["up"]
            down_slice = handle.get_slice(down_key)
            up_slice = handle.get_slice(up_key)
            down_shape = tuple(int(x) for x in down_slice.get_shape())
            up_shape = tuple(int(x) for x in up_slice.get_shape())
            down_dtype = down_slice.get_dtype()
            up_dtype = up_slice.get_dtype()
            if down_dtype != up_dtype:
                warnings.append(f"Mixed factor dtype for {target}: {down_dtype}/{up_dtype}; compute will use FP32")
            factor_dtypes.update((down_dtype, up_dtype))
            rank = int(down_shape[0])
            out_dim, in_flat, layout, down_tail, up_tail = _validate_shapes(target, down_shape, up_shape)
            scale, alpha_key = _effective_scale(handle, keyset, target, rank, adapter_config)
            if not math.isfinite(scale):
                raise MergeError("nonfinite_input", [path.name],
                                 f"{path.name}: {target}: alpha/effective scale contains NaN or Inf",
                                 source_paths=[str(path)])

            if target in layers:
                raise ValueError(f"Ambiguous duplicate LoRA factor styles for target {target!r}")
            layers[target] = LayerSpec(
                target=target,
                style=style,
                down_key=down_key,
                up_key=up_key,
                alpha_key=alpha_key,
                rank=rank,
                down_shape=down_shape,
                up_shape=up_shape,
                dtype_name=down_dtype if down_dtype == up_dtype else "MIXED",
                effective_scale=float(scale),
                layout=layout,
                out_dim=out_dim,
                in_flat=in_flat,
                down_tail=tuple(int(x) for x in down_tail),
                up_tail=tuple(int(x) for x in up_tail),
            )

        moss_layers = normalize_moss_layers(layers, handle, metadata, adapter_config)
        if moss_layers is None:
            consumed = {key for spec in layers.values()
                        for key in (spec.down_key, spec.up_key, spec.alpha_key) if key is not None}
            unused = sorted(keyset - consumed)
            unsupported.extend(unused)
        if moss_layers is not None:
            layers = moss_layers

    if not layers:
        raise ValueError(f"No supported Standard LoRA factor pairs found in {path.name}")

    if bool(adapter_config.get("use_dora", False)):
        unsupported.append("adapter_config.json: use_dora=true")

    info = SourceInfo(
        path=path,
        name=path.name,
        layers=layers,
        metadata={str(k): str(v) for k, v in metadata.items()},
        adapter_config=adapter_config,
        unsupported=sorted(set(unsupported)),
        warnings=warnings,
        factor_dtypes=factor_dtypes,
        architecture_hint="moss_image" if moss_layers is not None else architecture_hint(layers.keys()),
    )
    if compute_hash:
        info.sha256 = sha256_file(path)
    return info


class OpenedSources:
    def __init__(self, infos: list[SourceInfo]):
        self.infos = infos
        self._stack = ExitStack()
        self.handles = []

    def __enter__(self):
        self.handles = [
            self._stack.enter_context(safe_open(str(info.path), framework="pt", device="cpu"))
            for info in self.infos
        ]
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._stack.__exit__(exc_type, exc, tb)

    def read_factors(self, source_index: int, spec: LayerSpec) -> tuple[torch.Tensor, torch.Tensor]:
        handle = self.handles[source_index]
        info = self.infos[source_index]

        def checked(tensor, key):
            if not torch.isfinite(tensor).all().item():
                raise MergeError("nonfinite_input", [info.name],
                                 f"{info.name}: {key}: input factor contains NaN or Inf",
                                 source_paths=[str(info.path)])
            return tensor

        if spec.components:
            ups, downs = [], []
            for part in spec.components:
                down = checked(handle.get_tensor(part.down_key), part.down_key)
                up = checked(handle.get_slice(part.up_key)[part.row_start:part.row_start + part.row_count, :], part.up_key)
                if part.columns is not None:
                    up = up[:, part.columns]
                    down = down[list(part.columns), :]
                ups.append(up.float() * part.scale)
                downs.append(down.float())
            return torch.cat(ups, dim=1), torch.cat(downs, dim=0)
        down = checked(handle.get_tensor(spec.down_key), spec.down_key)
        up = checked(handle.get_tensor(spec.up_key), spec.up_key)
        return up, down
