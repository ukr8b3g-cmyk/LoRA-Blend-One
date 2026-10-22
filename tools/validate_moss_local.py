"""Blend real MOSS LoRAs and check saved files with the installed ComfyUI parser.

Uses CPU and a header-only model for key/shape binding; does not start ComfyUI or
load the base model's weights. Output files are kept in the requested directory.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.engine import merge_sources
from core.formats import scan_lora
from core.recipe import build_metadata, build_recipe, report_to_dict
from core.types import MergeSettings
from core.writer import save_verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--lora", type=Path, action="append", required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32])
    args = parser.parse_args()
    if len(args.lora) != len(args.weights):
        parser.error("Supply one direct coefficient per LoRA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    infos = [scan_lora(path, compute_hash=True) for path in args.lora]
    if any(info.architecture_hint != "moss_image" for info in infos):
        raise ValueError("Every input must be recognized as MOSS-Image")
    results = []
    for rank in args.ranks:
        settings = MergeSettings(method="Weighted", compression_mode="energy", output_rank=rank,
                                 energy_target=0.999, output_dtype="fp32", device="cpu")

        def progress(done, total):
            if done % 42 == 0 or done == total:
                print(f"rank {rank}: {done}/{total}", flush=True)

        tensors, report = merge_sources(infos, args.weights, settings, progress=progress)
        recipe = build_recipe(infos, args.weights, settings, report)
        path = args.output_dir / f"moss_blend_rank{rank}.safetensors"
        save_verified(tensors, path, metadata=build_metadata(recipe), report=report_to_dict(report))
        del tensors
        reread = scan_lora(path, compute_hash=False)
        if set(reread.layers) != set().union(*(info.layers for info in infos)):
            raise AssertionError("Saved targets differ from the normalized input union")
        if not all(spec.rank <= rank and spec.effective_scale == 1. for spec in reread.layers.values()):
            raise AssertionError("Saved rank or alpha is incorrect")
        results.append({"path": str(path), "rank_cap": rank, "layers": len(reread.layers),
                        "bytes": path.stat().st_size, "saved_rank_range": list(report.rank_range())})

    # Import the real parser in CPU mode without inheriting this script's arguments.
    sys.path.insert(0, str(args.comfy_root))
    sys.argv = ["moss-local-check", "--cpu"]
    importlib.import_module("comfy.options").enable_args_parsing()
    lora = importlib.import_module("comfy.lora")
    model_base = importlib.import_module("comfy.model_base")
    with safe_open(str(args.base), framework="pt", device="cpu") as handle:
        header = {("" if key.startswith("diffusion_model.") else "diffusion_model.") + key:
                  torch.empty(tuple(handle.get_slice(key).get_shape()), device="meta") for key in handle.keys()}

    class HeaderModel(model_base.MOSSImage):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.model_config = SimpleNamespace(unet_config={"image_model": "moss_image"})

        def state_dict(self):
            return header

    key_map = lora.model_lora_keys_unet(HeaderModel(), {})
    for result in results:
        tensors = load_file(result["path"], device="cpu")
        patches = lora.load_lora(tensors, key_map)
        if len(patches) != result["layers"]:
            raise AssertionError("ComfyUI did not load every layer")
        consumed = {key for adapter in patches.values() for key in adapter.loaded_keys}
        consumed.update(key for key in tensors if key.endswith(".alpha") and key[:-6] in key_map)
        if consumed != set(tensors):
            raise AssertionError(f"Unloaded tensors: {set(tensors) - consumed}")
        for destination, adapter in patches.items():
            key, offset = (destination, None) if isinstance(destination, str) else destination
            shape = header[key].shape
            expected = tuple(shape) if offset is None else (offset[2], shape[1])
            up, down, alpha = adapter.weights[:3]
            if (up.shape[0], down.shape[1]) != expected:
                raise AssertionError(f"ComfyUI patch shape mismatch: {key}")
        # Exercise the real weight adapter on one ordinary layer and one QKV slice.
        for target in ("diffusion_model.layers.0.proj.o", "diffusion_model.layers.0.proj.q"):
            adapter = patches[key_map[target]]
            up, down, alpha = adapter.weights[:3]
            reference = (up @ down) * (alpha / down.shape[0])
            applied = adapter.calculate_weight(torch.zeros_like(reference), target + ".weight", 1., 1., None, lambda x: x)
            torch.testing.assert_close(applied, reference)
        result["comfy_loaded_layers"] = len(patches)
        result["comfy_weight_apply"] = "PASS (O and Q)"
        print(json.dumps(result, ensure_ascii=False), flush=True)
        del tensors, patches
    for low, high in zip(results, results[1:]):
        if low["rank_cap"] < high["rank_cap"] and low["bytes"] >= high["bytes"]:
            raise AssertionError("Lower rank did not reduce saved size")
    summary = {"sources": [{"name": info.name, "layers": len(info.layers), "sha256": info.sha256} for info in infos],
               "weights": args.weights, "base_header": str(args.base), "results": results,
               "gpu_generation": "not run"}
    (args.output_dir / "validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
