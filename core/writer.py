from __future__ import annotations

import json
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def save_verified(
    tensors,
    output_path: str | Path,
    *,
    metadata: dict[str, str],
    report: dict | None = None,
    write_report: bool = True,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".partial")
    try:
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(temp_path), metadata=metadata)
        with safe_open(str(temp_path), framework="pt", device="cpu") as f:
            saved_keys = set(f.keys())
            expected_keys = set(tensors)
            if saved_keys != expected_keys:
                raise RuntimeError("Saved safetensors key set does not match the merge result")
            saved_meta = f.metadata() or {}
            if saved_meta.get("lbo_recipe_version") != metadata.get("lbo_recipe_version"):
                raise RuntimeError("Saved safetensors metadata verification failed")
            for key, tensor in tensors.items():
                if tuple(f.get_slice(key).get_shape()) != tuple(tensor.shape):
                    raise RuntimeError(f"Saved tensor shape mismatch for {key}")
        os.replace(temp_path, output_path)
        if write_report and report is not None:
            sidecar = output_path.with_suffix(output_path.suffix + ".report.json")
            with sidecar.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        return output_path
    except Exception:
        try:
            if temp_path.exists():
                temp_path.unlink()
        finally:
            raise
