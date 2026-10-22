# Validation — LoRA Blend One

## Current local checks — 2026-10-22

| Check | Result |
| --- | --- |
| Python suite: standard core, Krea2, MOSS, and node integration | 29 tests passed |
| Node integration after adding the model-status UI payload | Passed; split/legacy MOSS labels and generic label checked |
| Model-status drawing with Advanced loaded in either extension order | JavaScript VM passed |
| Original widget values/order through accordion serialization | JavaScript VM passed |
| Model-status JavaScript syntax | Passed |
| Local MOSS inventory | 10 Standard LoRA files recognized |

The eight MOSS cases check fused/split QKV equivalence across Weighted/TIES/DARE/DELLA, alpha scaling, exact zero-column removal, combined in-file contributions, ambiguous aliases, unsupported tensors, model-family isolation, text-embedder QKV, and save/rank/metadata round trips.

## Real MOSS adapter validation

Three local inputs with ranks **128 / 32 / 32** were blended at direct coefficients **0.8 / 0.4 / 0.2**, using Weighted / energy / FP32 / CPU.

| Output | Layers | Size | Installed ComfyUI parser |
| --- | --- | --- | --- |
| Rank 16 | 252 | 148,726,088 bytes | All 252 patches loaded |
| Rank 32 | 252 | 297,362,312 bytes | All 252 patches loaded |

Both saved outputs reopened successfully. Every saved tensor was consumed, and patch shapes matched the installed MOSS-Image-SFT BF16 model header. The real ComfyUI LoRA weight adapter applied Q and O patches on CPU and matched the saved factor products.

This used a **header-only model for binding**, not a full base-weight load. It does not establish GPU generation quality.

## Reproduce

Run with the Python environment used by your ComfyUI installation:

```sh
python -m unittest discover -s tests -p "test_*.py" -v
python tools/validate_moss_local.py --help
```

The real-weight utility takes the ComfyUI root, base-model file, local LoRA files, direct coefficients, and a new output directory. It keeps rank-16/rank-32 files and a validation JSON report. No model files are bundled or downloaded by the utility.

## Not verified

- GPU image generation or visual-quality A/B comparisons
- CUDA out-of-memory fallback in a live GPU session
- Final live frontend rendering, reload, or language-switch interaction
- Per-model generation quality for Krea2, FLUX, Qwen Image, Wan, or SDXL

Unit/VM checks and CPU parser/weight checks are reported separately from live UI and GPU results.
