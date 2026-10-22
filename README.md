[日本語ガイド / Japanese](README.ja.md)

# LoRA Blend One

Blend compatible LoRAs into one reusable `.safetensors` file in ComfyUI.

Use two connected nodes to choose up to eight LoRAs, set their strengths, and save a blend with a chosen rank and storage precision. Includes MOSS-Image format normalization and a dedicated Krea2 Gate-Aware method.

[日本語ガイド](README.ja.md) · [Validation](VALIDATION.md) · [Changelog](CHANGELOG.md) · [MIT license](LICENSE)

## Features

- Two-node layout: **LoRA Blend Sources → LoRA Blend One**
- Weighted, TIES, DARE, and DELLA blending
- MOSS-Image Base/SFT Standard LoRA support, with automatic detection and QKV normalization
- Krea2 Gate-Aware blending with explicit gate-conflict controls
- Different input ranks supported; choose an independent output rank
- FP32 computation with Auto / FP32 / FP16 / BF16 storage
- CPU/CUDA selection and same-settings CPU fallback on CUDA out-of-memory errors
- Expandable Advanced controls, verified temporary saves, recipe metadata, and optional JSON reports

## Install

Inside your ComfyUI `custom_nodes` directory:

```sh
git clone https://github.com/ukr8b3g-cmyk/LoRA-Blend-One.git ComfyUI-LoRA-Blend-One
```

Alternatively, download the repository ZIP and extract its contents into `ComfyUI/custom_nodes/ComfyUI-LoRA-Blend-One/`, with `__init__.py` directly inside that folder.

Restart ComfyUI and refresh its browser page. This package uses `torch` and `safetensors` already supplied by ComfyUI; no additional Python dependency is required.

## Quick start

Open the [LoRA Blend One template](example_workflows/LoRA_Blend_One.json) in ComfyUI, or find it under this package in **Workflow → Browse Templates** after restarting ComfyUI. The template connects Sources → Blend One → Preview Any; select your own LoRAs before running it. It starts at output rank 16 with Seed 0 / fixed and the detailed report off.

1. Add **LoRA Blend Sources** and **LoRA Blend One** from `model/merging/LoRA Blend One`.
2. Connect `sources` to `sources`.
3. Select compatible LoRAs and set their percentages, such as **30 / 70**.
4. Start with **Merge Method = Auto**, **Weight Mode = normalize**, and **Compression = auto**.
5. Choose an output rank or leave **Auto**. Set a filename prefix and queue the workflow.

The default prefix writes a file such as:

```text
ComfyUI/output/loras/LoRA_Blend_One_00001_.safetensors
```

To use the result in generation, copy it to `ComfyUI/models/loras/` or another configured LoRA directory, refresh the file list, and select it in your usual LoRA loader.

Open **Advanced** on the Sources node to select LoRAs 3–8. Seed (default 0), numerical controls, and diagnostics are in Advanced on the merge node. The detailed JSON report defaults to off on this two-node path.

The original `LoRABlendOne` node remains available as **LoRA Blend One (Legacy)** for existing workflows.

## Interface language

English is the default interface. In ComfyUI **Settings**, search for **Language** (`Comfy.Locale`) and select **Japanese** to use Japanese field labels, tooltips, Advanced controls, and model detection messages. Select English to switch back. Method identifiers such as Auto and Krea2 Gate-Aware keep their original names.

## Compatibility

Inputs should target the same base model and compatible layer names and input/output dimensions. Their **LoRA ranks may differ**: rank 16 and rank 32 can be blended, then saved at a separately selected output rank.

| Model family | Handling |
| --- | --- |
| MOSS-Image Base / SFT | Dedicated detection and key/QKV normalization for Standard LoRAs |
| Krea2 | Dedicated, explicitly selected Gate-Aware method; ordinary methods also available |
| FLUX, Qwen Image, Wan, SD-family and other compatible adapters | Generic Standard LoRA factor handling; model-specific key conversion is not provided for these families |

The generic path requires matching target names and shapes. Architecture hints are compatibility checks, not proof of model-specific generation quality. MOSS Base and SFT share a layer layout; this does not guarantee that a LoRA has the same visual effect on both.

Supported factor forms include `.lora_A/B.weight`, `.lora_down/up.weight`, `.lora.down/up.weight`, named PEFT `.lora_A/B.default.weight`, Diffusers `_lora.down/up.weight`, Transformers `.lora_linear_layer.down/up.weight`, and Mochi-style `.lora_A/B`. Linear factors and standard convolutional factors with a 1×1 up kernel are supported.

DoRA, LoHa, LoKr, LoCon-mid, full-diff/set-weight, and other non-Standard-LoRA adapters are outside the supported scope. MOSS inputs additionally reject unconsumed tensors, incomplete pairs, and ambiguous aliases.

## Strengths: normalize or direct

| Weight mode | Example input | Applied coefficients |
| --- | --- | --- |
| `normalize` | 30 / 70 | 0.3 / 0.7 |
| `normalize` | 80 / 40 / 20 | approximately 0.571 / 0.286 / 0.143 |
| `direct` | 80 / 40 / 20 | 0.8 / 0.4 / 0.2 |

Use **direct** to preserve strengths from an existing multi-LoRA loader. Normalize expects non-negative weights with a positive total; direct supports negative coefficients. These numbers scale model deltas and do not guarantee a visual mixture in those proportions.

Krea2 Gate-Aware always uses percentages directly, regardless of Weight Mode.

## Output rank, compression, and file size

A numerical Output Rank caps each saved layer or projection. Lower ranks generally reduce tensor size and may discard detail. Different input ranks do not prevent blending.

| Compression | Behavior with standard methods |
| --- | --- |
| `auto` | Auto output rank uses energy selection; a numeric rank uses fixed-rank compression |
| `energy` | Select by energy, with a numeric Output Rank as a maximum |
| `fixed_rank` | Requires a numeric Output Rank |
| `max_input_rank` | Use the largest contributing input rank, capped by a numeric Output Rank |
| `none` | Weighted only: exact concatenation with Auto, or when the selected rank is at least the concatenated rank; a lower rank requests SVD compression |

Energy Target measures retained numerical delta energy, not image-quality percentage. The optional report records target shortfalls. TIES/DARE/DELLA energy selection also respects `auto_rank_cap`.

Storage precision affects size too. **Auto preserves a common input dtype; mixed input dtypes save as FP32.** For example, combining FP32 and BF16 files can produce a larger file than expected. Select BF16 or FP16 explicitly if you want that storage precision. Computation remains FP32.

## Merge methods

| Method | Operation |
| --- | --- |
| Auto | Weighted |
| Weighted | Sum weighted model deltas; exact factor concatenation or QR/SVD compression |
| TIES | Magnitude sparsification and weighted sign consensus |
| DARE | Random dropping with inverse keep-probability rescaling |
| DELLA | Magnitude-ranked stochastic retention and rescaling |
| Krea2 Gate-Aware | Dedicated two-LoRA Krea2 gate handling |

TIES/DARE/DELLA materialize a dense delta one layer at a time and can need more memory than Weighted. Stochastic operations use local generators derived from Seed, target, and source.

### MOSS-Image: automatic detection

Select your MOSS LoRA files and use **Auto / Weighted**, or another standard method. MOSS-Image is detected from metadata or distinctive module shapes; it is not a separate entry in the Merge Method menu.

Before blending, MOSS keys are normalized and joint-block QKV is split into Q/K/V. Source alpha/rank scaling is preserved. Text-embedder QKV stays fused. Exactly zero columns in block-diagonal QKV exports are removed without approximation.

After a successful execution, the node shows **`Last detected model: MOSS-Image`** in a green status row. Before execution it shows `Model detection: run to detect`. These labels follow the selected interface language. The status describes the last successful result. The returned save summary also includes `model=MOSS-Image`.

Output files use ComfyUI-compatible `diffusion_model.<module>.lora_A/B.weight` keys, alpha equal to the saved rank, and MOSS family metadata. Auxiliary time-path files and dedicated sampler behavior from distillation packages are not incorporated into the blended LoRA.

### Krea2 Gate-Aware

Select **Krea2 Gate-Aware** explicitly and provide exactly two LoRAs.

Defaults use gate strength 0.5, `pick_strongest` for opposing gate deltas below cosine −0.05, and rank-32 compression for non-conflicting attention gates. With Output Rank Auto, ordinary matched layers and single-source layers retain exact factors by default.

A numeric Output Rank caps every Krea2 output layer; dedicated gate/normal compression controls can reduce ranks further. These controls appear in Advanced when this method is selected. Auto storage uses FP32 for this method. MOSS inputs cannot use Krea2 Gate-Aware.

## Saving and reproducibility

- Writes a temporary file, reopens it to verify keys/shapes/metadata, then renames it to the final output.
- Saves source strengths, optional source SHA-256 hashes, settings, and a recipe in metadata.
- Can write a per-layer `.safetensors.report.json` report.
- When a sibling PEFT `adapter_config.json` exists, reads alpha/rank patterns and rsLoRA scaling. Otherwise, a missing alpha tensor follows ComfyUI's effective scale of 1.

## Validation and limitations

The local suite passed **29 Python tests** and **3 JavaScript UI tests**, including MOSS format equivalence, existing standard/Krea2 behavior, rank reduction, node integration, locale switching, and workflow serialization/recovery.

Three real MOSS LoRAs with ranks **128 / 32 / 32** were blended on CPU at direct coefficients **0.8 / 0.4 / 0.2**. Both saved outputs reopened successfully; the installed ComfyUI parser loaded all **252 layers**, and its real weight adapter applied Q/O patches.

| FP32 output | Saved size |
| --- | --- |
| Rank 16 | 148,726,088 bytes (about 149 MB) |
| Rank 32 | 297,362,312 bytes (about 297 MB) |

These three-input CPU checks used the base model header for shape binding.

### MOSS-Image SFT: blend, save, reload, and GPU generation

An end-to-end check passed on **2026-10-22** in a running ComfyUI instance, using an existing Chrome tab:

| Stage | Verified result |
| --- | --- |
| Inputs | `LUC-LORA-MOSS-Epoch 12.safetensors` + `LUC-LORA-MOSS-Epoch 20.safetensors`, 50:50, `normalize` |
| Blend | Auto → Weighted, CPU computation, Output Rank 16, BF16 storage, Seed 0 / fixed |
| Saved file | **72,633,952 bytes** (about 72.6 MB), 246 layers, every saved rank 16, no blend warnings |
| Reload | All **738 tensors** finite and consumed by the installed ComfyUI parser; no unused tensors or shape mismatches |
| Generation | Saved blend selected in **Power Lora Loader (rgthree)** at strength **0.3**, with `moss_image_sft_bf16.safetensors` |
| Image | **896 × 1184**, 8 steps, CFG 1, `er_sde` / `simple`, fixed seed 525173493099460; WEBP saved and visually inspected |

The blend and generation jobs completed successfully in 3.774 s and 19.079 s respectively on the tested setup (RTX 5060 Ti, 16 GB VRAM). These are individual observed times, not general performance estimates. The actual execution history confirmed that the saved blend was enabled at strength 0.3, and the image decoded at the expected size without a black/noise-only result.

Other LoRAs from the original generation workflow remained enabled. This confirms loading and generation under those conditions; it is not an isolated visual-quality A/B comparison or a guarantee for every MOSS LoRA, MOSS-Image Base, or other model family.

Live Chrome checks also passed for template loading, Advanced expansion/collapse, English → Japanese → English switching, and workflow save/close/reopen. Custom Rank 48, Energy Target 0.975, and Storage Dtype BF16 remained intact; all 44 exported API settings matched the saved values and declared input types.

Run the automated suite with ComfyUI's Python environment:

```sh
python -m unittest discover -s tests -p "test_*.py" -v
```

For local real-weight validation, see `python tools/validate_moss_local.py --help`. Details are in [VALIDATION.md](VALIDATION.md).

## License

[MIT](LICENSE). LoRA files and base models retain their own licenses.
