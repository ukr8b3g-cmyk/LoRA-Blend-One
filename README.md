[日本語ガイド / Japanese](README.ja.md)

# LoRA Blend One

![LoRA Blend One — blend compatible LoRAs in ComfyUI](docs/images/lora-blend-one-banner.png)

Blend compatible LoRAs into one reusable `.safetensors` file in ComfyUI.

Use two connected nodes to choose up to eight LoRAs, set their strengths, and save a blend with a chosen rank and storage precision. Includes MOSS-Image format normalization and a dedicated Krea2 Gate-Aware method.

[日本語ガイド](README.ja.md) · [Validation](VALIDATION.md) · [Changelog](CHANGELOG.md) · [MIT license](LICENSE)

## Workflow overview

![Power LoRA Loader, LoRA Blend Sources, and LoRA Blend One in ComfyUI](docs/images/workflow-overview.png)

Left: **Power LoRA Loader (rgthree)**. Center: **LoRA Blend Sources**. Right: **LoRA Blend One**. Connect **Sources → Blend One**.

To copy loader settings, open **Sources → Advanced**, select the **Power LoRA Loader**, then click **Import Enabled LoRAs**. The loader itself needs no cable. This screenshot shows the interface layout; its import selector is still `(none)`. [See the illustrated import guide](#import-power-lora-loader-settings).

## Features

- Two-node layout: **LoRA Blend Sources → LoRA Blend One**
- Inline ON/OFF switches for each LoRA, preserving its filename and weight rows
- Weighted, TIES, DARE, and DELLA blending
- MOSS-Image Base/SFT Standard LoRA support, with automatic detection and QKV normalization
- Krea2 Gate-Aware blending with explicit gate-conflict controls
- Different input ranks supported; choose an independent output rank
- FP32 computation with Auto / FP32 / FP16 / BF16 storage
- CPU/CUDA selection and same-settings CPU fallback on CUDA out-of-memory errors
- Expandable Advanced controls, verified temporary saves, recipe metadata, and optional JSON reports
- Error panels with the affected LoRA names, stop reason, and suggested fix

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

Each LoRA keeps two rows: its filename, with an inline **ON/OFF** switch, and its weight. OFF retains the filename and weight while excluding that slot from file validation, loading, and blending. Existing workflows default to ON.

If a blend stops, a red panel on the node shows the affected LoRA slot/filename, the reason, and how to fix it. The panel follows the interface language and wraps long names. Settings/output errors and unidentified causes are reported without blaming an arbitrary LoRA. Retrying clears the previous error; successful execution restores the model-status display.

The original `LoRABlendOne` node remains available as **LoRA Blend One (Legacy)** for existing workflows.

## Import Power LoRA Loader settings

**Supported source: Power Lora Loader (rgthree) only.** The standard LoRA Loader and other loader/stack nodes are not import sources.

This is a **one-click settings copy**. Power LoRA Loader needs **no cables**, and its MODEL/CLIP inputs and outputs can all stay disconnected. Keep it in the **same workflow** as Sources. The only required cable is:

```text
LoRA Blend Sources [sources] ───▶ [LoRA Sources] LoRA Blend One
```

### 1. Open Advanced on Sources

Open **Advanced** on the **LoRA Blend Sources** node. This is where the import controls live.

![1. Open Advanced on LoRA Blend Sources](docs/images/power-loader-01-advanced.png)

### 2. Select the loader

Click the **Power LoRA Loader** row, which initially says **(none)**. Select the loader by its node ID/title, for example **#3 Power Lora Loader (rgthree)**. Your ID may differ. If the panel says **Cannot import / Select a Power LoRA Loader**, this selection is still missing.

![2. Change none to the Power LoRA Loader in this workflow](docs/images/power-loader-02-select.png)

### 3. Import all ON rows, including slots 1 and 2

Click **Import Enabled LoRAs**. It fills **all Sources slots from 1 onward**, including the two rows above Advanced. You do not need to enter those first two LoRAs manually. In this example, `0.80 / 0.20` becomes `80 / 20%`; the other ON rows fill slots 3–8 in displayed order. OFF rows remain in the original loader and are excluded. Unused Sources slots are cleared and turned OFF.

![3. Import button and all eight populated Sources slots](docs/images/power-loader-03-import.png)

Every directly connected Blend One is set to **direct**, preserving the loader's effective strengths. Other merge settings and the original loader are kept. **Import again after changing the loader**; this does not continuously synchronize the nodes.

### 4. Choose rank and output filename

On **LoRA Blend One**, choose **Output Rank** and **Filename Prefix**. The screenshot uses rank 32 as an example; choose your own rank/name. The imported files and strengths are in **Sources**.

![4. Output Rank and Filename Prefix on Blend One](docs/images/power-loader-04-output.png)

### 5. Run to save

Click **Run** to blend and save. Importing settings alone does not create a file. Use the saved blend in your usual generation LoRA loader afterward.

![5. Run starts blending and saving](docs/images/power-loader-05-run.png)

**Limits:** up to eight ON rows. Unavailable/ambiguous files, invalid strengths, and inputs overriding the imported controls block the import before values change. If connected CLIP uses different Model/CLIP strengths, a single blend coefficient cannot reproduce both; use equal strengths or disconnect CLIP before importing.

**Verified locally (2026-10-22):** the existing Chrome tab imported eight ON rows from a ten-row, fully disconnected Power LoRA Loader at `80 / 20 / 20 / 5 / 10 / 20 / 40 / 25%`. All filenames, strengths, and ON states matched the exported workflow and survived reopening. Both original workflow JSON exports were unchanged. English was checked live; Japanese labels/errors were checked in VM tests. This check did not queue blending or image generation.

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

- **direct:** divide each input by 100 and use that strength as entered. `80 / 40 / 20` becomes `0.8 / 0.4 / 0.2`, totaling **1.4**.
- **normalize:** divide each input by the total, retaining the ratio while making the coefficients sum to **1.0**. The same entries become approximately `0.571 / 0.286 / 0.143`.

Entries totaling 100, such as `80 / 20`, give the same result in both modes. The eight-source example above totals 220: normalize would change the first LoRA from `0.8` to about `0.364`. **Power LoRA Loader import selects direct to retain the existing loader strengths.** This weight setting is separate from the merge method.

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

![Merge Method choices in the ComfyUI node](docs/images/merge-methods-menu.png)

**Auto uses Weighted in this implementation.** Start there for the usual MOSS blend; compare the other standard methods when experimenting. None guarantees better image quality.

| Method | Operation |
| --- | --- |
| Auto | Weighted |
| Weighted | Sum weighted model deltas; exact factor concatenation or QR/SVD compression |
| TIES | Keep larger delta elements, then filter opposing signs by default. Advanced: density and sign consensus. |
| DARE | Randomly drop delta elements and rescale survivors. Advanced: density and Seed; sign consensus is off by default. |
| DELLA | Give larger delta elements a higher probability of surviving, then rescale. Sign consensus is on by default. Advanced: density, half width, and Seed. |
| Krea2 Gate-Aware | Gate-conflict handling for exactly two ON Krea2 LoRAs. MOSS inputs cannot use it. |

TIES/DARE/DELLA materialize a dense delta one layer at a time and can need more memory than Weighted. Stochastic operations use local generators derived from Seed, target, and source.

### Processing time

| Method | Work performed before output compression |
| --- | --- |
| Auto / Weighted | Works with low-rank factors; generally the lightest path here. |
| TIES | Reconstructs full-size deltas, selects values by magnitude, checks sign agreement by default, then factors the result. |
| DARE | Reconstructs full-size deltas, generates random masks and rescales, then factors the result. Sign consensus is off by default. |
| DELLA | Also sorts values by magnitude to calculate retention probabilities; sign consensus is on by default. This can be particularly expensive. |

TIES, DARE, and DELLA can all take much longer than Weighted, especially with many large LoRAs or CPU computation. The current sign-consensus path computes each source's sparse delta twice. There is no fixed speed ranking or time estimate across machines/settings. **A smaller output rank does not remove the earlier full-size reconstruction, filtering, or sorting work.** It is mainly an output compression setting. Seed controls random choices; it does not make these methods fast.

### MOSS-Image: automatic detection

Select your MOSS LoRA files and use **Auto / Weighted**, or another standard method. MOSS-Image is detected from metadata or distinctive module shapes; it is not a separate entry in the Merge Method menu.

Before blending, MOSS keys are normalized and joint-block QKV is split into Q/K/V. Source alpha/rank scaling is preserved. Text-embedder QKV stays fused. Exactly zero columns in block-diagonal QKV exports are removed without approximation.

After a successful execution, the node shows **`Last detected model: MOSS-Image`** in a green status row. Before execution it shows `Model detection: run to detect`. These labels follow the selected interface language. The status describes the last successful result. The returned save summary also includes `model=MOSS-Image`.

Output files use ComfyUI-compatible `diffusion_model.<module>.lora_A/B.weight` keys, alpha equal to the saved rank, and MOSS family metadata. Auxiliary time-path files and dedicated sampler behavior from distillation packages are not incorporated into the blended LoRA.

### Krea2 Gate-Aware

Select **Krea2 Gate-Aware** explicitly and turn ON exactly two LoRA slots.

Defaults use gate strength 0.5, `pick_strongest` for opposing gate deltas below cosine −0.05, and rank-32 compression for non-conflicting attention gates. With Output Rank Auto, ordinary matched layers and single-source layers retain exact factors by default.

A numeric Output Rank caps every Krea2 output layer; dedicated gate/normal compression controls can reduce ranks further. These controls appear in Advanced when this method is selected. Auto storage uses FP32 for this method. MOSS inputs cannot use Krea2 Gate-Aware.

## Saving and reproducibility

- Writes a temporary file, reopens it to verify keys/shapes/metadata, then renames it to the final output.
- Saves source strengths, optional source SHA-256 hashes, settings, and a recipe in metadata.
- Can write a per-layer `.safetensors.report.json` report.
- When a sibling PEFT `adapter_config.json` exists, reads alpha/rank patterns and rsLoRA scaling. Otherwise, a missing alpha tensor follows ComfyUI's effective scale of 1.

## Validation and limitations

The local suite passed **42 Python tests** and **19 JavaScript UI tests**, including MOSS format equivalence, existing standard/Krea2 behavior, rank reduction, node integration, locale switching, inline source switches, and workflow serialization/recovery. Structured failure tests also cover source attribution, duplicate filenames in different folders, memory errors, save errors, and cancellation. Error-panel rendering in English/Japanese and event routing were checked in the JavaScript VM. In the existing Chrome tab, a duplicate workflow was run with five MOSS-Image LoRAs and an enabled SDXL LoRA in slot 7. After restarting ComfyUI and reloading the browser, the actual compatibility error displayed the exact slot/filename, Japanese stop reason, and suggested fix on the node. The original workflow settings and connections were preserved. Reload the browser after updating so that the new error UI is loaded.

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

### Inline ON/OFF: live Chrome verification

On **2026-10-22**, after restarting ComfyUI, the switches were exercised in an existing Chrome tab using a copy of the current workflow. Each LoRA kept its two filename/weight rows, with the switch inside the filename row. LoRA 5 retained `silvana-s-style-moss-image-sft-epoch-07.safetensors` and weight **20** throughout:

| LoRA 5 switch | Saved result |
| --- | --- |
| OFF | Four sources; the saved recipe excluded LoRA 5 |
| ON again | Five sources; the saved recipe included LoRA 5 again |

Both blends saved successfully as MOSS-Image LoRAs with **246 layers**, every rank **32**, all **738 tensors finite**, and no blend warnings. The original workflow was preserved and restored in Chrome afterward. This check covered the switches, blending, and saved outputs; image generation is documented in the preceding section.

Run the automated suite with ComfyUI's Python environment:

```sh
python -m unittest discover -s tests -p "test_*.py" -v
```

For local real-weight validation, see `python tools/validate_moss_local.py --help`. Details are in [VALIDATION.md](VALIDATION.md).

## License

[MIT](LICENSE). LoRA files and base models retain their own licenses.
