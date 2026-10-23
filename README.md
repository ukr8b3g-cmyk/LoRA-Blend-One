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

## Inspect and restore a saved blend

Open **Advanced** on **LoRA Blend One**, select **Saved Blend LoRA**, and use **Read / Refresh Recipe**. The saved file must be in a configured LoRA folder. The node shows the source names, effective coefficients, SHA256 match status, method, rank, storage type, and global strength.

**Restore Sources + Blend One** restores the connected **LoRA Blend Sources** and the merge settings. It uses `direct` percentages (`coefficient × 100`), enables the recorded sources, clears/disables unused slots, and fixes the recorded seed. Filename Prefix is kept. The original input percentages and normalize/direct choice are not recorded by older recipes; their effective coefficients are restored instead.

Original source files are required. Missing files, hash mismatches, ambiguous names, or settings outside the node's supported ranges block restoration before either node changes. Recipes without hashes show an explicit name-only warning. Files without a supported LoRA Blend One recipe cannot be restored. This works with the same recipe format across model families; it does not extract source LoRAs from compressed weights. Restart ComfyUI and reload the browser after installing the update.

### Drag and drop

Drop a `.safetensors` blend made by LoRA Blend One onto the ComfyUI canvas. Its recipe opens in a **new workflow tab** with connected Sources and Blend One nodes, original source names/coefficients, merge settings, and a fixed seed. Existing tabs remain open. Each drop gets a fresh workflow name/ID, including repeated drops of the same file. No blend is queued automatically.

The dropped blend can be outside the model folders: only its header is read in the browser, and only the recipe is sent to the local ComfyUI API. Source LoRAs still need to be installed. Missing, ambiguous, or hash-mismatched sources produce an inspection tab with **all slots OFF**; resolve the files and use **Read / Refresh Recipe → Restore Sources + Blend One**. The imported recipe is retained when saving the workflow. Ordinary images, workflows, and LoRAs without a Blend One recipe keep their existing file-handler behavior.

## Extract a model difference (MOSS-Image)

Two additional nodes are available under **model/merging/LoRA Blend One**:

Open the [model extraction template](example_workflows/LoRA_Blend_Model_Extract.json), also available under this package in **Workflow → Browse Templates** after restarting ComfyUI. It connects **Inspect → Extract** and starts with **Full Scan**, **Fixed Rank 32**, BF16, Seed 0 / fixed, and **Review Only**. Replace the model-selection placeholders with your original and modified MOSS-Image models in both nodes, then Run to analyze without saving. Base Model is the original reference model, which can itself be an SFT model. To copy the completed analysis recommendation once, use **Apply Recommendations**; manual settings remain editable. Select **Extract and Save**, then Run, when you want to save the difference. Changing that button alone does not start extraction.

- **LoRA Blend Model Inspect:** compare a base model and its modified model. Header inspection checks supported keys/shapes; **Full Scan** also reads values and checks changed parameters.
- **LoRA Blend Model Extract:** save `modified - base` using **Fixed Rank**, **Auto Rank**, or **Full Delta**. Select the two registered model files, mode, storage precision, and output prefix. Rank limits, SVD method, device, and non-LoRA policy are in **Advanced**.

Extraction currently supports **single, non-quantized native MOSS-Image Safetensors files only**. Other model families, sharded files, and quantized inputs are not enabled. This is a separate path from LoRA blending; the generic blend compatibility list below does not apply to model extraction.

The default non-LoRA policy **Stop** rejects changed parameters that cannot be represented by the supported A/B factors. **Store Diff** preserves them as ComfyUI `.diff` tensors, producing a **hybrid_delta**; **Exclude With Report** produces an explicitly partial result. Hybrid/Full Delta files require ComfyUI's extended loader and **cannot be used as current Blend One Sources**. This feature does not recover the original training LoRA or guarantee a small output. Files are written inside `ComfyUI/output`; use the saved file in your configured LoRA folder to load it normally.

### Rank controls, recommendations, and saving

New Extract nodes default to **Fixed Rank 32**. **Rank Preset** offers 16 / 32 / 64 / 128 / 256; the Rank field still accepts any integer from 1 to 4096, including 17 or 24, subject to each matrix's dimensions. Auto Rank remains optional, with a default Maximum Rank of 64 that you can raise manually. Fixed Rank does not target 99% energy. Existing workflows retain their saved values.

Inspect displays a readable analysis and recommendation and provides an **extraction_plan** output. Connecting it leaves Extract's settings editable. Connecting, rerunning Inspect, or reopening a workflow never copies recommendations automatically: **Apply Recommendations** copies a completed Full Scan recommendation once, including the model pair, while preserving the output filename. You can then change rank, storage precision, and other settings manually. Required rank and output size are not measured by Full Scan; 32 is a practical starting value, not an accuracy guarantee. Different model pairs or files changed since inspection require a new analysis before connected extraction.

The side-by-side **Review Only** and **Extract and Save** buttons show the selected action with a check mark and amber/green highlighting. The caption explains what the next **Run** does. Selecting a button does not queue work. Review Only displays current manual settings separately from recommendations and saves nothing. Without Inspect connected, Extract runs directly with its manual settings. Extract/Inspect labels stay English; hover help and result explanations follow the English/Japanese locale.

For outputs with up to **256 MiB of saved tensor data**, extraction now buffers tensors in CPU memory and writes once. Larger outputs switch to disk spooling. This limit covers buffered tensor data, not total RAM use. Finite-value checks, full saved-tensor verification, cancellation handling, and non-overwriting publication remain enabled. This changes saving, not SVD or weight-error measurement.

### Current local verification — 2026-10-23–24

**88 Python tests and 37 JavaScript tests passed**, including manual connected settings, English/Japanese help, button selection and workflow serialization, bounded-memory/disk saving, corrupt-output detection, and existing MOSS/Krea2 blending regressions. JavaScript uses a test Graph/DOM, not a live ComfyUI screen.

The corrected comparison **original MOSS SFT BF16 → KR2 SFT BF16** (not the merged Moss_Luc model) passed CPU extraction with Fixed Rank 32, BF16, Seed 0, all targets, Stop, input hashing, and four CPU threads:

| Check | Result |
| --- | --- |
| Full Scan | 462 matched entries; 168 changed matrices; no changed nonmatrix parameters; 24.26 s |
| Connected Review Only | Manual rank 17 / FP32 retained separately from recommended rank 32; no files saved |
| Saved standard LoRA | 168 rank-32 targets; 101,914,936 bytes / 101.91 MB / 97.19 MiB |
| Weight reconstruction | 96.5484% relative error; 6.7841% retained difference energy; not an image-quality score |
| Native ComfyUI loader | 168 targets recognized, zero unused keys; actual Q-slice weight application matched |
| Extraction through saving | 299.99 s on CPU; no GPU/end-to-end speed comparison |
| Save-only comparison, identical tensors | 3.21695 s → 0.34744 s; 168 part files → 0; all tensor values matched |

The save-only timings are one old-then-new run on the same disk and include cache effects; they do not establish an overall extraction speedup or a comparison with KJ. Rank 32 favors size and does not faithfully reproduce this model difference. The extraction template passed JSON/schema/link/default checks and a small CPU Inspect→Extract Review Only run with no saved files. The running ComfyUI template-list API and template download both exposed it. **Live checks of the new extraction buttons/template in Chrome, GPU image generation, and image-quality comparisons remain unperformed.** Restart ComfyUI and reload the browser after updating. See [VALIDATION.md](VALIDATION.md) for details.

### Earlier merged-model extraction results — 2026-10-23

The same local pair, `moss_image_sft_bf16.safetensors` → `Moss_Luc-sft-bf16.safetensors`, was tested on CPU with BF16 storage, seed 0, all parameters, and **Store Diff**. Input SHA256 values matched across runs.

| Extraction | Relative weight error | Measured retained energy | Output size (decimal) |
| --- | ---: | ---: | ---: |
| Fixed Rank 16; automatic SVD | 95.0658% | 9.6249% | 80,053,060 bytes / 80.05 MB |
| Auto Rank, 99% target, cap 256; automatic SVD | 83.8158% | 29.7491% | 1,268,436,340 bytes / 1.2684 GB |
| Auto Rank, 99% target, cap 3072; **exact SVD** | **9.9223%** | **99.0155%** | **11,108,963,604 bytes / 11.1090 GB (10.3460 GiB)** |

For the exact run, the **269 matrix targets used ranks 125–2898, median 1823**, with no target reaching cap 3072. All reached 99% before storage conversion. **After BF16 rounding, 49 matrix targets were slightly below 99%, with a minimum of 98.99945%.** Thus, the saved file meets 99% **overall**, but does not guarantee at least 99% in every individual layer. The maximum selected rank is specific to exact SVD and this pair; it is not a general rank recommendation. The cap-256 run used approximate SVD on large matrices, so this comparison changes both the cap and decomposition method.

Retained energy is `1 - relative_Frobenius_error²`; it describes reconstruction of the **model weight difference**, not image similarity or quality. A 99% energy target corresponds to roughly 10% relative norm error. The output exceeds the approximately 10.16 GB source model, so high retention did **not** provide a size reduction for this pair. The exact CPU run took about 35.5 minutes including save verification; this is one observed run, not a performance estimate.

All three outputs were saved and reopened. The installed native ComfyUI MOSS loader recognized **344 changed targets** (269 matrices and 75 direct differences), with **zero unused keys**; actual Q-slice, O-projection, and norm1 weight application passed. A separate, explicitly partial one-layer standard-LoRA extraction also passed Blend One blend/save/reload; this does not make the complete hybrid output a supported blend source. No live extraction-node browser check, GPU generation, or image-quality comparison was performed. See [VALIDATION.md](VALIDATION.md) and the [implementation design and scope](docs/MODEL_DELTA_EXTRACT_DESIGN.ja.md).

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

DoRA, LoHa, LoKr, LoCon-mid, full-diff/set-weight, and other non-Standard-LoRA adapters are outside the supported **blend-input** scope. MOSS inputs additionally reject unconsumed tensors, incomplete pairs, and ambiguous aliases.

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

### Input and save safety (2026-10-23)

- **NaN / Inf:** input factors, alpha/effective blend coefficients, and tensors after storage conversion are checked. Invalid output stops before the final file is replaced, with source/layer details and English/Japanese recovery hints. A finite FP32 factor can overflow during FP16 conversion even when its matrix product is small. Choose FP32/BF16 or check strengths when this happens; storage precision is never changed automatically.
- **Unsupported tensors:** generic adapters account for all tensor keys. Extra bias, OFT, normalization, or incomplete factor tensors block blending with filename/key details instead of being silently discarded. MOSS retains its existing strict checks; known unsupported adapters retain their rejection path.
- **Optional report failure:** a successfully saved LoRA remains a success even if `.report.json` fails. The node returns the main-file location and warning details. Reports use a temporary file, so a failed write does not truncate an existing report. Report OFF remains unchanged.

All three issues were reproduced before fixing them. The isolated publication snapshot passed **56 Python tests** (including **14 new safety regression tests**, MOSS **8**, and Krea2 **6**) and **19 JavaScript tests**. Checks include finite FP16/BF16/FP32 save/reload, rejected non-finite input/storage overflow, unsupported keys, and injected report I/O failure with successful node output. CPU tests use temporary files and stubbed ComfyUI services; JavaScript uses test Graph/DOM. This update was not rechecked in a live browser, real-LoRA GPU generation, CUDA OOM recovery, or image-quality comparisons. Earlier live results below are separate checks. Restart ComfyUI to load the updated Python modules.

The local suite passed **65 Python tests** and **34 JavaScript UI tests**, including MOSS format equivalence, existing standard/Krea2 behavior, rank reduction, node integration, locale switching, inline source switches, and workflow serialization/recovery. Structured failure tests also cover source attribution, duplicate filenames in different folders, memory errors, save errors, and cancellation. Error-panel rendering in English/Japanese and event routing were checked in the JavaScript VM. In the existing Chrome tab, a duplicate workflow was run with five MOSS-Image LoRAs and an enabled SDXL LoRA in slot 7. After restarting ComfyUI and reloading the browser, the actual compatibility error displayed the exact slot/filename, Japanese stop reason, and suggested fix on the node. The original workflow settings and connections were preserved. Reload the browser after updating so that the new error UI is loaded.

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
