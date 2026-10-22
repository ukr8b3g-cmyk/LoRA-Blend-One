# Validation — LoRA Blend One

## Current local checks — 2026-10-22

| Check | Result |
| --- | --- |
| Python suite: standard core, Krea2, MOSS, node integration, and failure diagnostics | 42 tests passed |
| JavaScript UI suite: locale, store serialization/reload, legacy recovery, template, inline source toggles, and failure panels | 11 tests passed |
| Node integration after adding the model-status UI payload | Passed; split/legacy MOSS labels and generic label checked |
| Model-status drawing with Advanced loaded in either extension order | JavaScript VM passed |
| Original widget values/order through accordion serialization | JavaScript VM passed |
| Model-status JavaScript syntax | Passed |
| Local MOSS inventory | 10 Standard LoRA files recognized |
| Live Chrome: template, Advanced, English/Japanese switching, save/close/reopen | Passed; 45 saved widget settings and 44 exported API settings matched |
| Real two-input MOSS blend, file reload, and GPU image generation | Passed; details below |
| Live Chrome: two-row source layout and OFF → ON blending | Passed; four-source then five-source saves and recipes checked |

The eight MOSS cases check fused/split QKV equivalence across Weighted/TIES/DARE/DELLA, alpha scaling, exact zero-column removal, combined in-file contributions, ambiguous aliases, unsupported tensors, model-family isolation, text-embedder QKV, and save/rank/metadata round trips.

## Real MOSS adapter validation

Three local inputs with ranks **128 / 32 / 32** were blended at direct coefficients **0.8 / 0.4 / 0.2**, using Weighted / energy / FP32 / CPU.

| Output | Layers | Size | Installed ComfyUI parser |
| --- | --- | --- | --- |
| Rank 16 | 252 | 148,726,088 bytes | All 252 patches loaded |
| Rank 32 | 252 | 297,362,312 bytes | All 252 patches loaded |

Both saved outputs reopened successfully. Every saved tensor was consumed, and patch shapes matched the installed MOSS-Image-SFT BF16 model header. The real ComfyUI LoRA weight adapter applied Q and O patches on CPU and matched the saved factor products.

This used a **header-only model for binding**, not a full base-weight load. It does not establish GPU generation quality.

## Real MOSS SFT end-to-end generation — 2026-10-22

The installed nodes were executed in a running ComfyUI instance through an existing Chrome tab. `LUC-LORA-MOSS-Epoch 12.safetensors` and `LUC-LORA-MOSS-Epoch 20.safetensors` were blended at 50:50 with normalize, Auto → Weighted, CPU computation, Output Rank 16, BF16 storage, and Seed 0 / fixed.

| Check | Result |
| --- | --- |
| Blend job | Success, 3.774 s, no blend warnings |
| Saved file | 72,633,952 bytes, 246 layers, ranks 16–16 |
| Independent file reload | 738/738 tensors finite; all ranks/alpha 16, effective scale 1 |
| Installed ComfyUI parser with MOSS model header | 738/738 tensors consumed; unused tensors, shape mismatches, errors all 0 |
| File SHA256 | `4a764d25ece88578450df946f14fd529878fef56771d62f5044eb773b64218b1` |
| GPU generation | Success, 19.079 s, RTX 5060 Ti / 16 GB VRAM |
| Output image | WEBP, 896 × 1184; decoded and visually inspected, no black/noise-only output |

The saved file was selected in Power Lora Loader (rgthree), enabled at strength 0.3, and applied to `moss_image_sft_bf16.safetensors`. Generation used 8 steps, CFG 1, er_sde/simple, denoise 1, and fixed seed 525173493099460. Execution history confirmed the selected file/strength and successful completion; no nodes were cached for that generation job. The running ComfyUI log contained no LoRA-key loading failure, shape mismatch, or generation exception for the run.

The generation workflow was a copy of the user's existing workflow. It retained lora_latest 0.7, OST_Lora_Moss-8 0.1, tkn_epoch9 0.2, and nanmokaken_moss_base_v1 0.1, replacing its LUC Epoch12 slot with the blend. This was one loading/generation smoke test, not an isolated quality A/B comparison. Timings are observations from this setup, not general performance estimates. Model and LoRA weights are not bundled in this repository.

Live Chrome checks also covered template loading, both Advanced controls, English → Japanese → English switching, and workflow save/close/reopen. Custom Rank 48, Energy Target 0.975, and Storage Dtype BF16 survived reopening. All 45 widget values were saved in the correct positional/named order, and all 44 API settings matched those values and the installed input schemas. Legacy corrupted-file recovery was checked in the JavaScript suite; direct browser import of that old file was not exercised.

## Inline source toggles in live Chrome — 2026-10-22

After restarting ComfyUI, the existing Chrome tab showed each LoRA as two rows: an ON/OFF switch embedded in the filename row, followed by its weight row. A duplicate of the current workflow was used for testing. Its five MOSS inputs were LUC Epoch12, tkn epoch9, OST Moss-10, nanmokaken base v1, and silvana style epoch07, with weights 30 / 25 / 15 / 30 / 20. Both runs used Auto → Weighted, normalize, and Output Rank 32.

| Check | OFF | Back ON |
| --- | --- | --- |
| Slot 5 switch | Disabled | Enabled |
| Slot 5 filename / weight | silvana style epoch07 / 20 retained | Same file / 20 retained |
| Execution and save | Success; 4 sources | Success; 5 sources |
| Saved recipe | silvana excluded | silvana included |
| Saved size | 145,183,088 bytes | 145,183,336 bytes |
| Family / layers / ranks | MOSS-Image / 246 / 32–32 | MOSS-Image / 246 / 32–32 |
| Finite saved tensors / warnings | 738 of 738 / 0 | 738 of 738 / 0 |

The submitted source-node inputs retained the filename and weight while only `enabled_5` changed. The independent saved-file check confirmed that source inclusion matched the switch. The original workflow and its settings were preserved, and its view was restored after testing. This check covers the switches, blending, and saving; image generation was verified separately in the earlier SFT check.

The automated cases also cover disabled unavailable files, legacy defaults, appended switch serialization without shifted file/weight values, switch-state round trips, Advanced visibility, and Krea2's requirement for two enabled slots.

## Failure diagnostics

Python checks verify source attribution for mixed model families, unsupported adapters, incompatible shapes, and missing target layers. The node boundary also identifies unreadable/missing files and reports save or memory failures without blaming a specific LoRA. Full registered paths distinguish identically named files in different folders. Original failures are re-raised; user interruption remains an interruption.

JavaScript VM checks verify the red, wrapped English/Japanese source/reason/fix panel, raw error details, recovery after retry/success, native input-validation failures, and event/listener isolation. Normal source rows retain their layout, and Sources has no empty status strip. Live verification in the existing Chrome tab passed on 2026-10-22 after ComfyUI restart and browser reload. A duplicate of the original workflow combined five MOSS-Image LoRAs with enabled slot 7, `DMD2\dmd2_sdxl_4step_lora.safetensors`, at weight 1.0. The actual `MergeError` stopped node 5 before blend output was saved. The red panel displayed:

- Source: the exact slot 7 and registered filename, with wrapping.
- Reason (Japanese): the selected sources include a LoRA not recognized as MOSS-Image.
- Suggested fix (Japanese): turn the listed LoRA OFF or use LoRAs for the same base model.
- Details: the original incompatible-source message, including `sd_family`.

The first attempt still had an older frontend module loaded and showed only the generic failure toast. Reloading the existing Chrome page and rerunning the same copy displayed the new panel. The original four nodes, widget values, and connections were equal before/after; the saved original workflow SHA256 was unchanged. The test copy was closed and the original tab restored. English live rendering was not repeated; English/Japanese rendering is covered by the VM checks.

## Reproduce

Run with the Python environment used by your ComfyUI installation:

```sh
python -m unittest discover -s tests -p "test_*.py" -v
node --test tests/test_ui.mjs tests/test_source_toggles.mjs tests/test_error_ui.mjs
python tools/validate_moss_local.py --help
```

The real-weight utility takes the ComfyUI root, base-model file, local LoRA files, direct coefficients, and a new output directory. It keeps rank-16/rank-32 files and a validation JSON report. No model files are bundled or downloaded by the utility.

## Not verified

- Visual-quality A/B comparisons, including compressed versus uncompressed blends
- MOSS-Image Base GPU generation and other MOSS LoRAs/settings beyond the SFT run above
- CUDA out-of-memory fallback in a live GPU session
- Per-model generation quality for Krea2, FLUX, Qwen Image, Wan, or SDXL

Unit/VM checks and CPU parser/weight checks are reported separately from live UI and GPU results.
