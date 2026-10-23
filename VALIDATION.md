# Validation — LoRA Blend One

## Current local checks — 2026-10-23

| Check | Result |
| --- | --- |
| Python suite: standard core, Krea2, MOSS, node integration, failure diagnostics, recipe restoration/API, and model extraction | 75 tests passed |
| JavaScript UI suite: locale, store serialization/reload, legacy recovery, template, source toggles, failure panels, recipe controls, Power LoRA Loader import, and extractor UI | 35 tests passed |
| Node integration after adding the model-status UI payload | Passed; split/legacy MOSS labels and generic label checked |
| Model-status drawing with Advanced loaded in either extension order | JavaScript VM passed |
| Original widget values/order through accordion serialization | JavaScript VM passed |
| Model-status JavaScript syntax | Passed |
| Local MOSS inventory | 10 Standard LoRA files recognized |
| Live Chrome: template, Advanced, English/Japanese switching, save/close/reopen | Passed; 45 saved widget settings and 44 exported API settings matched |
| Real two-input MOSS blend, file reload, and GPU image generation | Passed; details below |
| Live Chrome: two-row source layout and OFF → ON blending | Passed; four-source then five-source saves and recipes checked |
| Live Chrome: imported recipe inspection, restoring changed values, reblend, save, and file reload | Passed; all 738 saved tensors equal to the original blend |
| Live Chrome: import eight ON rows from Power LoRA Loader and export/reopen the workflow | Passed; all names, strengths, ON states, and selected loader retained |

The eight MOSS cases check fused/split QKV equivalence across Weighted/TIES/DARE/DELLA, alpha scaling, exact zero-column removal, combined in-file contributions, ambiguous aliases, unsupported tensors, model-family isolation, text-embedder QKV, and save/rank/metadata round trips.

## Model-difference extraction — 2026-10-23

The complete local suite after adding the MOSS-only extraction path passed **75 Python tests** (65 existing plus 10 extraction tests) and **35 JavaScript tests** (34 existing plus one extraction UI VM case). The extraction cases cover MOSS layout matching, exact/automatic rank, Full Delta, nonmatrix policies, explicit partial output, BF16 save/reload, finite/quantized-input rejection, cancellation cleanup, no-overwrite saves, and report-failure handling. These are offline tests, not a live browser/UI check.

Real-file CPU extraction used the same `moss_image_sft_bf16.safetensors` and `Moss_Luc-sft-bf16.safetensors` inputs across all three runs, confirmed by SHA256. Each used BF16, seed 0, all targets, and Store Diff. All 462 logical parameter keys matched after QKV normalization; `rope.bands` was excluded as a known reconstructed buffer. 344 targets changed; 118 unchanged targets were omitted.

| Mode / cap / decomposition | Relative Frobenius error | Overall retained energy after BF16 | Saved bytes |
| --- | ---: | ---: | ---: |
| Fixed 16 / automatic | 0.950658008228 | 9.624935139% | 80,053,060 |
| Auto 99% / 256 / automatic | 0.838158000879 | 29.749116556% | 1,268,436,340 |
| Auto 99% / 3072 / exact | 0.0992230429695 | 99.015478774% | 11,108,963,604 |

At cap 256, 266 of 269 matrices missed the target. An independent exact singular-value check on `text_embedder.proj.weight` (3072 × 4096) found optimal rank-256 retention of 30.6558%, compared with 27.9980% from randomized SVD; exact 99% required rank 2562 for that example layer. The cap therefore materially constrained that layer even with exact decomposition.

The cap-3072 run used exact SVD throughout. The 269 selected matrix ranks ranged from 125 to 2898, median 1823, with no cap hits. All matrices met 99% before conversion. After BF16 rounding, **49 individual matrices were just below 99%**, minimum **98.999447865%** (maximum individual relative error 10.002760292%). The saved file meets the **global** 99% target, not a strict per-layer post-storage guarantee. The minimum rank claim applies to the pre-storage exact-SVD criterion only. High-rank output was 11.1090 decimal GB / 10.3460 GiB, larger than the approximately 10.16 GB input model. Total wall time including save verification was 2132.375 seconds on CPU with four threads; this is one measured run.

Each complete output is **hybrid_delta**, preserving 75 nonmatrix targets with native `.diff`. Official Safetensors-reader header/shape/dtype/finite checks and payload-copy verification passed. The installed native ComfyUI MOSS loader loaded all **344 patches with zero unused keys** for each file; Q-slice, O-projection, and norm1 direct-difference weight application passed against actual base-model weights. Current Blend One Sources intentionally rejects mixed `.diff` files. A separate explicit one-layer partial extraction of `layers.0.proj.o.weight` saved a standard LoRA and passed existing Blend scan/merge/save/reload; 343 changed targets were reported excluded, so that check does not establish a complete equivalent model.

Retained energy is `1 - relative_error²` in weight-difference space, not image quality. Product code and installed runtime were unchanged during the cap-256/cap-3072 measurements. No live extraction-node UI, GPU generation, CUDA OOM recovery, or visual-quality comparison was exercised. Other model families remain disabled for extraction. Private model weights and generated multi-GB output files are not bundled in this repository.

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

## Saved-recipe restoration

The recipe reader checks registered file names and source SHA256 values without reading tensor data. Python cases cover old recipes, custom ranks, direct coefficients including negative weights, same-basename matching, missing/ambiguous/mismatching files, name-only warnings, unsupported settings, Krea2 controls, and local HTTP routes. JavaScript VM cases cover English/Japanese summaries, Advanced visibility, schema-order serialization, restoring the connected Sources, unused-slot clearing, fixed seeds, unchanged output names, rechecking files before applying, linked inputs, and stale requests.

Offline CPU verification used `MOSS_LUC12_LUC20_rank16__.safetensors`. Both recorded LUC Epoch 12/20 sources matched their SHA256 values. Restoring coefficients 0.5/0.5, rank 16, BF16, and the recorded settings through the real split node produced a saved file that reopened successfully. All 738 tensors had identical shapes, dtypes, and values to the supplied blend (maximum absolute difference 0). New recipes also recorded the registered source names. This result is specific to these files/settings and environment.

Drag/drop VM checks verify bounded header reads, metadata-only requests, connected nodes in a new tab, coefficients/settings/fixed seed, repeated-drop unique IDs, original graph preservation, all-OFF handling for unresolved sources, ordinary-file delegation, unavailable API, missing node cleanup, and malformed header/recipe handling. Imported recipes survive serialization and can be rechecked from outside the model folders. The metadata endpoint checks the same registered-source allowlist and SHA256 matching as installed-file inspection.

The installed frontend's `app.handleFile` and `app.loadGraphData` paths were inspected locally: normal canvas file drops invoke `handleFile`; `loadGraphData` preserves the outgoing workflow draft and creates a new temporary workflow for the fresh name. Tests simulate that tab API; they do not constitute a live browser drop test.

### Live recipe controls and reproduction — 2026-10-22

The existing Chrome tab already contained a recipe-imported workflow for `MOSS_LUC12_LUC20_rank16_verify_20261022.safetensors`. Its Advanced panel displayed both original LoRA names, coefficients 0.5/0.5, matching source hashes, Weighted, rank 16, BF16, and fixed seed 0. A duplicate workflow was used to change Source 1 from 50 to 20, turn Source 2 OFF, and change rank 16 to 32. Clicking **Restore Sources + Blend One** restored 50/50, both sources ON, and rank 16, while retaining the test output prefix.

Running that copy in ComfyUI saved a 72,634,136-byte blend successfully: MOSS-Image, Weighted, two sources, 246 layers, ranks 16–16, and no warnings. Independent file reload found all 738 tensors identical in shape, dtype, and value to the supplied blend, with maximum absolute difference 0. The new recipe included both registered source names. The file size differs because its metadata changed.

The original workflow's complete exported JSON was identical before and after validation, including nodes, values, properties, and links. The original tab was restored; the validation copy was retained after an additional LoRA Loader appeared in it. ComfyUI was not restarted. This live check covers the imported recipe controls and CPU reblend/save/reload. A fresh physical file drag/drop was not replayed, and image generation was not repeated for this recipe check.

## Power LoRA Loader import — 2026-10-22

Eight JavaScript VM cases cover Advanced-only English/Japanese controls, settings-only import without changing the original loader, displayed row order, direct percentages, unused-slot clearing, zero/negative strengths, eight ON rows among ten total rows, missing/ambiguous filenames, invalid strengths, separate Model/CLIP limits, connected-input overrides, multiple Blend One consumers, stale/other-workflow selection, and serialization/reload without shifting saved values. Existing recipe/drop/toggle/error UI cases also pass. The Python blending implementation and rgthree files are unchanged.

Live verification used the existing Chrome tab and a duplicate of the user's loader workflow. The new controls appeared inside Sources Advanced after a browser reload. A ten-row Power LoRA Loader had eight ON rows and two OFF rows. Clicking **Import Enabled LoRAs** copied the eight enabled filenames in displayed order at **80 / 20 / 20 / 5 / 10 / 20 / 40 / 25 percent**, excluded the two OFF rows, and changed the connected Blend One from normalize to direct. No model inputs were connected or loaded.

The exported workflow contained exactly the existing 24 Sources values, with no import controls inserted into the schema order. All eight names, percentages, and ON states matched the loader; its widget values and properties were unchanged. Other merge settings, including the output prefix, were kept. Switching away and reopening the test tab retained the imported values and selected loader. Complete before/after JSON exports of both original workflow tabs were identical. The test copy was retained for inspection. Japanese labels/errors were covered by VM tests; the live import was checked in English. No blend execution or image generation was queued for this settings-import check, and generation equivalence after compression is not established.

## Reproduce

Run with the Python environment used by your ComfyUI installation:

```sh
python -m unittest discover -s tests -p "test_*.py" -v
node --test tests/test_ui.mjs tests/test_source_toggles.mjs tests/test_error_ui.mjs tests/test_recipe_ui.mjs tests/test_recipe_drop.mjs tests/test_power_loader.mjs tests/test_extractor_ui.mjs
python tools/validate_moss_local.py --help
```

The real-weight utility takes the ComfyUI root, base-model file, local LoRA files, direct coefficients, and a new output directory. It keeps rank-16/rank-32 files and a validation JSON report. No model files are bundled or downloaded by the utility.

## Not verified

- Visual-quality A/B comparisons, including compressed versus uncompressed blends
- MOSS-Image Base GPU generation and other MOSS LoRAs/settings beyond the SFT run above
- CUDA out-of-memory fallback in a live GPU session
- Per-model generation quality for Krea2, FLUX, Qwen Image, Wan, or SDXL

Unit/VM checks and CPU parser/weight checks are reported separately from live UI and GPU results.
