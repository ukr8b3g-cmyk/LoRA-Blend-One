# Changelog

## Unreleased

- Default model extraction to Fixed Rank 32, add rank presets, and keep connected Inspect recommendations advisory until explicitly applied.
- Show Review Only / Extract and Save as highlighted buttons and include a connected Inspect→Extract distribution template with Review Only defaults.
- Buffer up to 256 MiB of extracted tensor data for one verified save, retaining disk spooling for larger outputs; document CPU/native-loader validation and the save-only timing comparison.

- Preserve schema widget order in LiteGraph store serialization and repair identifiable early exports with shifted settings.
- Add the supplied three-node workflow template with validated settings and Seed 0 / fixed.
- Follow ComfyUI's locale for custom Advanced and model detection UI; default to English.
- Split source selection and merging into two connected nodes; preserve the legacy node.
- Add an Advanced accordion, Seed default 0, and report-off default on the two-node path.
- Make numeric Output Rank constrain all standard compression modes.
- Add explicit Krea2 Gate-Aware merging and rank caps.
- Add MOSS-Image detection, module-name and QKV normalization, and output metadata.
- Show the last detected model on the merge node after execution.
- Verify real MOSS LoRA blend/save/reload with the installed ComfyUI parser on CPU.
- Add English and Japanese usage guides and public validation notes.

## 0.1.0 — 2026-09-29

Initial implementation.

- One-node 1–8 Standard LoRA merge
- Basic percentages, Auto rank, filename
- Weighted / TIES / DARE / DELLA
- exact Weighted concat and two-sided QR + small-core SVD compression
- deterministic randomized SVD for nonlinear methods
- FP32 compute with Auto/FP32/FP16/BF16 storage
- per-layer CPU/CUDA planning and same-settings CPU OOM fallback
- conservative format validation and unsupported-adapter rejection
- gate conflict diagnostics without implicit gate modification
- verified safetensors save and recipe metadata
- optional detailed JSON report
- English/Japanese native ComfyUI i18n labels/descriptions/tooltips
