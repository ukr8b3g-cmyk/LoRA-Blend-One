from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.engine import merge_sources, resolve_weights
from core.formats import scan_lora
from core.recipe import build_metadata, build_recipe, report_to_dict
from core.types import MergeSettings
from core.writer import save_verified


def make_lora(path: Path, *, target="diffusion_model.layers.0.attention.to_q", rank=4, out_dim=24, in_dim=16, seed=1, style="peft", dtype=torch.float32, alpha=None, conv=False):
    g = torch.Generator().manual_seed(seed)
    if conv:
        down = torch.randn(rank, in_dim, 3, 3, generator=g, dtype=dtype)
        up = torch.randn(out_dim, rank, 1, 1, generator=g, dtype=dtype)
    else:
        down = torch.randn(rank, in_dim, generator=g, dtype=dtype)
        up = torch.randn(out_dim, rank, generator=g, dtype=dtype)
    if style == "peft":
        keys = {f"{target}.lora_A.weight": down, f"{target}.lora_B.weight": up}
    elif style == "regular":
        keys = {f"{target}.lora_down.weight": down, f"{target}.lora_up.weight": up}
    elif style == "dot":
        keys = {f"{target}.lora.down.weight": down, f"{target}.lora.up.weight": up}
    else:
        raise ValueError(style)
    if alpha is not None:
        keys[f"{target}.alpha"] = torch.tensor(float(alpha), dtype=torch.float32)
    save_file(keys, str(path), metadata={"test": "1"})
    return up.float(), down.float(), float(alpha / rank if alpha is not None else 1.0)


def output_delta(tensors: dict[str, torch.Tensor], target: str):
    pairs = [
        (f"{target}.lora_B.weight", f"{target}.lora_A.weight"),
        (f"{target}.lora_up.weight", f"{target}.lora_down.weight"),
        (f"{target}.lora.up.weight", f"{target}.lora.down.weight"),
    ]
    for upk, downk in pairs:
        if upk in tensors and downk in tensors:
            up = tensors[upk].float().reshape(tensors[upk].shape[0], tensors[upk].shape[1])
            down = tensors[downk].float().reshape(tensors[downk].shape[0], -1)
            alpha = float(tensors[f"{target}.alpha"].item())
            return (alpha / down.shape[0]) * (up @ down)
    raise KeyError(target)


class LoRABlendOneCoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.target = "diffusion_model.layers.0.attention.to_q"

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_common_formats_and_comfy_alpha_default(self):
        for style in ("peft", "regular", "dot"):
            p = self.dir / f"{style}.safetensors"
            make_lora(p, target=self.target, style=style, alpha=None)
            info = scan_lora(p, compute_hash=True)
            self.assertIn(self.target, info.layers)
            spec = info.layers[self.target]
            self.assertEqual(spec.rank, 4)
            self.assertAlmostEqual(spec.effective_scale, 1.0)
            self.assertEqual(len(info.sha256), 64)
            self.assertFalse(info.unsupported)

    def test_adapter_config_rslora_scale(self):
        p = self.dir / "adapter_model.safetensors"
        make_lora(p, target=self.target, rank=4, alpha=None)
        (self.dir / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8, "use_rslora": True}))
        info = scan_lora(p, compute_hash=False)
        self.assertAlmostEqual(info.layers[self.target].effective_scale, 8 / math.sqrt(4))


    def test_scan_all_current_comfy_standard_suffixes(self):
        target = "transformer.block.0.to_q"
        styles = {
            "underscore": (f"{target}_lora.down.weight", f"{target}_lora.up.weight"),
            "transformers": (f"{target}.lora_linear_layer.down.weight", f"{target}.lora_linear_layer.up.weight"),
            "named_default": (f"{target}.lora_A.default.weight", f"{target}.lora_B.default.weight"),
            "mochi": (f"{target}.lora_A", f"{target}.lora_B"),
        }
        for name, (down_key, up_key) in styles.items():
            p = self.dir / f"suffix_{name}.safetensors"
            save_file({down_key: torch.randn(3, 10), up_key: torch.randn(12, 3)}, str(p))
            info = scan_lora(p, compute_hash=False)
            self.assertIn(target, info.layers, name)
            self.assertEqual(info.layers[target].rank, 3, name)

    def test_local_rng_is_not_advanced_by_sparse_merge(self):
        p1, p2 = self.dir / "rng1.safetensors", self.dir / "rng2.safetensors"
        make_lora(p1, target=self.target, seed=700, alpha=4)
        make_lora(p2, target=self.target, seed=701, alpha=4)
        infos = [scan_lora(p1, compute_hash=False), scan_lora(p2, compute_hash=False)]
        torch.manual_seed(987654)
        before = torch.random.get_rng_state().clone()
        settings = MergeSettings(method="DARE", compression_mode="fixed_rank", output_rank=4, density=0.8, seed=42, output_dtype="fp32", device="cpu")
        merge_sources(infos, [0.5, 0.5], settings)
        after = torch.random.get_rng_state().clone()
        self.assertTrue(torch.equal(before, after))

    def test_weight_normalization(self):
        self.assertEqual(resolve_weights([20, 30, 50], "normalize"), [0.2, 0.3, 0.5])
        self.assertEqual(resolve_weights([50, -25], "direct"), [0.5, -0.25])
        with self.assertRaises(ValueError):
            resolve_weights([50, -25], "normalize")

    def test_weighted_exact_three_way_matches_dense_reference(self):
        paths = [self.dir / f"a{i}.safetensors" for i in range(3)]
        refs = [make_lora(p, target=self.target, rank=4, seed=i + 10, alpha=4) for i, p in enumerate(paths)]
        infos = [scan_lora(p, compute_hash=False) for p in paths]
        weights = [0.2, 0.3, 0.5]
        settings = MergeSettings(method="Weighted", compression_mode="none", output_dtype="fp32", device="cpu")
        tensors, report = merge_sources(infos, weights, settings)
        got = output_delta(tensors, self.target)
        expected = sum(w * (scale * (up @ down)) for w, (up, down, scale) in zip(weights, refs))
        self.assertLess(torch.linalg.norm(got - expected) / torch.linalg.norm(expected), 2e-6)
        self.assertTrue(report.layer_reports[0].exact)
        self.assertEqual(report.layer_reports[0].output_rank, 12)

    def test_weighted_energy_compression(self):
        paths = [self.dir / f"b{i}.safetensors" for i in range(2)]
        refs = [make_lora(p, target=self.target, rank=4, seed=i + 30, alpha=4) for i, p in enumerate(paths)]
        infos = [scan_lora(p, compute_hash=False) for p in paths]
        settings = MergeSettings(method="Weighted", compression_mode="energy", output_rank=None, energy_target=0.95, output_dtype="fp32", device="cpu")
        tensors, report = merge_sources(infos, [0.5, 0.5], settings)
        got = output_delta(tensors, self.target)
        expected = 0.5 * (refs[0][0] @ refs[0][1]) + 0.5 * (refs[1][0] @ refs[1][1])
        err = float(torch.linalg.norm(got - expected) / torch.linalg.norm(expected))
        # 95% Frobenius energy permits up to sqrt(5%) ~= 0.2236 relative error.
        self.assertLessEqual(err, 0.24)
        self.assertGreaterEqual(report.layer_reports[0].captured_energy, 0.94)

    def test_selected_rank_caps_standard_methods_across_compression_modes(self):
        paths = [self.dir / f"cap{i}.safetensors" for i in range(2)]
        for i, path in enumerate(paths):
            make_lora(path, target=self.target, rank=4, seed=450 + i, alpha=4)
        infos = [scan_lora(path, compute_hash=False) for path in paths]
        cases = (
            ("Weighted", "none"),
            ("Weighted", "energy"),
            ("Weighted", "max_input_rank"),
            ("TIES", "energy"),
            ("DARE", "energy"),
            ("DELLA", "energy"),
            ("TIES", "max_input_rank"),
        )
        for method, mode in cases:
            with self.subTest(method=method, compression=mode):
                common = dict(method=method, compression_mode=mode, energy_target=0.9999,
                              auto_rank_cap=8, density=0.85, seed=42, output_dtype="fp32", device="cpu")
                uncapped, base_report = merge_sources(infos, [0.5, 0.5], MergeSettings(**common))
                capped, cap_report = merge_sources(infos, [0.5, 0.5], MergeSettings(**common, output_rank=1))
                self.assertGreater(base_report.layer_reports[0].output_rank, 1)
                self.assertEqual(cap_report.layer_reports[0].output_rank, 1)
                if mode == "energy":
                    self.assertTrue(cap_report.layer_reports[0].warnings)
                base_path = self.dir / f"{method}_{mode}_base.safetensors"
                cap_path = self.dir / f"{method}_{mode}_cap.safetensors"
                save_file(uncapped, str(base_path))
                save_file(capped, str(cap_path))
                self.assertLess(cap_path.stat().st_size, base_path.stat().st_size)

        exact, report = merge_sources(infos, [0.5, 0.5], MergeSettings(
            method="Weighted", compression_mode="none", output_rank=8, output_dtype="fp32", device="cpu"))
        self.assertEqual(report.layer_reports[0].output_rank, 8)
        self.assertTrue(report.layer_reports[0].exact)
        self.assertTrue(torch.isfinite(output_delta(exact, self.target)).all())

    def test_conv_weighted_exact(self):
        target = "lora_unet_conv"
        p1, p2 = self.dir / "c1.safetensors", self.dir / "c2.safetensors"
        r1 = make_lora(p1, target=target, rank=3, out_dim=8, in_dim=4, seed=100, style="regular", alpha=3, conv=True)
        r2 = make_lora(p2, target=target, rank=2, out_dim=8, in_dim=4, seed=101, style="regular", alpha=2, conv=True)
        infos = [scan_lora(p1, compute_hash=False), scan_lora(p2, compute_hash=False)]
        settings = MergeSettings(method="Weighted", compression_mode="none", output_dtype="fp32", device="cpu")
        tensors, _ = merge_sources(infos, [0.4, 0.6], settings)
        got = output_delta(tensors, target)
        expected = 0.4 * (r1[0].reshape(8, 3) @ r1[1].reshape(3, -1)) + 0.6 * (r2[0].reshape(8, 2) @ r2[1].reshape(2, -1))
        self.assertTrue(torch.allclose(got, expected, rtol=2e-5, atol=2e-5))


    def test_gate_conflict_is_diagnostic_only(self):
        target = "diffusion_model.layers.0.attn.to_gate"
        p1, p2 = self.dir / "g1.safetensors", self.dir / "g2.safetensors"
        down = torch.randn(2, 6, generator=torch.Generator().manual_seed(800))
        up = torch.randn(7, 2, generator=torch.Generator().manual_seed(801))
        save_file({f"{target}.lora_A.weight": down, f"{target}.lora_B.weight": up, f"{target}.alpha": torch.tensor(2.0)}, str(p1))
        save_file({f"{target}.lora_A.weight": down.clone(), f"{target}.lora_B.weight": -up.clone(), f"{target}.alpha": torch.tensor(2.0)}, str(p2))
        infos = [scan_lora(p1, compute_hash=False), scan_lora(p2, compute_hash=False)]
        settings = MergeSettings(method="Weighted", compression_mode="none", output_dtype="fp32", device="cpu", gate_conflict_threshold=0.05)
        tensors, report = merge_sources(infos, [0.5, 0.5], settings)
        self.assertEqual(report.gate_targets, 1)
        self.assertEqual(report.gate_conflicts, 1)
        # Opposite equal deltas cancel; diagnostics must not secretly keep one side.
        got = output_delta(tensors, target)
        self.assertLess(float(torch.linalg.norm(got)), 1e-5)

    def test_sparse_methods_are_finite_and_deterministic(self):
        p1, p2 = self.dir / "s1.safetensors", self.dir / "s2.safetensors"
        make_lora(p1, target=self.target, rank=4, out_dim=24, in_dim=16, seed=200, alpha=4)
        make_lora(p2, target=self.target, rank=4, out_dim=24, in_dim=16, seed=201, alpha=4)
        infos = [scan_lora(p1, compute_hash=False), scan_lora(p2, compute_hash=False)]
        for method in ("TIES", "DARE", "DELLA"):
            settings = MergeSettings(
                method=method,
                compression_mode="fixed_rank",
                output_rank=6,
                density=0.85,
                della_half_width=0.05,
                seed=12345,
                output_dtype="fp32",
                device="cpu",
            )
            t1, r1 = merge_sources(infos, [0.5, 0.5], settings)
            t2, r2 = merge_sources(infos, [0.5, 0.5], settings)
            d1 = output_delta(t1, self.target)
            d2 = output_delta(t2, self.target)
            self.assertTrue(torch.isfinite(d1).all(), method)
            self.assertTrue(torch.equal(d1, d2), method)
            self.assertEqual(r1.layer_reports[0].output_rank, 6)
            self.assertEqual(r2.layer_reports[0].output_rank, 6)

    def test_unsupported_dora_is_rejected_by_engine(self):
        p = self.dir / "dora.safetensors"
        make_lora(p, target=self.target, rank=4, alpha=4)
        # Save a separate file because Windows keeps the source memory-mapped.
        with safe_open(str(p), framework="pt", device="cpu") as f:
            data = {k: f.get_tensor(k) for k in f.keys()}
        data[f"{self.target}.dora_scale"] = torch.ones(24)
        dora_path = self.dir / "dora_with_scale.safetensors"
        save_file(data, str(dora_path))
        info = scan_lora(dora_path, compute_hash=False)
        self.assertTrue(info.unsupported)
        with self.assertRaises(ValueError):
            merge_sources([info], [1.0], MergeSettings(method="Weighted", compression_mode="none", device="cpu"))

    def test_save_verify_and_recipe_metadata(self):
        p1, p2 = self.dir / "m1.safetensors", self.dir / "m2.safetensors"
        make_lora(p1, target=self.target, seed=300, alpha=4)
        make_lora(p2, target=self.target, seed=301, alpha=4)
        infos = [scan_lora(p1, compute_hash=True), scan_lora(p2, compute_hash=True)]
        settings = MergeSettings(method="Weighted", compression_mode="fixed_rank", output_rank=4, output_dtype="fp16", device="cpu")
        tensors, report = merge_sources(infos, [0.5, 0.5], settings)
        recipe = build_recipe(infos, [0.5, 0.5], settings, report)
        metadata = build_metadata(recipe)
        out = self.dir / "merged.safetensors"
        save_verified(tensors, out, metadata=metadata, report={"recipe": recipe, **report_to_dict(report)}, write_report=True)
        self.assertTrue(out.exists())
        self.assertTrue(out.with_suffix(".safetensors.report.json").exists())
        with safe_open(str(out), framework="pt", device="cpu") as f:
            meta = f.metadata()
            self.assertEqual(meta["lbo_recipe_version"], "1.0")
            self.assertEqual(meta["lbo_method"], "Weighted")
            self.assertIn("sources", json.loads(meta["lbo_recipe_json"]))
            factor_dtypes = {f.get_slice(k).get_dtype() for k in f.keys() if not k.endswith(".alpha")}
            self.assertEqual(factor_dtypes, {"F16"})

    def test_locale_files_cover_all_declared_inputs(self):
        en = json.loads((ROOT / "locales/en/nodeDefs.json").read_text(encoding="utf-8"))["LoRABlendOne"]
        ja = json.loads((ROOT / "locales/ja/nodeDefs.json").read_text(encoding="utf-8"))["LoRABlendOne"]
        self.assertEqual(set(en["inputs"]), set(ja["inputs"]))
        for lang in (en, ja):
            for value in lang["inputs"].values():
                self.assertTrue(value.get("name"))
                self.assertTrue(value.get("tooltip"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
