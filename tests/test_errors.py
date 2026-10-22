from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.engine import merge_sources
from core.errors import MergeError
from core.formats import scan_lora
from core.types import MergeSettings


class MergeErrorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.target = "diffusion_model.layers.0.attention.to_q"

    def tearDown(self):
        self.tmp.cleanup()

    def source(self, name, *, target=None, out_dim=6, offset=0., conv_tail=None, metadata=None, unsupported=False):
        target = target or self.target
        down_shape = (2, *(conv_tail or (4,)))
        up_shape = (out_dim, 2, 1, 1) if conv_tail else (out_dim, 2)
        tensors = {
            target + ".lora_A.weight": torch.arange(math.prod(down_shape), dtype=torch.float32).reshape(down_shape) / 10 + offset,
            target + ".lora_B.weight": torch.arange(math.prod(up_shape), dtype=torch.float32).reshape(up_shape) / 10 + offset,
        }
        if unsupported:
            tensors[target + ".dora_scale"] = torch.ones(out_dim)
        path = self.root / (name + ".safetensors")
        save_file(tensors, str(path), metadata=metadata or {})
        return scan_lora(path, compute_hash=False)

    def assert_merge_error(self, infos, code, sources, phrase, *, method="Weighted", unmatched_policy="scaled_carry"):
        with self.assertRaises(MergeError) as caught:
            merge_sources(infos, [1. / len(infos)] * len(infos),
                          MergeSettings(method=method, compression_mode="none", device="cpu",
                                        unmatched_policy=unmatched_policy))
        error = caught.exception
        self.assertIsInstance(error, ValueError)
        self.assertEqual(error.code, code)
        self.assertEqual(error.sources, sources)
        self.assertEqual(str(error), error.detail)
        self.assertIn(phrase, str(error))
        for name in sources:
            self.assertIn(name, str(error))
        return error

    def test_moss_mixed_identifies_the_non_moss_file(self):
        moss = self.source("moss", target="diffusion_model.layers.0.proj.o", metadata={"ss_base_model_version": "moss_image"})
        sdxl = self.source("wrong_sdxl", target="lora_unet_down_blocks_0_to_q")
        self.assert_merge_error([moss, sdxl], "moss_mixed", [sdxl.name], "MOSS-Image LoRAs must be blended")

    def test_same_filename_in_different_folders_identifies_only_the_incompatible_path(self):
        (self.root / "moss").mkdir()
        (self.root / "sdxl").mkdir()
        moss = self.source("moss/adapter_model", target="diffusion_model.layers.0.proj.o",
                           metadata={"ss_base_model_version": "moss_image"})
        sdxl = self.source("sdxl/adapter_model", target="lora_unet_down_blocks_0_to_q")
        self.assertEqual(moss.name, sdxl.name)
        error = self.assert_merge_error([moss, sdxl], "moss_mixed", [sdxl.name], "MOSS-Image LoRAs must be blended")
        self.assertEqual(error.source_paths, [str(sdxl.path)])
        self.assertNotIn(str(moss.path), error.source_paths)

    def test_unsupported_adapter_identifies_its_file(self):
        source = self.source("unsupported_dora", unsupported=True)
        self.assert_merge_error([source], "unsupported_adapter", [source.name], "contains unsupported adapter tensors")

    def test_disjoint_model_families_identify_both_files(self):
        sdxl = self.source("sdxl", target="lora_unet_down_blocks_0_to_q")
        flux = self.source("flux", target="double_blocks.0.img_attn.qkv")
        self.assert_merge_error([sdxl, flux], "model_mismatch", [sdxl.name, flux.name], "no matching target keys")

    def test_shape_mismatch_identifies_comparison_files_in_both_methods(self):
        first = self.source("shape_a")
        second = self.source("shape_b", out_dim=7)
        for method in ("Weighted", "Krea2 Gate-Aware"):
            with self.subTest(method=method):
                error = self.assert_merge_error([first, second], "shape_mismatch", [first.name, second.name],
                                                "incompatible shapes/layouts", method=method)
                self.assertIn(self.target, error.detail)

    def test_conv_kernel_mismatch_identifies_comparison_files(self):
        first = self.source("kernel_a", conv_tail=(4, 2, 2))
        second = self.source("kernel_b", conv_tail=(1, 4, 4))
        self.assert_merge_error([first, second], "shape_mismatch", [first.name, second.name],
                                "incompatible convolution kernel shapes")

    def test_missing_target_identifies_file_without_the_target(self):
        first = self.source("has_target", target="A.extra")
        second = self.source("missing_target", target="B.extra")
        for method in ("Weighted", "Krea2 Gate-Aware"):
            with self.subTest(method=method):
                error = self.assert_merge_error([first, second], "missing_targets", [second.name],
                                                "is not present", method=method, unmatched_policy="error")
                self.assertIn("A.extra", error.detail)

    def test_krea2_rejects_moss_with_source_names(self):
        first = self.source("moss_a", target="diffusion_model.layers.0.proj.o", metadata={"ss_base_model_version": "moss_image"})
        second = self.source("moss_b", target="diffusion_model.layers.0.proj.o", metadata={"ss_base_model_version": "moss_image"})
        self.assert_merge_error([first, second], "krea2_moss", [first.name, second.name],
                                "Krea2 Gate-Aware cannot be used with MOSS-Image", method="Krea2 Gate-Aware")

    def test_valid_weighted_blend_keeps_the_expected_delta(self):
        infos = [self.source("valid_a"), self.source("valid_b", offset=.2)]
        weights = [.3, .7]
        expected = torch.zeros(6, 4)
        for info, weight in zip(infos, weights):
            tensors = load_file(str(info.path), device="cpu")
            expected.add_(tensors[self.target + ".lora_B.weight"] @ tensors[self.target + ".lora_A.weight"], alpha=weight)
        result, report = merge_sources(infos, weights,
                                       MergeSettings(method="Weighted", compression_mode="none", output_dtype="fp32", device="cpu"))
        up, down = result[self.target + ".lora_B.weight"], result[self.target + ".lora_A.weight"]
        actual = up @ down * (result[self.target + ".alpha"] / down.shape[0])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(report.source_count, 2)
        self.assertEqual(report.warnings, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
