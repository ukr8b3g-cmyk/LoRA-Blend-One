from __future__ import annotations

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

from core.engine import merge_sources
from core.formats import OpenedSources, scan_lora
from core.recipe import build_metadata, build_recipe
from core.types import MergeSettings
from core.writer import save_verified


PREFIX = "diffusion_model.layers.0.proj."
MOSS = {"ss_base_model_version": "moss_image"}


def pair(tensors, target, up, down, alpha=None, style="peft"):
    suffixes = (".lora_B.weight", ".lora_A.weight") if style == "peft" else (".lora_up.weight", ".lora_down.weight")
    tensors[target + suffixes[0]] = up.clone().contiguous()
    tensors[target + suffixes[1]] = down.clone().contiguous()
    if alpha is not None:
        tensors[target + ".alpha"] = torch.tensor(float(alpha))


def delta(tensors, target):
    up, down = tensors[target + ".lora_B.weight"].float(), tensors[target + ".lora_A.weight"].float()
    return up @ down * (float(tensors[target + ".alpha"]) / down.shape[0])


class MossMergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def source(self, name, tensors, metadata=MOSS):
        path = self.root / (name + ".safetensors")
        save_file(tensors, str(path), metadata=metadata)
        return scan_lora(path, compute_hash=False)

    def test_qkv_and_prefix_forms_agree_for_all_four_methods(self):
        gen = torch.Generator().manual_seed(345)
        down = torch.randn(2, 4, generator=gen)
        up = torch.randn(12, 2, generator=gen)
        fused, split, other = {}, {}, {}
        pair(fused, "lora_unet_layers_0_proj_qkv", up, down, 1, "regular")
        for name, block in zip(("q", "k", "v"), up.chunk(3, 0)):
            pair(split, "base_model.model.transformer.layers.0.proj." + name, block, down, 1)
            pair(other, PREFIX + name, torch.randn(4, 1, generator=gen), torch.randn(1, 4, generator=gen), 1)
        a, b, c = self.source("fused", fused), self.source("split", split), self.source("other", other)
        for method in ("Weighted", "TIES", "DARE", "DELLA"):
            with self.subTest(method=method):
                settings = MergeSettings(method=method, compression_mode="max_input_rank", density=0.8,
                                         seed=31, output_dtype="fp32", device="cpu")
                first, _ = merge_sources([a, c], [0.8, -0.2], settings)
                second, _ = merge_sources([b, c], [0.8, -0.2], settings)
                self.assertEqual(set(first), set(second))
                for name in ("q", "k", "v"):
                    torch.testing.assert_close(delta(first, PREFIX + name), delta(second, PREFIX + name))

    def test_ties_resolves_conflict_across_fused_and_split_inputs(self):
        a, b = {}, {}
        pair(a, PREFIX + "qkv", torch.tensor([[2.], [0.], [0.]]), torch.ones(1, 1))
        pair(b, PREFIX + "q", torch.tensor([[-1.]]), torch.ones(1, 1))
        tensors, _ = merge_sources([self.source("a", a), self.source("b", b)], [0.5, 0.5],
                                  MergeSettings(method="TIES", density=1., device="cpu", output_dtype="fp32"))
        torch.testing.assert_close(delta(tensors, PREFIX + "q"), torch.tensor([[1.]]))

    def test_block_diagonal_qkv_preserves_alpha_and_logical_rank(self):
        tensors = {}
        up = torch.diag(torch.tensor([2., 3., 4.]))
        down = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        pair(tensors, PREFIX + "qkv", up, down, 1.5)
        info = self.source("block", tensors)
        self.assertTrue(all(spec.rank == 1 for spec in info.layers.values()))
        with OpenedSources([info]) as opened:
            for idx, name in enumerate(("q", "k", "v")):
                spec = info.layers[PREFIX + name]
                u, d = opened.read_factors(0, spec)
                torch.testing.assert_close(u @ d * spec.effective_scale, (up @ down)[idx:idx + 1] * 0.5)

    def test_fused_and_split_within_one_source_are_added_before_sparsifying(self):
        tensors = {}
        pair(tensors, PREFIX + "qkv", torch.tensor([[2.], [0.], [0.]]), torch.ones(1, 1))
        pair(tensors, PREFIX + "q", torch.tensor([[-1.]]), torch.ones(1, 1))
        info = self.source("stacked", tensors)
        output, _ = merge_sources([info], [1.], MergeSettings(method="TIES", density=1., device="cpu"))
        torch.testing.assert_close(delta(output, PREFIX + "q"), torch.tensor([[1.]]))

    def test_rejects_duplicate_aliases_and_unconsumed_tensors(self):
        tensors = {}
        pair(tensors, PREFIX + "o", torch.ones(2, 1), torch.ones(1, 2))
        duplicate = dict(tensors)
        pair(duplicate, "lora_unet_layers_0_proj_o", torch.ones(2, 1), torch.ones(1, 2), style="regular")
        with self.assertRaisesRegex(ValueError, "Duplicate MOSS-Image aliases"):
            self.source("duplicate", duplicate)
        for idx, suffix in enumerate((".dora_scale", ".lokr_w1", ".oft_blocks", ".w_norm", ".lora_B.bias", ".diff")):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "unsupported or incomplete"):
                self.source("unsupported" + str(idx), {**tensors, PREFIX + "o" + suffix: torch.ones(1)})

    def test_identifies_real_shape_without_metadata_and_leaves_other_models_alone(self):
        tensors = {}
        pair(tensors, "transformer.layers.0.proj.ffn_up_gate", torch.ones(16384, 1), torch.ones(1, 3072))
        info = self.source("real_shape", tensors, metadata={})
        self.assertEqual(info.architecture_hint, "moss_image")
        other = {}
        target = "transformer_blocks.0.attn.to_q"
        pair(other, target, torch.ones(3, 1), torch.ones(1, 2))
        generic = self.source("generic", other, metadata={})
        self.assertNotEqual(generic.architecture_hint, "moss_image")
        self.assertEqual(set(generic.layers), {target})
        self.assertFalse(generic.layers[target].components)
        with self.assertRaisesRegex(ValueError, "recognized MOSS-Image"):
            merge_sources([info, generic], [0.5, 0.5], MergeSettings())
        with self.assertRaisesRegex(ValueError, "Krea2 Gate-Aware"):
            merge_sources([info], [1.], MergeSettings(method="Krea2 Gate-Aware"))

    def test_text_embedder_qkv_is_not_split(self):
        tensors = {}
        target = "lora_unet_text_embedder_blocks_0_qkv"
        pair(tensors, target, torch.ones(6, 1), torch.ones(1, 2), style="regular")
        info = self.source("text", tensors)
        self.assertEqual(set(info.layers), {"diffusion_model.text_embedder.blocks.0.qkv"})

    def test_rank_save_reload_and_metadata(self):
        gen = torch.Generator().manual_seed(500)
        tensors = {}
        pair(tensors, PREFIX + "qkv", torch.randn(24, 4, generator=gen).bfloat16(),
             torch.randn(4, 8, generator=gen).bfloat16(), 2)
        info = self.source("save_source", tensors)
        sizes = []
        for rank in (1, 3):
            settings = MergeSettings(method="Weighted", output_rank=rank, output_dtype="bf16", device="cpu")
            output, report = merge_sources([info], [0.8], settings)
            metadata = build_metadata(build_recipe([info], [0.8], settings, report))
            path = self.root / f"rank{rank}.safetensors"
            save_verified(output, path, metadata=metadata, write_report=False)
            loaded = scan_lora(path, compute_hash=False)
            self.assertEqual(loaded.architecture_hint, "moss_image")
            self.assertTrue(all(spec.rank <= rank for spec in loaded.layers.values()))
            self.assertTrue(all(spec.effective_scale == 1. for spec in loaded.layers.values()))
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                self.assertEqual(handle.metadata()["lbo_moss_layout"], "split_qkv")
                self.assertTrue(torch.equal(handle.get_tensor(PREFIX + "q.lora_A.weight"), output[PREFIX + "q.lora_A.weight"]))
            sizes.append(path.stat().st_size)
        self.assertLess(sizes[0], sizes[1])


if __name__ == "__main__":
    unittest.main()
