from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.engine import merge_sources
from core.formats import scan_lora
from core.recipe import build_recipe
from core.types import MergeSettings


GATE = "diffusion_model.block.0.attn.to_gate"
NORMAL = "diffusion_model.block.0.attn.to_q"
UNPAIRED = "diffusion_model.block.1.ff.gate"


def pair(tensors, target, up, down):
    tensors[f"{target}.lora_B.weight"] = torch.tensor(up, dtype=torch.float32)
    tensors[f"{target}.lora_A.weight"] = torch.tensor(down, dtype=torch.float32)


def delta(tensors, target):
    return tensors[f"{target}.lora_B.weight"] @ tensors[f"{target}.lora_A.weight"]


class Krea2MergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.a, self.b = {}, {}
        pair(self.a, GATE, [[2.0], [0.0]], [[1.0, 0.0]])
        pair(self.b, GATE, [[-1.0], [0.0]], [[1.0, 0.0]])
        pair(self.a, NORMAL, [[1.0], [2.0]], [[1.0, 0.0]])
        pair(self.b, NORMAL, [[0.0], [3.0]], [[0.0, 1.0]])
        pair(self.a, UNPAIRED, [[4.0], [0.0]], [[0.0, 1.0]])
        pa, pb = root / "a.safetensors", root / "b.safetensors"
        save_file(self.a, str(pa))
        save_file(self.b, str(pb))
        self.infos = [scan_lora(pa, compute_hash=False), scan_lora(pb, compute_hash=False)]

    def tearDown(self):
        self.tmp.cleanup()

    def test_conflict_pick_and_unpaired_gate_match_reference_rules(self):
        settings = MergeSettings(method="Krea2 Gate-Aware", output_dtype="auto", device="cpu")
        output, report = merge_sources(self.infos, [0.5, 0.25], settings)
        self.assertEqual(report.gate_conflicts, 1)
        self.assertEqual(output[f"{GATE}.lora_A.weight"].shape[0], 1)
        torch.testing.assert_close(delta(output, GATE), delta(self.a, GATE) * 0.5 * 0.5)
        torch.testing.assert_close(delta(output, NORMAL), delta(self.a, NORMAL) * 0.5 + delta(self.b, NORMAL) * 0.25)
        torch.testing.assert_close(delta(output, UNPAIRED), delta(self.a, UNPAIRED) * 0.5)
        self.assertEqual(output[f"{GATE}.lora_A.weight"].dtype, torch.float32)

    def test_zero_out_and_nonconflicting_gate_compression(self):
        settings = MergeSettings(method="Krea2 Gate-Aware", krea2_opposite_strategy="zero_out", krea2_gate_rank=2)
        output, report = merge_sources(self.infos, [0.5, 0.25], settings)
        self.assertEqual(report.gate_conflicts, 1)
        self.assertEqual(output[f"{GATE}.lora_A.weight"].shape[0], 2)
        self.assertEqual(torch.count_nonzero(delta(output, GATE)).item(), 0)

        self.b[f"{GATE}.lora_B.weight"] *= -1
        save_file(self.b, str(self.infos[1].path) + ".new")
        fresh = scan_lora(str(self.infos[1].path) + ".new", compute_hash=False)
        settings.krea2_gate_rank = 1
        output, report = merge_sources([self.infos[0], fresh], [0.5, 0.25], settings)
        self.assertEqual(report.gate_conflicts, 0)
        self.assertEqual(output[f"{GATE}.lora_A.weight"].shape[0], 1)
        torch.testing.assert_close(delta(output, GATE), delta(self.a, GATE) * 0.25 + delta(self.b, GATE) * 0.125)

    def test_krea2_is_explicit_and_requires_two_sources(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            merge_sources(self.infos[:1], [0.5], MergeSettings(method="Krea2 Gate-Aware"))
        ordinary, report = merge_sources(self.infos, [0.5, 0.25], MergeSettings(method="Weighted", compression_mode="none"))
        self.assertEqual(report.method, "Weighted")
        torch.testing.assert_close(delta(ordinary, GATE), delta(self.a, GATE) * 0.5 + delta(self.b, GATE) * 0.25)
        recipe = build_recipe(self.infos, [0.5, 0.25], MergeSettings(method="Weighted"), report)
        self.assertFalse(any(key.startswith("krea2_") for key in recipe["settings"]))

    def test_balance_and_normal_rank_cap(self):
        settings = MergeSettings(method="Krea2 Gate-Aware", krea2_balance_mode="per_layer_norm",
                                 krea2_gate_compression="off", krea2_normal_compression="off")
        output, _ = merge_sources(self.infos, [0.5, 0.25], settings)
        a = delta(self.a, NORMAL) * 0.5
        b = delta(self.b, NORMAL) * 0.25
        na, nb = torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)
        expected = a * ((na + nb) / (2 * na)) + b * ((na + nb) / (2 * nb))
        torch.testing.assert_close(delta(output, NORMAL), expected)
        torch.testing.assert_close(delta(output, UNPAIRED), delta(self.a, UNPAIRED) * 0.5)

        settings.krea2_normal_compression = "max_input_rank"
        output, _ = merge_sources(self.infos, [0.5, 0.25], settings)
        self.assertEqual(output[f"{NORMAL}.lora_A.weight"].shape[0], 1)

    def test_output_rank_caps_krea2_layers_and_reduces_saved_size(self):
        auto = MergeSettings(method="Krea2 Gate-Aware", krea2_opposite_strategy="zero_out", krea2_gate_rank=2)
        full, _ = merge_sources(self.infos, [0.5, 0.25], auto)
        capped = MergeSettings(method="Krea2 Gate-Aware", output_rank=1,
                               krea2_opposite_strategy="zero_out", krea2_gate_rank=2)
        smaller, report = merge_sources(self.infos, [0.5, 0.25], capped)

        self.assertEqual(full[f"{NORMAL}.lora_A.weight"].shape[0], 2)
        self.assertEqual(full[f"{GATE}.lora_A.weight"].shape[0], 2)
        self.assertTrue(all(layer.output_rank <= 1 for layer in report.layer_reports))
        self.assertTrue(all(tensor.shape[0] <= 1 for name, tensor in smaller.items()
                            if name.endswith(".lora_A.weight")))
        self.assertEqual(report.gate_conflicts, 1)
        self.assertEqual(torch.count_nonzero(delta(smaller, GATE)).item(), 0)

        dense = delta(self.a, NORMAL) * 0.5 + delta(self.b, NORMAL) * 0.25
        u, s, vh = torch.linalg.svd(dense, full_matrices=False)
        expected = (u[:, :1] * s[:1]) @ vh[:1, :]
        torch.testing.assert_close(delta(smaller, NORMAL), expected)

        full_path = Path(self.tmp.name) / "auto.safetensors"
        small_path = Path(self.tmp.name) / "rank1.safetensors"
        save_file(full, str(full_path))
        save_file(smaller, str(small_path))
        self.assertLess(small_path.stat().st_size, full_path.stat().st_size)

    def test_output_rank_caps_picked_conflicting_gate(self):
        first, second = {}, {}
        pair(first, GATE, [[2.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])
        pair(second, GATE, [[-1.0, 0.0], [0.0, -1.0]], [[1.0, 0.0], [0.0, 1.0]])
        first_path = Path(self.tmp.name) / "gate_a.safetensors"
        second_path = Path(self.tmp.name) / "gate_b.safetensors"
        save_file(first, str(first_path))
        save_file(second, str(second_path))
        infos = [scan_lora(first_path, compute_hash=False), scan_lora(second_path, compute_hash=False)]

        output, report = merge_sources(infos, [0.5, 0.25],
                                       MergeSettings(method="Krea2 Gate-Aware", output_rank=1))
        self.assertEqual(report.gate_conflicts, 1)
        self.assertEqual(output[f"{GATE}.lora_A.weight"].shape[0], 1)
        torch.testing.assert_close(delta(output, GATE), torch.tensor([[0.5, 0.0], [0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
