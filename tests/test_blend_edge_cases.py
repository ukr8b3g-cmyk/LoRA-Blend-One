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
from core.errors import MergeError
from core.formats import scan_lora
from core.math_ops import ties_sparsify
from core.types import MergeSettings


class BlendEdgeCases(unittest.TestCase):
    def test_ties_keeps_first_equal_magnitudes_after_larger_values(self):
        values = torch.tensor([[1., -2., 2., 9.], [-2., 1., -2., -8.]])
        expected = torch.tensor([[0., -2., 2., 9.], [0., 0., 0., -8.]])
        torch.testing.assert_close(ties_sparsify(values, 0.5), expected)

    def test_ties_density_with_large_outlier(self):
        values = torch.arange(1, 2_000_002, dtype=torch.float32)
        values[-1] = 1e15
        keep = round(0.25 * values.numel())
        expected = values.clone()
        expected[:-keep] = 0
        self.assertTrue(torch.equal(ties_sparsify(values, 0.25), expected))

    def test_krea2_drop_rejects_a_merge_without_shared_layers(self):
        with tempfile.TemporaryDirectory() as td:
            infos = []
            for i in range(2):
                path = Path(td) / f"source{i}.safetensors"
                layer = f"diffusion_model.blocks.{i}.attn.gate"
                save_file({layer + ".lora_A.weight": torch.ones(1, 3),
                           layer + ".lora_B.weight": torch.ones(2, 1)}, str(path))
                infos.append(scan_lora(path, compute_hash=False))
            settings = MergeSettings(method="Krea2 Gate-Aware", unmatched_policy="drop", device="cpu")
            with self.assertRaisesRegex(MergeError, "dropped as unmatched") as error:
                merge_sources(infos, [0.3, 0.7], settings)
            self.assertEqual(error.exception.code, "missing_targets")
