import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from core.formats import scan_lora


class PeftPatternTests(unittest.TestCase):
    def test_scaled_layers_use_peft_regex_order_boundaries_and_anchors(self):
        # Use the key names written by PEFT's save_pretrained().
        modules = {
            "blocks.0.q_proj": (4, 3.0),
            "nested.blocks.0.q_proj": (1, 4.0),
            "blocks.0.notq_proj": (1, 1.0),
            "blocks.0.v_proj": (4, 2.0),
            "blocks.0.k_proj": (4, 2.0),
        }
        config = {
            "r": 1,
            "lora_alpha": 1,
            "use_rslora": True,
            "rank_pattern": {r"^blocks\.\d+\.(q_proj|v_proj)": 4},
            "alpha_pattern": {
                r"^blocks\.\d+\.q_proj": 6, "q_proj": 4,
                "v_proj": 4, r"^blocks\.\d+\.v_proj": 16, "k_proj": 48,
            },
        }
        tensors = {}
        for module, (rank, _) in modules.items():
            prefix = f"base_model.model.{module}"
            tensors[f"{prefix}.lora_A.weight"] = torch.ones(rank, 4)
            tensors[f"{prefix}.lora_B.weight"] = torch.ones(4, rank)
        tensors["base_model.model.blocks.0.k_proj.alpha"] = torch.tensor(8.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_model.safetensors"
            path.with_name("adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
            save_file(tensors, str(path))
            info = scan_lora(path, compute_hash=False)
        for module, (_, scale) in modules.items():
            with self.subTest(module=module):
                self.assertEqual(info.layers[f"base_model.model.{module}"].effective_scale, scale)
