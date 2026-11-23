import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.engine import merge_sources
from core.errors import MergeError
from core.formats import scan_lora
from core.types import MergeSettings


class AliasBlendTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)
        self.settings = MergeSettings(compression_mode="none", output_dtype="fp32", device="cpu")

    def source(self, name, targets, metadata=None):
        state = {}
        for key, value in targets.items():
            state[key + ".lora_A.weight"] = torch.eye(2)
            state[key + ".lora_B.weight"] = torch.eye(2) * value
        p = self.folder / (name + ".safetensors")
        save_file(state, str(p), metadata=metadata)
        return scan_lora(p, compute_hash=False)

    def delta(self, state, k):
        return state[k + ".lora_B.weight"] @ state[k + ".lora_A.weight"]

    def test_qwen_prefixes_merge_but_other_projections_stay_disjoint(self):
        q = "transformer_blocks.0.attn.add_q_proj"
        v = "transformer_blocks.0.attn.add_v_proj"
        infos = [self.source("a", {q: 1}),
                 self.source("b", {"transformer." + q: 4, "transformer." + v: 2}),
                 self.source("c", {"diffusion_model." + q: 2})]
        out, report = merge_sources(infos, [.2, .5, .3], self.settings)
        torch.testing.assert_close(self.delta(out, q), torch.eye(2) * 2.8)
        torch.testing.assert_close(self.delta(out, "transformer." + v), torch.eye(2))
        self.assertEqual(sorted(l.contributors for l in report.layer_reports), [1, 3])
        self.assertIn("transformer." + q, infos[1].layers)

    def test_krea_native_and_diffusers_match_for_weighted_and_gate_aware(self):
        native = "diffusion_model.blocks.1.attn.gate"
        alt = "transformer.transformer_blocks.1.attn.to_gate"
        infos = [self.source("a", {native: 1}, {"ss_base_model_version": "krea2"}),
                 self.source("b", {alt: 4})]
        for method, value in (("Weighted", 3.1), ("Krea2 Gate-Aware", 1.55)):
            with self.subTest(method=method):
                self.settings.method = method
                out, report = merge_sources(infos, [.3, .7], self.settings)
                self.assertEqual(len(report.layer_reports), 1)
                self.assertEqual(report.gate_targets, 1)
                torch.testing.assert_close(self.delta(out, native), torch.eye(2) * value)

    def test_sdxl_first_and_second_text_encoders_stay_distinct(self):
        path = "text_model.encoder.layers.0.self_attn.q_proj"
        ko = "lora_te1_" + path.replace(".", "_")
        te1, te2 = "text_encoder." + path, "text_encoder_2." + path
        infos = [self.source("a", {ko: 1}), self.source("b", {te1: 4, te2: 2})]
        out, report = merge_sources(infos, [.3, .7], self.settings)
        self.assertEqual(len(report.layer_reports), 2)
        torch.testing.assert_close(self.delta(out, te1), torch.eye(2) * 3.1)
        torch.testing.assert_close(self.delta(out, te2), torch.eye(2) * 1.4)

    def test_duplicate_physical_target_within_one_input_identifies_both_aliases(self):
        q = "transformer_blocks.0.attn.add_q_proj"
        info = self.source("ambiguous", {q: 1, "transformer." + q: 2})
        with self.assertRaises(MergeError) as caught:
            merge_sources([info], [1], self.settings)
        self.assertIn(repr(q), str(caught.exception))
        self.assertIn(repr("transformer." + q), str(caught.exception))

    def test_krea_native_and_flat_keys_work_without_a_diffusers_source(self):
        k = "diffusion_model.blocks.1.attn.gate"
        flat = "lora_unet_blocks_1_attn_gate"
        self.settings.method = "Krea2 Gate-Aware"
        for a, b in ((k, flat), (flat, "lycoris_transformer_blocks_1_attn_to_gate"), (flat, flat)):
            with self.subTest(a=a, b=b):
                out, report = merge_sources([self.source("a", {a: 1}), self.source("b", {b: 4})],
                                            [.3, .7], self.settings)
                self.assertEqual(len(report.layer_reports), 1)
                self.assertEqual(report.gate_targets, 1)
                torch.testing.assert_close(self.delta(out, report.layer_reports[0].target), torch.eye(2) * 1.55)

    def test_clip_kohya_names_share_a_layer_without_diffusers_input(self):
        a = self.source("a", {"lora_te_text_model_encoder_layers_0_mlp_fc1": 1})
        b = self.source("b", {"lora_te1_text_model_encoder_layers_0_mlp_fc1": 4})
        out, report = merge_sources([a, b], [.3, .7], self.settings)
        self.assertEqual(len(report.layer_reports), 1)
        torch.testing.assert_close(self.delta(out, report.layer_reports[0].target), torch.eye(2) * 3.1)

    def test_unknown_family_identifies_unresolved_aliases_and_allows_disjoint_layers(self):
        a = self.source("a", {"diffusion_model.blocks.0.attn.wq": 1})
        b = self.source("b", {"transformer.transformer_blocks.0.attn.to_q": 4})
        with self.assertRaisesRegex(MergeError, "Cannot identify the model.*attn.wq.*to_q"):
            merge_sources([a, b], [.3, .7], self.settings)
        b = self.source("b", {"transformer.transformer_blocks.1.attn.to_q": 4})
        out, report = merge_sources([a, b], [.3, .7], self.settings)
        self.assertEqual(len(report.layer_reports), 2)

    def test_different_families_do_not_gain_shared_layers(self):
        a = self.source("a", {"diffusion_model.blocks.1.attn.wq": 1}, {"ss_base_model_version":"krea2"})
        b = self.source("b", {"transformer.transformer_blocks.1.attn.to_q": 1,
                              "transformer.transformer_blocks.1.attn.add_q_proj": 4})
        out, report = merge_sources([a, b], [.3, .7], self.settings, compatibility_mode="allow_disjoint")
        self.assertEqual(len(report.layer_reports), 3)
        self.assertTrue(all(l.unmatched for l in report.layer_reports))
