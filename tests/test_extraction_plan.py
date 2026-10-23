import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file, load_file


class ExtractionPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = Path(__file__).resolve().parents[1]
        self.a = {'layers.0.proj.qkv.weight': torch.zeros(6, 2), 'final_norm_video.weight': torch.ones(2)}
        self.b = {key: value + .1 for key, value in self.a.items()}
        self.write()
        package = types.ModuleType('plan_test'); package.__path__ = [str(self.repo)]
        folders = types.ModuleType('folder_paths')
        folders.get_filename_list = lambda category: ['base.safetensors', 'modified.safetensors']
        folders.get_full_path_or_raise = lambda category, name: str(self.root / name)
        folders.get_output_directory = lambda: str(self.root / 'output')
        comfy = types.ModuleType('comfy')
        management = types.ModuleType('comfy.model_management')
        management.throw_exception_if_processing_interrupted = lambda: None
        utils = types.ModuleType('comfy.utils')
        utils.ProgressBar = lambda total: types.SimpleNamespace(update_absolute=lambda *args: None)
        comfy.model_management = management; comfy.utils = utils
        self.modules = patch.dict(sys.modules, {'plan_test': package, 'folder_paths': folders, 'comfy': comfy,
            'comfy.model_management': management, 'comfy.utils': utils, 'server': None})
        self.modules.start()
        spec = importlib.util.spec_from_file_location('plan_test.extract_nodes', self.repo / 'extract_nodes.py')
        self.module = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.module)
        self.base = 'diffusion_models: base.safetensors'
        self.modified = 'diffusion_models: modified.safetensors'

    def tearDown(self):
        self.modules.stop(); self.tmp.cleanup()

    def write(self):
        save_file(self.a, str(self.root / 'base.safetensors'))
        save_file(self.b, str(self.root / 'modified.safetensors'))

    def inspect(self, **kwargs):
        return self.module.LoRABlendModelInspect().run(self.base, self.modified, **kwargs)

    def run_node(self, plan=None, **kwargs):
        options = dict(device='cpu', non_lora='Store Diff', hash_inputs=False)
        options.update(kwargs)
        return self.module.LoRABlendModelExtract().run(self.base, self.modified, 'Fixed Rank', 2, .99, 'fp32',
            'check', extraction_plan=plan, **options)

    def test_legacy_output_and_new_plan(self):
        result = self.inspect(scan='Full Scan')
        info, plan = result['result']
        self.assertEqual(json.loads(info)['changed_non_matrix'], 1)
        self.assertEqual(plan['settings']['non_lora'], 'Store Diff')
        self.assertTrue(plan['ready'])
        self.assertEqual(plan['version'], 2)
        self.assertEqual(plan['settings']['mode'], 'Fixed Rank')
        self.assertEqual(plan['settings']['rank'], 32)
        self.assertEqual(plan['settings']['max_rank'], 64)
        schema = self.module.LoRABlendModelExtract.INPUT_TYPES()
        self.assertEqual(schema['required']['rank'][1]['default'], 32)
        self.assertEqual(plan['non_matrix_examples'], ['diffusion_model.final_norm_video.weight'])
        self.assertIsNone(plan['required_rank']); self.assertIsNone(plan['output_bytes'])
        self.assertEqual(self.module.LoRABlendModelInspect.RETURN_TYPES, ('STRING', 'LBO_EXTRACTION_PLAN'))
        self.assertEqual(self.module.LoRABlendModelExtract.RETURN_TYPES, ('STRING',))

    def test_review_never_calls_extract_or_saves(self):
        plan = self.inspect(scan='Full Scan')['result'][1]
        with patch.object(self.module, 'extract', side_effect=AssertionError('must not extract')):
            result = self.run_node(plan)
        self.assertTrue(json.loads(result['result'][0])['review_only'])
        self.assertEqual(json.loads(result['result'][0])['effective_settings']['rank'], 2)
        self.assertFalse((self.root / 'output').exists())

    def test_connected_settings_preserve_manual_and_roundtrip(self):
        plan = self.inspect(scan='Full Scan')['result'][1]
        # Keep the real engine and assert the boundary arguments without allocating a GPU.
        engine = self.module.extract
        def cpu_extract(*args, **kwargs):
            self.assertEqual(kwargs['mode'], 'Fixed Rank')
            self.assertEqual(kwargs['rank'], 2)
            self.assertEqual(kwargs['dtype'], 'fp32')
            self.assertEqual(kwargs.get('seed', 0), 0)
            self.assertEqual(kwargs['non_lora'], 'Store Diff')
            kwargs['device'] = 'cpu'
            return engine(*args, **kwargs)
        with patch.object(self.module, 'extract', side_effect=cpu_extract):
            result = self.run_node(plan, plan_action='Extract and Save')
        self.assertIn('hybrid_delta', result['result'][0])
        saved = next((self.root / 'output').glob('*.safetensors'))
        tensors = load_file(str(saved))
        self.assertIn('diffusion_model.final_norm_video.diff', tensors)
        self.assertTrue(all(torch.isfinite(t).all() for t in tensors.values()))

    def test_manual_unchanged_even_with_default_review_action(self):
        result = self.run_node()
        self.assertIn('hybrid_delta', result['result'][0])
        report = json.loads(next((self.root / 'output').glob('*.report.json')).read_text())
        self.assertEqual(report['mode'], 'Fixed Rank'); self.assertEqual(report['dtype'], 'fp32')

    def test_header_advice_does_not_block_explicit_manual_extraction(self):
        plan = self.inspect(scan='Header')['result'][1]
        self.assertEqual(plan['blocked_reason'], 'full_scan_required')
        self.run_node(plan)
        result = self.run_node(plan, plan_action='Extract and Save')
        self.assertIn('hybrid_delta', result['result'][0])

    def test_partial_consent_and_standard_output(self):
        plan = self.inspect(scan='Full Scan', purpose='Blend One Source')['result'][1]
        self.assertEqual(plan['blocked_reason'], 'partial_consent_required')
        with self.assertRaisesRegex(ValueError, 'changed non-LoRA parameter'):
            self.run_node(plan, plan_action='Extract and Save', non_lora='Stop')
        plan = self.inspect(scan='Full Scan', purpose='Blend One Source', allow_partial=True)['result'][1]
        self.assertTrue(plan['partial'])
        result = self.run_node(plan, plan_action='Extract and Save', non_lora='Exclude With Report')
        self.assertIn('standard_lora', result['result'][0]); self.assertIn('PARTIAL', result['result'][0])

    def test_recommendation_never_overrides_manual_and_changed_file_refused(self):
        plan = self.inspect(scan='Full Scan')['result'][1]
        plan['settings']['max_rank'] = 3072
        result = self.run_node(plan, plan_action='Extract and Save')
        self.assertEqual(json.loads(result['ui']['text'][0])['effective_settings']['max_rank'], 64)
        plan = self.inspect(scan='Full Scan')['result'][1]
        self.b['final_norm_video.weight'] += 1; self.write()
        with self.assertRaisesRegex(ValueError, 'changed after inspection'):
            self.run_node(plan, plan_action='Extract and Save')

    def test_mismatched_pair_and_old_plan_refused(self):
        plan = self.inspect(scan='Full Scan')['result'][1]
        plan['base_model'] = self.modified
        with self.assertRaisesRegex(ValueError, 'Models differ'):
            self.run_node(plan, plan_action='Extract and Save')
        plan = self.inspect(scan='Full Scan')['result'][1]
        plan['version'] = 1
        with self.assertRaises(ValueError):
            self.run_node(plan, plan_action='Extract and Save')
        self.assertFalse((self.root / 'output').exists())

    def test_no_changes_and_nonmatrix_only_partial(self):
        self.b = {key: value.clone() for key, value in self.a.items()}; self.write()
        self.assertEqual(self.inspect(scan='Full Scan')['result'][1]['blocked_reason'], 'no_difference')
        self.b['final_norm_video.weight'] += 1; self.write()
        self.assertEqual(self.inspect(scan='Full Scan', purpose='Blend One Source', allow_partial=True)['result'][1]['blocked_reason'], 'no_matrix_difference')


if __name__ == '__main__':
    unittest.main()
