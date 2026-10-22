import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.engine import merge_sources, resolve_weights
from core.formats import scan_lora
from core.types import MergeSettings
from core.writer import save_verified


class SaveSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.target = 'diffusion_model.block.0.attn.to_q'

    def tearDown(self):
        self.tmp.cleanup()

    def source(self, name='input', up=2., down=0.5, extra=None, metadata=None):
        tensors = {self.target + '.lora_A.weight': torch.tensor([[down]], dtype=torch.float32),
                   self.target + '.lora_B.weight': torch.tensor([[up]], dtype=torch.float32)}
        tensors.update(extra or {})
        path = self.root / (name + '.safetensors')
        save_file(tensors, str(path), metadata=metadata)
        return scan_lora(path, compute_hash=False)

    def test_nonfinite_input_factors_rejected(self):
        for side in ('up', 'down'):
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(side=side, value=value):
                    with self.assertRaisesRegex(ValueError, 'input.*' + self.target):
                        info = self.source(**{side: value})
                        merge_sources([info], [1.], MergeSettings(device='cpu', compression_mode='none'))

    def test_nonfinite_alpha_rejected(self):
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.source(extra={self.target + '.alpha': torch.tensor(value)})

    def test_nonfinite_adapter_alpha_rejected(self):
        (self.root / 'adapter_config.json').write_text(json.dumps({'lora_alpha': float('nan')}))
        with self.assertRaises(ValueError):
            self.source()

    def test_nonfinite_coefficients_rejected(self):
        info = self.source()
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_weights([value], 'direct')
                with self.assertRaises(ValueError):
                    merge_sources([info], [value], MergeSettings(device='cpu', compression_mode='none'))
                with self.assertRaises(ValueError):
                    merge_sources([info], [1.], MergeSettings(device='cpu', output_scale=value))

    def test_fp16_overflow_rejected_weighted_and_krea2(self):
        infos = [self.source('large', up=100000., down=0.00001), self.source('small')]
        for method in ('Weighted', 'Krea2 Gate-Aware'):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'float16'):
                merge_sources(infos, [1., 1.], MergeSettings(method=method, device='cpu',
                              compression_mode='none', output_dtype='fp16'))

    def test_finite_storage_roundtrip(self):
        infos = [self.source('a'), self.source('b')]
        for method in ('Weighted', 'Krea2 Gate-Aware'):
            for dtype in ('fp16', 'bf16', 'fp32'):
                with self.subTest(method=method, dtype=dtype):
                    tensors, _ = merge_sources(infos, [.5, .25], MergeSettings(method=method, device='cpu',
                                      compression_mode='none', output_dtype=dtype))
                    output = self.root / 'out.safetensors'
                    save_verified(tensors, output, metadata={})
                    loaded = load_file(str(output))
                    self.assertTrue(all(torch.isfinite(t).all() for t in loaded.values()))
                    actual = loaded[self.target + '.lora_B.weight'].float() @ loaded[self.target + '.lora_A.weight'].float()
                    torch.testing.assert_close(actual, torch.tensor([[.75]]), atol=.005, rtol=.005)

    def test_unconsumed_tensors_rejected(self):
        for suffix in ('.lora_B.bias', '.oft_blocks', '.w_norm'):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, 'input.*' + suffix):
                info = self.source(extra={self.target + suffix: torch.ones(1)})
                merge_sources([info], [1.], MergeSettings(device='cpu', compression_mode='none'))

    def test_writer_nonfinite_preserves_existing_file(self):
        output = self.root / 'existing.safetensors'
        output.write_bytes(b'existing')
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'broken'):
                save_verified({'broken': torch.tensor(value)}, output, metadata={})
            self.assertEqual(output.read_bytes(), b'existing')
            self.assertFalse(output.with_name(output.name + '.partial').exists())

    def test_report_io_failure_keeps_success_and_existing_report(self):
        output = self.root / 'out.safetensors'
        sidecar = output.with_suffix('.safetensors.report.json')
        sidecar.write_text('previous report')
        report = {'warnings': []}
        original_open = Path.open
        def failing_open(path, *args, **kwargs):
            if '.report.json' in path.name:
                raise OSError('injected report failure')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', failing_open):
            result = save_verified({'ok': torch.ones(1)}, output, metadata={}, report=report)
        self.assertEqual(result, output)
        self.assertTrue(output.is_file())
        self.assertTrue(torch.equal(load_file(str(output))['ok'], torch.ones(1)))
        self.assertIn(str(output), report['warnings'][0])
        self.assertIn('injected report failure', report['warnings'][0])
        self.assertEqual(sidecar.read_text(), 'previous report')

    def test_report_success(self):
        output = self.root / 'out.safetensors'
        save_verified({'ok': torch.ones(1)}, output, metadata={}, report={'warnings': []})
        self.assertEqual(json.loads(output.with_suffix('.safetensors.report.json').read_text()), {'warnings': []})

    def test_moss_nonfinite_components_rejected(self):
        self.target = 'diffusion_model.layers.0.proj.qkv'
        path = self.root / 'moss.safetensors'
        save_file({self.target + '.lora_A.weight': torch.ones(1, 1),
                   self.target + '.lora_B.weight': torch.tensor([[1.], [float('nan')], [1.]])},
                  str(path), metadata={'ss_base_model_version': 'moss_image'})
        info = scan_lora(path, compute_hash=False)
        with self.assertRaisesRegex(ValueError, 'moss.*NaN or Inf'):
            merge_sources([info], [1.], MergeSettings(device='cpu', compression_mode='none'))

    def test_effective_coefficient_overflow_rejected(self):
        info = self.source(extra={self.target + '.alpha': torch.tensor(1e30)})
        with self.assertRaises(ValueError):
            merge_sources([info], [1e200], MergeSettings(device='cpu', output_scale=1e200))

    def test_report_disabled_does_not_open_sidecar(self):
        output = self.root / 'out.safetensors'
        with patch.object(Path, 'open', side_effect=OSError('must not open report')):
            self.assertEqual(save_verified({'ok': torch.ones(1)}, output, metadata={},
                                          report={'warnings': []}, write_report=False), output)

    def test_nonfinite_krea_gate_coefficient_rejected(self):
        self.target = 'diffusion_model.block.0.attn.to_gate'
        infos = [self.source('a'), self.source('b')]
        with self.assertRaisesRegex(ValueError, 'NaN or Inf'):
            merge_sources(infos, [.5, .5], MergeSettings(method='Krea2 Gate-Aware', device='cpu',
                          krea2_gate_strength=float('nan')))


if __name__ == '__main__':
    unittest.main()
