import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.writer import Spool


class WriterTests(unittest.TestCase):
    def test_memory_and_spill_are_bitwise_equal(self):
        with tempfile.TemporaryDirectory() as root:
            payload = {'x': torch.arange(24, dtype=torch.bfloat16).reshape(6, 4),
                       'alpha': torch.tensor(32.), 'zero': torch.tensor(-0.)}
            for limit in (0, 49, 1024):
                folder = Path(root) / str(limit); folder.mkdir()
                spool = Spool(folder, lambda: None, memory_limit=limit)
                spool.add({'x': payload['x']})
                if limit: self.assertEqual(list(folder.iterdir()), [])
                spool.add({k: v for k, v in payload.items() if k != 'x'})
                if limit == 49: self.assertFalse(spool.in_memory)
                output = spool.finish({'recipe': 'test'})
                actual = load_file(str(output))
                self.assertEqual(set(actual), set(payload))
                for key in payload:
                    self.assertTrue(torch.equal(actual[key].reshape(-1).view(torch.uint8),
                                                payload[key].reshape(-1).view(torch.uint8)))
                with safe_open(str(output), framework='pt') as handle:
                    self.assertEqual(handle.metadata(), {'recipe': 'test'})

    def test_retained_values_are_owned_and_invalid_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            spool = Spool(root, lambda: None)
            value = torch.ones(2)
            spool.add({'value': value}); value.fill_(10)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                spool.add({'value': value})
            with self.assertRaisesRegex(ValueError, 'NaN/Inf'):
                spool.add({'nan': torch.tensor(float('nan'))})
            self.assertTrue(torch.equal(load_file(str(spool.finish({})))['value'], torch.ones(2)))

    def test_corrupt_finite_payload_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as root:
            spool = Spool(root, lambda: None)
            spool.add({'value': torch.ones(2)})
            def corrupted(tensors, filename, **kwargs):
                save_file({k: v + 1 for k, v in tensors.items()}, filename, **kwargs)
            with patch('extractor.writer.save_file', side_effect=corrupted):
                with self.assertRaisesRegex(ValueError, 'data verification failed'):
                    spool.finish({})

    def test_cancellation_before_save(self):
        with tempfile.TemporaryDirectory() as root:
            spool = Spool(root, lambda: None)
            spool.add({'value': torch.ones(2)})
            def cancel(): raise RuntimeError('cancelled')
            spool.cancel = cancel
            with self.assertRaisesRegex(RuntimeError, 'cancelled'): spool.finish({})
            self.assertEqual(list(Path(root).iterdir()), [])


if __name__ == '__main__': unittest.main()
