import hashlib
import json
import os
import struct
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch


def finite(tensor, label):
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f'{label}: NaN/Inf detected; check input/strength or use FP32/BF16')


def publish(temp, final):
    if os.name == 'nt':
        os.rename(temp, final)
    else:
        os.link(temp, final)
        Path(temp).unlink()


class Spool:
    def __init__(self, folder, cancel):
        self.folder = Path(folder)
        self.cancel = cancel
        self.entries = {}
        self.parts = []
        self.size = 0

    def add(self, tensors):
        path = self.folder / f'{len(self.parts)}.safetensors'
        for key, tensor in tensors.items():
            finite(tensor, key)
        save_file(tensors, str(path))
        with path.open('rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(length))
        for key, entry in header.items():
            if key in self.entries:
                raise ValueError(f'Duplicate output key: {key}')
            lo, hi = entry['data_offsets']
            self.entries[key] = {'dtype': entry['dtype'], 'shape': entry['shape'], 'data_offsets': [self.size+lo, self.size+hi]}
        payload_size = path.stat().st_size - 8 - length
        self.parts.append((path, 8+length))
        self.size += payload_size

    def finish(self, metadata):
        header = dict(self.entries, __metadata__=metadata)
        raw = json.dumps(header, separators=(',', ':')).encode()
        raw += b' ' * (-len(raw) % 8)
        output = self.folder / 'assembled.partial'
        expected = hashlib.sha256()
        with output.open('xb') as out:
            out.write(struct.pack('<Q', len(raw))); out.write(raw)
            for path, offset in self.parts:
                with path.open('rb') as f:
                    f.seek(offset)
                    while block := f.read(8*1024*1024):
                        self.cancel(); out.write(block); expected.update(block)
        actual = hashlib.sha256()
        with output.open('rb') as f:
            f.seek(8+len(raw))
            while block := f.read(8*1024*1024):
                self.cancel(); actual.update(block)
        if expected.digest() != actual.digest():
            raise ValueError('Output payload verification failed')
        with safe_open(str(output), framework='pt', device='cpu') as handle:
            if set(handle.keys()) != set(self.entries) or handle.metadata() != metadata:
                raise ValueError('Output header verification failed')
            for key, entry in self.entries.items():
                self.cancel()
                if list(handle.get_slice(key).get_shape()) != entry['shape'] or handle.get_slice(key).get_dtype() != entry['dtype']:
                    raise ValueError(f'Output shape/dtype mismatch: {key}')
                finite(handle.get_tensor(key), key)
        return output
