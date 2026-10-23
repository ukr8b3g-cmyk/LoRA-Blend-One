"""CPU real-model extraction; keeps results and never starts ComfyUI."""
import sys
import argparse
import json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.engine import extract

if __name__ == '__main__':
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--modified',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    output, report = extract(args.base, args.modified,
                            args.output_dir/'MOSS_delta_rank16', rank=16, device='cpu', non_lora='Store Diff',
                            hash_inputs=True, progress=lambda stage,key,done,total: print(stage,done,total,key,flush=True)
                            if stage != 'Hash' else None)
    print(json.dumps({'path': str(output), 'type': report['artifact_type'], 'layers': len(report['layers']),
                      'relative_error': report['relative_error_all_parameters'], 'seconds': report['seconds']}, indent=2), flush=True)
