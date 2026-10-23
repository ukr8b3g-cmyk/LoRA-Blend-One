import sys
import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extractor.engine import extract
from extractor.model_files import make_plan


class ExtractorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.base = self.root/'base.safetensors'
        self.modified = self.root/'modified.safetensors'
        self.key = 'layers.0.proj.qkv.weight'
        self.a = {self.key: torch.zeros(6, 2), 'layers.0.norm1.weight': torch.ones(2)}
        self.b = {self.key: torch.arange(12, dtype=torch.float32).reshape(6,2)/10, 'layers.0.norm1.weight': torch.ones(2)}
        self.write()

    def tearDown(self):
        self.tmp.cleanup()

    def write(self):
        save_file(self.a,str(self.base)); save_file(self.b,str(self.modified))

    def run_extract(self, **kw):
        return extract(self.base,self.modified,self.root/'out',device='cpu',hash_inputs=False,**kw)

    def test_all_modes_roundtrip(self):
        for mode in ('Fixed Rank','Auto Rank','Full Delta'):
            path, report = self.run_extract(mode=mode, rank=2, max_rank=2, energy=.999, dtype='fp32')
            tensors = load_file(str(path))
            self.assertLess(report['relative_error_all_parameters'], .032 if mode == 'Auto Rank' else 1e-5)
            self.assertEqual(path.with_suffix('.safetensors.report.json').is_file(), True)
            self.assertFalse(any('partial' in p.name for p in self.root.iterdir()))
            self.assertTrue(all(torch.isfinite(t).all() for t in tensors.values()))

    def test_nonmatrix_strict_hybrid_partial(self):
        self.b['layers.0.norm1.weight'] += .1; self.write()
        with self.assertRaisesRegex(ValueError, 'non-LoRA'):
            self.run_extract()
        self.assertEqual(list(self.root.glob('out*')), [])
        path, report = self.run_extract(non_lora='Store Diff')
        self.assertEqual(report['artifact_type'], 'hybrid_delta')
        self.assertIn('diffusion_model.layers.0.norm1.diff',load_file(str(path)))
        _, report = self.run_extract(non_lora='Exclude With Report')
        self.assertEqual(len(report['excluded']),1)

    def test_same_model_no_output(self):
        self.b = self.a; self.write()
        path, report = self.run_extract()
        self.assertIsNone(path)
        self.assertEqual(report['status'],'No difference')

    def test_bad_inputs_and_aliases(self):
        self.b['extra.weight'] = torch.ones(1); self.write()
        with self.assertRaises(ValueError): make_plan(self.base,self.modified)
        del self.b['extra.weight']; self.b[self.key][0,0] = float('nan'); self.write()
        with self.assertRaisesRegex(ValueError,'NaN'): self.run_extract()
        self.b[self.key] = torch.zeros(6,3); self.write()
        with self.assertRaisesRegex(ValueError,'shapes'): self.run_extract()

    def test_no_overwrite_and_cancellation_cleanup(self):
        path,_ = self.run_extract(); original=path.read_bytes()
        second,_=self.run_extract()
        self.assertNotEqual(path,second); self.assertEqual(path.read_bytes(),original)
        class Stop(BaseException): pass
        with self.assertRaises(Stop):
            self.run_extract(cancel=lambda: (_ for _ in ()).throw(Stop()))
        self.assertFalse(any(p.name.startswith('.lbo_extract_') for p in self.root.iterdir()))

    def test_quantized_rejected(self):
        self.b[self.key] = torch.ones(6,2,dtype=torch.int8); self.write()
        with self.assertRaisesRegex(ValueError,'quantized'): self.run_extract()

    def test_storage_overflow_does_not_publish(self):
        self.b[self.key].fill_(100000.);self.write()
        with self.assertRaisesRegex(ValueError,'NaN/Inf'):
            self.run_extract(mode='Full Delta',dtype='fp16')
        self.assertEqual(list(self.root.glob('out*')),[])

    def test_duplicate_alias_rejected(self):
        self.b['diffusion_model.'+self.key]=self.b[self.key].clone();self.write()
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            self.run_extract()

    def test_report_failure_preserves_success(self):
        from unittest.mock import patch
        original=Path.write_text
        def fail(path,*args,**kwargs):
            if path.name=='report.partial': raise OSError('test report error')
            return original(path,*args,**kwargs)
        with patch.object(Path,'write_text',fail):
            path,report=self.run_extract()
        self.assertTrue(path.is_file());self.assertIn('report failed',report['warnings'][0])

    def test_node_schema_selection_save_and_path_boundary(self):
        import importlib.util
        import types
        from unittest.mock import patch
        root=Path(__file__).resolve().parents[1]
        package=types.ModuleType('delta_node_test');package.__path__=[str(root)]
        folders=types.ModuleType('folder_paths')
        folders.get_filename_list=lambda category:['base.safetensors','modified.safetensors']
        folders.get_full_path_or_raise=lambda category,name:str(self.root/name)
        folders.get_output_directory=lambda:str(self.root/'output')
        comfy=types.ModuleType('comfy');management=types.ModuleType('comfy.model_management');utils=types.ModuleType('comfy.utils')
        management.throw_exception_if_processing_interrupted=lambda:None
        utils.ProgressBar=lambda total:types.SimpleNamespace(update_absolute=lambda *args:None)
        comfy.model_management=management;comfy.utils=utils
        with patch.dict(sys.modules,{'delta_node_test':package,'folder_paths':folders,'comfy':comfy,
                         'comfy.model_management':management,'comfy.utils':utils,'server':None}):
            spec=importlib.util.spec_from_file_location('delta_node_test.extract_nodes',root/'extract_nodes.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            base='diffusion_models: base.safetensors';modified='diffusion_models: modified.safetensors'
            info=module.LoRABlendModelInspect().run(base,modified)
            self.assertIn('MOSS-Image',info['result'][0])
            result=module.LoRABlendModelExtract().run(base,modified,'Fixed Rank',2,.99,'fp32','loras/test',device='cpu',hash_inputs=False)
            self.assertIn('standard_lora',result['result'][0])
            with self.assertRaises(ValueError):
                module.LoRABlendModelExtract().run(base,modified,'Fixed Rank',2,.99,'fp32','../escape')

if __name__ == '__main__': unittest.main()
