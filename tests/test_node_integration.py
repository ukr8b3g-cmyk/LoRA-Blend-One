from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


class NodeIntegrationTests(unittest.TestCase):
    def test_package_import_and_end_to_end_node_save(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            loras = base / "loras"
            output = base / "output"
            loras.mkdir(); output.mkdir()
            target = "diffusion_model.layers.0.attention.to_q"
            for i in range(3):
                g = torch.Generator().manual_seed(500 + i)
                rank = 4
                save_file({
                    f"{target}.lora_A.weight": torch.randn(rank, 12, generator=g),
                    f"{target}.lora_B.weight": torch.randn(16, rank, generator=g),
                    f"{target}.alpha": torch.tensor(float(rank)),
                }, str(loras / f"l{i}.safetensors"))

            folder_paths = types.ModuleType("folder_paths")
            folder_paths.get_filename_list = lambda category: sorted(p.name for p in loras.glob("*.safetensors"))
            resolved_names = []
            def get_full_path_or_raise(category, name):
                resolved_names.append(name)
                p = loras / name
                if not p.is_file():
                    raise FileNotFoundError(name)
                return str(p)
            folder_paths.get_full_path_or_raise = get_full_path_or_raise
            folder_paths.get_output_directory = lambda: str(output)
            def get_save_image_path(prefix, out_dir):
                p = Path(prefix)
                subfolder = str(p.parent) if str(p.parent) != "." else ""
                folder = Path(out_dir) / subfolder
                folder.mkdir(parents=True, exist_ok=True)
                return str(folder), p.name, 1, subfolder, prefix
            folder_paths.get_save_image_path = get_save_image_path
            sys.modules["folder_paths"] = folder_paths

            comfy = types.ModuleType("comfy")
            mm = types.ModuleType("comfy.model_management")
            mm.throw_exception_if_processing_interrupted = lambda: None
            cu = types.ModuleType("comfy.utils")
            class ProgressBar:
                def __init__(self, total): self.total = total; self.value = 0
                def update(self, n): self.value += n
            cu.ProgressBar = ProgressBar
            comfy.model_management = mm
            comfy.utils = cu
            sys.modules["comfy"] = comfy
            sys.modules["comfy.model_management"] = mm
            sys.modules["comfy.utils"] = cu

            package_name = "lbo_integration_pkg"
            spec = importlib.util.spec_from_file_location(
                package_name,
                root / "__init__.py",
                submodule_search_locations=[str(root)],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
            self.assertEqual(module.WEB_DIRECTORY, "./web")
            self.assertTrue((root / "web" / "accordion.js").is_file())
            Node = module.NODE_CLASS_MAPPINGS["LoRABlendOne"]
            schema = Node.INPUT_TYPES()
            self.assertIn("tooltip", schema["required"]["lora_1"][1])
            self.assertTrue(schema["optional"]["merge_method"][1]["advanced"])

            result = Node().merge(
                "l0.safetensors", 20.0,
                "l1.safetensors", 30.0,
                "16",
                "loras/integration",
                lora_3="l2.safetensors",
                weight_3=50.0,
                merge_method="Weighted",
                weight_mode="normalize",
                compression_mode="fixed_rank",
                custom_rank=16,
                energy_target=0.999,
                auto_rank_cap=64,
                density=0.90,
                sign_consensus="auto",
                della_half_width=0.05,
                output_scale=1.0,
                output_dtype="fp16",
                device="cpu",
                fallback_policy="auto_cpu",
                unmatched_policy="scaled_carry",
                compatibility_mode="safe",
                seed=123,
                gate_conflict_threshold=0.05,
                hash_inputs=True,
                write_report=True,
            )
            self.assertIn("Saved", result["result"][0])
            self.assertEqual(result["ui"]["model_detection"], ["Standard LoRA"])
            out = output / "loras" / "integration_00001_.safetensors"
            self.assertTrue(out.exists())
            self.assertTrue(out.with_suffix(".safetensors.report.json").exists())
            with safe_open(str(out), framework="pt", device="cpu") as f:
                meta = f.metadata() or {}
                self.assertEqual(meta.get("lbo_method"), "Weighted")
                self.assertEqual(f.get_slice(f"{target}.lora_A.weight").get_dtype(), "F16")
                self.assertEqual(f.get_slice(f"{target}.lora_A.weight").get_shape()[0], 12)  # true merged rank <= sum input ranks

            Sources = module.NODE_CLASS_MAPPINGS["LoRABlendSources"]
            Blend = module.NODE_CLASS_MAPPINGS["LoRABlendOneSplit"]
            source_schema = Sources.INPUT_TYPES()
            blend_schema = Blend.INPUT_TYPES()
            self.assertEqual(set(source_schema["required"]), {"lora_1", "weight_1", "lora_2", "weight_2"})
            self.assertEqual(set(blend_schema["required"]), {"sources", "output_rank", "filename_prefix"})
            self.assertTrue(source_schema["optional"]["lora_3"][1]["advanced"])
            for key in ("merge_method", "weight_mode", "compression_mode", "output_dtype"):
                self.assertNotIn("advanced", blend_schema["optional"][key][1])
            self.assertEqual(blend_schema["optional"]["seed"][1]["default"], 0)
            self.assertTrue(blend_schema["optional"]["seed"][1]["advanced"])
            self.assertFalse(blend_schema["optional"]["write_report"][1]["default"])
            self.assertTrue(blend_schema["optional"]["write_report"][1]["advanced"])
            self.assertNotIn("weight_mode", source_schema["optional"])
            enabled_keys = [f"enabled_{idx}" for idx in range(1, 9)]
            old_source_keys = [key for idx in range(3, 9) for key in (f"lora_{idx}", f"weight_{idx}")]
            self.assertEqual(list(schema["optional"])[-8:], enabled_keys)
            self.assertEqual(list(source_schema["optional"]), old_source_keys + enabled_keys)
            for key in enabled_keys:
                self.assertEqual(source_schema["optional"][key], schema["optional"][key])
                self.assertEqual(schema["optional"][key][0], "BOOLEAN")
                self.assertTrue(schema["optional"][key][1]["default"])
                self.assertNotIn(key, blend_schema["optional"])

            sources = Sources().select(
                "l0.safetensors", 20.0, "l1.safetensors", 30.0,
                lora_3="l2.safetensors", weight_3=50.0,
            )[0]
            split_result = Blend().merge(
                sources, "16", "loras/split",
                merge_method="Weighted", weight_mode="normalize",
                compression_mode="fixed_rank", output_dtype="fp16", device="cpu",
            )
            self.assertIn("Saved", split_result["result"][0])
            split_out = output / "loras" / "split_00001_.safetensors"
            self.assertTrue(split_out.exists())
            self.assertFalse(split_out.with_suffix(".safetensors.report.json").exists())
            original_open = Path.open
            def report_failure(path, *args, **kwargs):
                if '.report.json' in path.name:
                    raise OSError('injected report failure')
                return original_open(path, *args, **kwargs)
            with patch.object(Path, 'open', report_failure):
                warning_result = Blend().merge(
                    sources, "16", "loras/report_warning", device="cpu", write_report=True,
                )
            self.assertIn('Saved', warning_result['result'][0])
            self.assertIn('warnings=1', warning_result['result'][0])
            self.assertIn('report could not be saved', warning_result['ui']['text'][0])
            self.assertIn('injected report failure', warning_result['ui']['text'][0])
            self.assertTrue((output / 'loras' / 'report_warning_00001_.safetensors').is_file())
            with safe_open(str(out), framework="pt", device="cpu") as old_file, safe_open(str(split_out), framework="pt", device="cpu") as split_file:
                self.assertEqual(set(old_file.keys()), set(split_file.keys()))
                for key in old_file.keys():
                    self.assertTrue(torch.equal(old_file.get_tensor(key), split_file.get_tensor(key)), key)

            krea_sources = Sources().select("l0.safetensors", 20.0, "l1.safetensors", 30.0)[0]
            Blend().merge(krea_sources, "Auto", "loras/krea2",
                          merge_method="Krea2 Gate-Aware", device="cpu", hash_inputs=False)
            with safe_open(str(output / "loras" / "krea2_00001_.safetensors"), framework="pt", device="cpu") as f:
                self.assertEqual(f.metadata()["lbo_method"], "Krea2 Gate-Aware")
                recipe = json.loads(f.metadata()["lbo_recipe_json"])
                self.assertEqual([x["weight"] for x in recipe["sources"]], [0.2, 0.3])

            moss_target = "diffusion_model.layers.0.proj.o"
            save_file({moss_target + ".lora_A.weight": torch.ones(1, 3),
                       moss_target + ".lora_B.weight": torch.ones(4, 1)},
                      str(loras / "moss.safetensors"), metadata={"ss_base_model_version": "moss_image"})
            moss = Blend().merge(((1, "moss.safetensors", 100.),), "Auto", "loras/moss_status",
                                 device="cpu", hash_inputs=False)
            self.assertEqual(moss["ui"]["model_detection"], ["MOSS-Image"])
            self.assertIn("model=MOSS-Image", moss["result"][0])
            legacy = Node().merge("moss.safetensors", 100., "(none)", 0., "Auto", "loras/moss_legacy_status",
                                  device="cpu", hash_inputs=False)
            self.assertEqual(legacy["ui"]["model_detection"], ["MOSS-Image"])

            selection = dict(
                lora_1="moss.safetensors", weight_1=100.,
                lora_2="missing.safetensors", weight_2=35., enabled_2=False,
                lora_3="l0.safetensors", weight_3=25., enabled_3=False,
                lora_8="also_missing.safetensors", weight_8=70., enabled_8=False,
            )
            unchanged_selection = dict(selection)
            self.assertIs(Sources.VALIDATE_INPUTS(**selection), True)
            self.assertIn("LoRA 2", Sources.VALIDATE_INPUTS(**(selection | {"enabled_2": True})))
            self.assertIn("unavailable", Sources.VALIDATE_INPUTS(**(selection | {"enabled_2": True})))
            selected = Sources().select(**selection)[0]
            self.assertEqual(selection, unchanged_selection)
            self.assertEqual(selected, ((1, "moss.safetensors", 100.),))
            resolved_names.clear()
            toggled = Blend().merge(selected, "Auto", "loras/toggled",
                                    device="cpu", hash_inputs=False)
            self.assertEqual(toggled["ui"]["model_detection"], ["MOSS-Image"])
            self.assertEqual(resolved_names, ["moss.safetensors"])
            with safe_open(str(output / "loras" / "toggled_00001_.safetensors"), framework="pt", device="cpu") as f:
                recipe = json.loads(f.metadata()["lbo_recipe_json"])
                self.assertEqual(recipe["summary"]["source_count"], 1)
                self.assertEqual([x["name"] for x in recipe["sources"]], ["moss.safetensors"])
                self.assertEqual([x["weight"] for x in recipe["sources"]], [1.])

            resolved_names.clear()
            legacy_toggled = Node().merge(**selection, output_rank="Auto", filename_prefix="loras/legacy_toggled",
                                          device="cpu", hash_inputs=False, write_report=False)
            self.assertEqual(legacy_toggled["ui"]["model_detection"], ["MOSS-Image"])
            self.assertEqual(resolved_names, ["moss.safetensors"])
            with safe_open(str(output / "loras" / "legacy_toggled_00001_.safetensors"), framework="pt", device="cpu") as f:
                recipe = json.loads(f.metadata()["lbo_recipe_json"])
                self.assertEqual(recipe["summary"]["source_count"], 1)
            for method in ("Auto", "Weighted", "TIES", "DARE", "DELLA"):
                resolved_names.clear()
                result = Node().merge("moss.safetensors", 100., "l0.safetensors", 0., "16", "loras/zero_" + method,
                                      merge_method=method, device="cpu", hash_inputs=False, write_report=False)
                self.assertEqual(result['ui']['model_detection'], ['MOSS-Image'])
                self.assertEqual(resolved_names, ['moss.safetensors'])
            resolved_names.clear()
            self.assertIs(Sources.VALIDATE_INPUTS(lora_1='moss.safetensors', weight_1=100.,
                                                lora_2='missing.safetensors', weight_2=0.), True)
            resolved_names.clear()
            zero_sources = Sources().select('moss.safetensors', 100., 'missing.safetensors', 0.)[0]
            Blend().merge(zero_sources, '16', 'loras/zero_missing', device='cpu', hash_inputs=False)
            self.assertEqual(resolved_names, ['moss.safetensors'])
            with self.assertRaises(ValueError) as caught:
                Node().merge('moss.safetensors', 0., 'l0.safetensors', 0., '16', 'loras/all_zero', device='cpu')
            self.assertEqual(caught.exception.code, 'invalid_weights')
            for mode in ('normalize', 'direct'):
                resolved_names.clear()
                Node().merge('l0.safetensors', 20., 'l0.safetensors', -20., '16', 'loras/cancel_' + mode,
                             lora_3='moss.safetensors', weight_3=100., weight_mode=mode,
                             device='cpu', hash_inputs=False)
                self.assertEqual(resolved_names, ['moss.safetensors'])
            Node().merge('l0.safetensors', 0., 'l1.safetensors', 100., '16', 'loras/krea_zero',
                         merge_method='Krea2 Gate-Aware', device='cpu', hash_inputs=False)
            with safe_open(str(output / 'loras' / 'krea_zero_00001_.safetensors'), framework='pt') as f:
                recipe = json.loads(f.metadata()['lbo_recipe_json'])
                self.assertEqual([x['weight'] for x in recipe['sources']], [0., 1.])

            all_off = dict(lora_1="moss.safetensors", weight_1=100., lora_2="l0.safetensors", weight_2=50.,
                           enabled_1=False, enabled_2=False)
            with self.assertRaisesRegex(ValueError, "at least one enabled LoRA"):
                Sources().select(**all_off)
            with self.assertRaisesRegex(ValueError, "at least one enabled LoRA"):
                Node().merge(**all_off, output_rank="Auto", filename_prefix="loras/all_off", device="cpu")

            krea_toggled_sources = Sources().select(
                "l0.safetensors", 20., "missing.safetensors", 40., enabled_2=False,
                lora_3="l1.safetensors", weight_3=30.,
            )[0]
            self.assertEqual([row[0] for row in krea_toggled_sources], [1, 3])
            resolved_names.clear()
            Blend().merge(krea_toggled_sources, "Auto", "loras/krea2_toggled",
                          merge_method="Krea2 Gate-Aware", device="cpu", hash_inputs=False)
            self.assertEqual(resolved_names, ["l0.safetensors", "l1.safetensors"])
            with safe_open(str(output / "loras" / "krea2_toggled_00001_.safetensors"), framework="pt", device="cpu") as f:
                recipe = json.loads(f.metadata()["lbo_recipe_json"])
                self.assertEqual(recipe["summary"]["source_count"], 2)
                self.assertEqual([x["name"] for x in recipe["sources"]], ["l0.safetensors", "l1.safetensors"])
                self.assertEqual([x["weight"] for x in recipe["sources"]], [0.2, 0.3])
            for invalid_count in (krea_toggled_sources[:1], krea_toggled_sources + ((4, "l2.safetensors", 10.),)):
                with self.assertRaisesRegex(ValueError, "exactly two selected LoRA slots"):
                    Blend().merge(invalid_count, "Auto", "loras/krea2_invalid_count",
                                  merge_method="Krea2 Gate-Aware", device="cpu", hash_inputs=False)

            # Failures stay failures, while the UI gets the originating slot/file.
            with patch.object(module.nodes, "publish_failure") as notify:
                with self.assertRaisesRegex(ValueError, "MOSS-Image LoRAs") as caught:
                    Blend().merge(((1, "moss.safetensors", 50.), (7, "l0.safetensors", 50.)),
                                  "32", "loras/failure", device="cpu", hash_inputs=False)
                self.assertEqual(caught.exception.code, "moss_mixed")
                self.assertEqual(notify.call_args.args[1], [(1, "moss.safetensors"), (7, "l0.safetensors")])
                self.assertFalse((output / "loras" / "failure_00001_.safetensors").exists())
            for directory, original in (("MOSS", "moss.safetensors"), ("Other", "l0.safetensors")):
                (loras / directory).mkdir()
                (loras / directory / "adapter_model.safetensors").write_bytes((loras / original).read_bytes())
            with patch.object(module.nodes, "publish_failure") as notify:
                with self.assertRaises(ValueError) as caught:
                    Blend().merge(((1, "MOSS/adapter_model.safetensors", 50.),
                                   (7, "Other/adapter_model.safetensors", 50.)),
                                  "32", "loras/same_basename", device="cpu", hash_inputs=False)
                self.assertEqual(caught.exception.sources, ["Other/adapter_model.safetensors"])
                reporting = sys.modules[package_name + ".error_reporting"]
                payload = reporting.describe_failure(caught.exception, notify.call_args.args[1])
                self.assertEqual(payload["sources"], [{"slot": 7, "name": "Other/adapter_model.safetensors"}])
            with patch.object(module.nodes, "publish_failure") as notify:
                with self.assertRaises(ValueError) as caught:
                    Node().merge("missing.safetensors", 100., "(none)", 0., "32", "loras/missing",
                                 device="cpu", hash_inputs=False)
                self.assertEqual(caught.exception.code, "missing_file")
                self.assertEqual(caught.exception.sources, ["missing.safetensors"])
                notify.assert_called_once()
            with patch.object(module.nodes, "publish_failure") as notify:
                with self.assertRaises(ValueError) as caught:
                    Sources().select(**all_off)
                self.assertEqual(caught.exception.code, "no_sources")
                notify.assert_called_once()
            with patch.object(module.nodes, "publish_failure") as notify, \
                 patch.object(module.nodes, "save_verified", side_effect=PermissionError("output is read-only")):
                with self.assertRaises(ValueError) as caught:
                    Blend().merge(((1, "moss.safetensors", 100.),), "32", "loras/denied",
                                  device="cpu", hash_inputs=False)
                self.assertEqual(caught.exception.code, "save_failed")
                self.assertEqual(caught.exception.sources, [])
                notify.assert_called_once()
            class Interrupted(Exception):
                pass
            with patch.object(mm, "InterruptProcessingException", Interrupted, create=True), \
                 patch.object(module.nodes, "merge_sources", side_effect=Interrupted()), \
                 patch.object(module.nodes, "publish_failure") as notify:
                with self.assertRaises(Interrupted):
                    Blend().merge(((1, "moss.safetensors", 100.),), "32", "loras/cancelled",
                                  device="cpu", hash_inputs=False)
                notify.assert_not_called()
            with patch.object(module.nodes, "scan_lora", side_effect=MemoryError("not enough RAM")), \
                 patch.object(module.nodes, "publish_failure") as notify:
                with self.assertRaises(MemoryError):
                    Blend().merge(((1, "moss.safetensors", 100.),), "32", "loras/read_oom",
                                  device="cpu", hash_inputs=False)
                self.assertIsInstance(notify.call_args.args[0], MemoryError)
            with patch.object(mm, "InterruptProcessingException", Interrupted, create=True), \
                 patch.object(module.nodes, "scan_lora", side_effect=Interrupted()), \
                 patch.object(module.nodes, "publish_failure") as notify:
                with self.assertRaises(Interrupted):
                    Blend().merge(((1, "moss.safetensors", 100.),), "32", "loras/read_cancelled",
                                  device="cpu", hash_inputs=False)
                notify.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
