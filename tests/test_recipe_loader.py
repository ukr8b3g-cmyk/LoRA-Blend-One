from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("lbo_recipe_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
loader = importlib.import_module("lbo_recipe_test.recipe_loader")


class RecipeLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.files = {}
        self.folders = types.ModuleType("folder_paths")
        self.folders.get_filename_list = lambda _: list(self.files)
        self.folders.get_full_path_or_raise = lambda _, name: str(self.files[name])
        with patch.dict(sys.modules, {"folder_paths": self.folders}):
            self.node = importlib.import_module("lbo_recipe_test.nodes").LoRABlendOne
        self.source("moss/a.safetensors", 1)
        self.source("moss/b.safetensors", 2)

    def tearDown(self):
        self.tmp.cleanup()

    def source(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"dummy": torch.tensor([float(value)])}, path)
        self.files[name] = path
        return loader.sha256_file(path)

    def recipe(self, *, settings=None, sources=None, **extra):
        data = {"recipe_version": "1.0", "resolved_method": "Weighted",
                "settings": {"output_rank": 16, "output_dtype": "bf16", "seed": 17, **(settings or {})},
                "sources": sources if sources is not None else [
                    {"name": "a.safetensors", "weight": 0.8, "sha256": loader.sha256_file(self.files["moss/a.safetensors"])},
                    {"name": "b.safetensors", "weight": -0.4, "sha256": loader.sha256_file(self.files["moss/b.safetensors"])},
                ], **extra}
        path = self.root / "blend.safetensors"
        save_file({"dummy": torch.zeros(1)}, path, metadata={"lbo_recipe_json": json.dumps(data)})
        self.files["blend.safetensors"] = path
        return data

    def inspect(self, name="blend.safetensors"):
        with patch.dict(sys.modules, {"folder_paths": self.folders}):
            return loader.inspect_recipe(name, list(self.files), self.files.__getitem__, self.node.INPUT_TYPES())

    def test_old_recipe_restores_coefficients_in_direct_mode_with_all_settings(self):
        self.recipe(settings={"output_rank": 7, "output_scale": 1.2, "density": 0.6, "device": "cpu"})
        result = self.inspect()
        self.assertTrue(result["ready"])
        self.assertEqual([s["selected"] for s in result["sources"]], ["moss/a.safetensors", "moss/b.safetensors"])
        self.assertEqual([s["weight"] * 100 for s in result["sources"]], [80., -40.])
        self.assertEqual(result["settings"]["output_rank"], "Custom")
        self.assertEqual(result["settings"]["custom_rank"], 7)
        self.assertEqual(result["settings"]["weight_mode"], "direct")
        self.assertEqual(result["settings"]["output_scale"], 1.2)
        self.assertEqual(result["settings"]["seed"], 17)

    def test_same_basename_is_resolved_by_hash_even_when_recorded_path_changed(self):
        expected = self.source("other/a.safetensors", 99)
        self.recipe(sources=[{"name": "a.safetensors", "registered_name": "moss/a.safetensors", "weight": .5, "sha256": expected}])
        result = self.inspect()
        self.assertTrue(result["ready"])
        self.assertEqual(result["sources"][0]["selected"], "other/a.safetensors")

    def test_missing_mismatching_and_ambiguous_sources_block_restore(self):
        self.source("other/a.safetensors", 99)
        for entry, status in [
            ({"name": "gone.safetensors", "weight": 1}, "missing"),
            ({"name": "a.safetensors", "weight": 1, "sha256": "0" * 64}, "hash_mismatch"),
            ({"name": "a.safetensors", "weight": 1}, "ambiguous"),
        ]:
            with self.subTest(status=status):
                self.recipe(sources=[entry])
                result = self.inspect()
                self.assertFalse(result["ready"])
                self.assertEqual(result["sources"][0]["status"], status)
        self.recipe(sources=[{"name": "b.safetensors", "weight": 1}])
        self.assertEqual(self.inspect()["sources"][0]["status"], "unverified")

    def test_bad_recipes_and_unrepresentable_values_are_not_silently_restored(self):
        for extra in [{"recipe_version": "2.0"}, {"sources": []},
                      {"sources": [{"name": "a.safetensors", "weight": float("nan")}]},
                      {"settings": {"output_rank": 0}}]:
            self.recipe(**extra)
            with self.assertRaises(loader.RecipeError): self.inspect()
        self.recipe(settings={"seed": -1})
        self.assertFalse(self.inspect()["ready"])
        self.recipe(sources=[{"name": "b.safetensors", "weight": 4.1}])
        self.assertFalse(self.inspect()["ready"])
        self.recipe()
        with self.assertRaisesRegex(loader.RecipeError, "no LoRA Blend One recipe"):
            self.inspect("moss/a.safetensors")

    def test_only_registered_names_can_be_opened(self):
        resolve = unittest.mock.Mock()
        for name in ["../outside.safetensors", str(self.files["moss/a.safetensors"]), None, {}]:
            with self.assertRaises(loader.RecipeError):
                loader.inspect_recipe(name, list(self.files), resolve, {})
        resolve.assert_not_called()

    def test_krea2_specific_settings_survive_restore(self):
        self.recipe(resolved_method="Krea2 Gate-Aware", settings={"output_rank": None, "krea2_gate_strength": .3,
                    "krea2_gate_rank": 24, "krea2_normal_compression": "max_input_rank"})
        result = self.inspect()
        self.assertTrue(result["ready"])
        self.assertEqual(result["settings"]["output_rank"], "Auto")
        self.assertEqual(result["settings"]["krea2_gate_strength"], .3)

    def test_dropped_metadata_needs_no_registered_blend_file(self):
        recipe = self.recipe()
        del self.files["blend.safetensors"]
        with patch.dict(sys.modules, {"folder_paths": self.folders}):
            result = loader.inspect_recipe_data(recipe, list(self.files), self.files.__getitem__,
                                                self.node.INPUT_TYPES(), "from_downloads.safetensors")
        self.assertTrue(result["ready"])
        self.assertEqual(result["file"], "from_downloads.safetensors")
        self.assertEqual([row["weight"] for row in result["sources"]], [.8, -.4])

    def test_dropped_metadata_cannot_open_paths_outside_registered_sources(self):
        recipe = self.recipe(sources=[{"name": "../../outside.safetensors", "registered_name": "../../outside.safetensors", "weight": 1}])
        resolve = unittest.mock.Mock(side_effect=self.files.__getitem__)
        with patch.dict(sys.modules, {"folder_paths": self.folders}):
            result = loader.inspect_recipe_data(recipe, list(self.files), resolve, self.node.INPUT_TYPES(), "test.safetensors")
        self.assertFalse(result["ready"])
        resolve.assert_not_called()
        recipe["unused"] = "x" * (256 * 1024)
        with self.assertRaisesRegex(loader.RecipeError, "too large"):
            loader.inspect_recipe_data(recipe, list(self.files), resolve, {}, "test.safetensors")

    def test_local_http_routes_read_recipe_and_reject_unknown_file(self):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer
        routes = web.RouteTableDef()
        server = types.ModuleType("server")
        server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes))
        recipe = self.recipe()

        async def check():
            application = web.Application()
            application.add_routes(routes)
            async with TestClient(TestServer(application)) as client:
                response = await client.get("/lora_blend_one/recipe_files")
                self.assertIn("blend.safetensors", await response.json())
                response = await client.post("/lora_blend_one/recipe", json={"filename": "blend.safetensors"})
                self.assertEqual(response.status, 200)
                self.assertTrue((await response.json())["ready"])
                response = await client.post("/lora_blend_one/recipe", json={"filename": "../outside"})
                self.assertEqual(response.status, 400)
                self.assertEqual((await response.json())["code"], "unavailable_file")
                response = await client.post("/lora_blend_one/recipe_from_metadata", json={
                    "filename": r"Downloads\outside_model_folder.safetensors", "recipe": recipe})
                self.assertEqual(response.status, 200)
                result = await response.json()
                self.assertTrue(result["ready"])
                self.assertEqual(result["file"], "outside_model_folder.safetensors")
                response = await client.post("/lora_blend_one/recipe_from_metadata", json={"filename": "invalid.safetensors", "recipe": []})
                self.assertEqual(response.status, 400)
        with patch.dict(sys.modules, {"server": server, "folder_paths": self.folders}):
            loader.register_routes()
            asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
