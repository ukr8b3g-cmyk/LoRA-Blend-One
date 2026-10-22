from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("lbo_error_test")
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location("lbo_error_test.error_reporting", ROOT / "error_reporting.py")
reporting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporting)


class ErrorReportingTests(unittest.TestCase):
    def test_specific_file_is_matched_to_its_enabled_slot_with_both_languages(self):
        error = reporting.MergeError("moss_mixed", ["dmd2.safetensors"], "MOSS-Image mixed with SDXL")
        payload = reporting.describe_failure(error, [(1, r"moss\a.safetensors"), (7, r"DMD2\dmd2.safetensors")])
        self.assertEqual(payload["sources"], [{"slot": 7, "name": r"DMD2\dmd2.safetensors"}])
        self.assertIn("MOSS-Image", payload["reason"]["en"])
        self.assertIn("混在", payload["reason"]["ja"])
        self.assertIn("OFF", payload["hint"]["en"])
        self.assertIn("OFF", payload["hint"]["ja"])

    def test_unknown_and_memory_errors_do_not_blame_an_arbitrary_lora(self):
        slots = [(1, "a.safetensors"), (2, "b.safetensors")]
        for error in (RuntimeError("SVD failed"), MemoryError("not enough memory")):
            payload = reporting.describe_failure(error, slots)
            self.assertEqual(payload["sources"], [])
            self.assertEqual(payload["detail"], str(error))
        self.assertEqual(reporting.describe_failure(MemoryError(), slots)["code"], "out_of_memory")

    def test_full_registered_name_is_preferred_over_another_files_basename(self):
        error = reporting.MergeError("read_failed", [r"A\adapter_model.safetensors"], "bad header")
        payload = reporting.describe_failure(error, [(1, r"A\adapter_model.safetensors"), (7, r"B\adapter_model.safetensors")])
        self.assertEqual(payload["sources"], [{"slot": 1, "name": r"A\adapter_model.safetensors"}])

    def test_event_is_scoped_to_the_executing_prompt_and_client(self):
        send = Mock()
        server = types.ModuleType("server")
        server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(client_id="chrome-client", send_sync=send))
        execution = types.ModuleType("comfy_execution")
        execution.__path__ = []
        utils = types.ModuleType("comfy_execution.utils")
        utils.get_executing_context = Mock(return_value=types.SimpleNamespace(node_id="5", prompt_id="prompt-a"))
        with patch.dict(sys.modules, {"server": server, "comfy_execution": execution, "comfy_execution.utils": utils}):
            error = reporting.MergeError("shape_mismatch", ["b.safetensors"], "incompatible shapes")
            reporting.publish_failure(error, [(2, "b.safetensors")])
            event, payload, client = send.call_args.args
            self.assertEqual((event, payload["node_id"], payload["prompt_id"], client),
                             ("lora_blend_one.error", "5", "prompt-a", "chrome-client"))
            self.assertEqual(payload["error"]["sources"], [{"slot": 2, "name": "b.safetensors"}])
            send.reset_mock()
            utils.get_executing_context.return_value = None
            reporting.publish_failure(error, [])
            send.assert_not_called()
            utils.get_executing_context.return_value = types.SimpleNamespace(node_id="5", prompt_id="prompt-a")
            server.PromptServer.instance.client_id = None
            reporting.publish_failure(error, [])
            send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
