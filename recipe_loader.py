"""Read saved recipes and resolve their sources without loading tensor data."""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import asdict
from pathlib import Path, PureWindowsPath

from safetensors import SafetensorError, safe_open

from .core.formats import sha256_file
from .core.types import MergeSettings


class RecipeError(ValueError):
    def __init__(self, message, code="invalid_recipe"):
        super().__init__(message)
        self.code = code


def inspect_recipe(filename, names, resolve_path, schema):
    if not isinstance(filename, str) or filename not in names:
        raise RecipeError("Choose an installed LoRA file.", "unavailable_file")
    try:
        with safe_open(str(resolve_path(filename)), framework="pt", device="cpu") as handle:
            raw = (handle.metadata() or {}).get("lbo_recipe_json")
        if not raw:
            raise RecipeError("This file has no LoRA Blend One recipe.", "no_recipe")
        recipe = json.loads(raw)
    except RecipeError:
        raise
    except (OSError, ValueError, SafetensorError) as error:
        raise RecipeError(str(error)) from error
    return inspect_recipe_data(recipe, names, resolve_path, schema, filename)


def inspect_recipe_data(recipe, names, resolve_path, schema, filename):
    """Resolve a file recipe or dropped metadata using only registered source paths."""
    try:
        encoded = json.dumps(recipe, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise RecipeError(str(error)) from error
    if len(encoded.encode("utf-8")) > 256 * 1024:
        raise RecipeError("The recipe is too large.")
    if not isinstance(recipe, dict) or recipe.get("recipe_version") != "1.0":
        raise RecipeError("Unsupported recipe version.", "unsupported_version")
    sources, settings = recipe.get("sources"), recipe.get("settings")
    if not isinstance(sources, list) or not 1 <= len(sources) <= 8 or not isinstance(settings, dict):
        raise RecipeError("The recipe must contain settings and 1–8 sources.")

    resolved = []
    hashes = {}
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("name"), str):
            raise RecipeError("Invalid source name in the recipe.")
        weight = source.get("weight")
        if type(weight) not in (int, float) or not math.isfinite(weight):
            raise RecipeError("Invalid source coefficient in the recipe.")
        expected = source.get("sha256")
        if expected is not None and (not isinstance(expected, str) or len(expected) != 64
                                     or any(c not in "0123456789abcdefABCDEF" for c in expected)):
            raise RecipeError("Invalid source SHA256 in the recipe.")
        registered = source.get("registered_name", source["name"])
        candidates = [name for name in names if PureWindowsPath(name).name == PureWindowsPath(source["name"]).name]
        # A recorded relative path is still verified by hash; identical basenames
        # in other folders remain candidates if that path is no longer present.
        if registered in names:
            candidates = [registered] + [name for name in candidates if name != registered]
        matches, unreadable = [], False
        for candidate in candidates:
            if not expected:
                matches.append(candidate)
                continue
            try:
                if candidate not in hashes:
                    hashes[candidate] = sha256_file(resolve_path(candidate))
                if hashes[candidate].lower() == expected.lower():
                    matches.append(candidate)
            except OSError:
                unreadable = True
        selected = registered if registered in matches else matches[0] if len(matches) == 1 else None
        status = ("matched" if expected else "unverified") if selected else (
            "ambiguous" if matches else "unreadable" if unreadable else "hash_mismatch" if candidates else "missing")
        resolved.append({"name": source["name"], "weight": float(weight), "sha256": expected,
                         "selected": selected, "status": status, "candidates": candidates})

    specs = {**schema["required"], **schema["optional"]}
    values = asdict(MergeSettings())
    values.update({key: value for key, value in settings.items() if key in values})
    rank = values.pop("output_rank")
    if rank is not None and (type(rank) is not int or not 1 <= rank <= 2048):
        raise RecipeError("The output rank cannot be restored by this node.")
    restored = {key: value for key, value in values.items() if key in specs}
    restored["merge_method"] = recipe.get("resolved_method", values["method"])
    restored["output_rank"] = "Auto" if rank is None else str(rank) if str(rank) in specs["output_rank"][0] else "Custom"
    restored["custom_rank"] = rank if rank is not None else 32
    restored["weight_mode"] = "direct"
    restored["compatibility_mode"] = settings.get("compatibility_mode", "safe")
    issues = []
    for key, value in restored.items():
        kind, options = specs[key]
        if isinstance(kind, list):
            valid = value in kind
        else:
            valid = type(value) in (int, float) and math.isfinite(value) and (
                kind != "INT" or type(value) is int) and options.get("min", -math.inf) <= value <= options.get("max", math.inf)
        if not valid:
            issues.append(f"{key}: {value}")
    if any(abs(source["weight"] * 100) > specs["weight_1"][1]["max"] for source in resolved):
        issues.append("Source coefficient exceeds the node's weight range.")
    return {"file": filename, "created_utc": recipe.get("created_utc"), "sources": resolved,
            "settings": restored, "issues": issues,
            "ready": not issues and all(source["selected"] for source in resolved)}


def register_routes():
    try:
        from server import PromptServer
    except ImportError:
        return
    if PromptServer.instance is None:
        return
    import folder_paths
    from aiohttp import web
    from .nodes import LoRABlendOne

    @PromptServer.instance.routes.get("/lora_blend_one/recipe_files")
    async def recipe_files(request):
        return web.json_response(folder_paths.get_filename_list("loras"))

    @PromptServer.instance.routes.post("/lora_blend_one/recipe")
    async def recipe_detail(request):
        try:
            body = await request.json()
            filename = body.get("filename") if isinstance(body, dict) else None
            result = await asyncio.to_thread(
                inspect_recipe, filename, list(folder_paths.get_filename_list("loras")),
                lambda name: Path(folder_paths.get_full_path_or_raise("loras", name)),
                LoRABlendOne.INPUT_TYPES())
            return web.json_response(result)
        except (RecipeError, OSError, ValueError) as error:
            code = error.code if isinstance(error, RecipeError) else "read_failed"
            return web.json_response({"error": str(error), "code": code}, status=400)

    @PromptServer.instance.routes.post("/lora_blend_one/recipe_from_metadata")
    async def dropped_recipe(request):
        try:
            body = await request.json()
            if not isinstance(body, dict) or not isinstance(body.get("filename"), str):
                raise RecipeError("Invalid dropped recipe.")
            result = await asyncio.to_thread(
                inspect_recipe_data, body.get("recipe"), list(folder_paths.get_filename_list("loras")),
                lambda name: Path(folder_paths.get_full_path_or_raise("loras", name)),
                LoRABlendOne.INPUT_TYPES(), PureWindowsPath(body["filename"]).name)
            return web.json_response(result)
        except (RecipeError, OSError, ValueError) as error:
            code = error.code if isinstance(error, RecipeError) else "read_failed"
            return web.json_response({"error": str(error), "code": code}, status=400)
