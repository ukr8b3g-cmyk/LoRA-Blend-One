from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from .types import MergeReport, MergeSettings, SourceInfo


RECIPE_VERSION = "1.0"
ENGINE_VERSION = "0.1.0"


def build_recipe(infos: list[SourceInfo], weights: list[float], settings: MergeSettings, report: MergeReport) -> dict:
    recorded_settings = asdict(settings)
    if report.method != "Krea2 Gate-Aware":
        recorded_settings = {key: value for key, value in recorded_settings.items() if not key.startswith("krea2_")}
    return {
        "recipe_version": RECIPE_VERSION,
        "engine_version": ENGINE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "name": info.name,
                "sha256": info.sha256,
                "architecture_hint": info.architecture_hint,
                "weight": float(weight),
                "layer_count": len(info.layers),
            }
            for info, weight in zip(infos, weights)
        ],
        "settings": recorded_settings,
        "resolved_method": report.method,
        "summary": {
            "source_count": report.source_count,
            "layer_count": len(report.layer_reports),
            "gate_targets": report.gate_targets,
            "gate_conflicts": report.gate_conflicts,
            "rank_range": list(report.rank_range()),
            "warnings": report.warnings,
        },
    }


def build_metadata(recipe: dict) -> dict[str, str]:
    settings = recipe["settings"]
    metadata = {
        "lbo_recipe_version": str(recipe["recipe_version"]),
        "lbo_engine_version": str(recipe["engine_version"]),
        "lbo_method": str(recipe["resolved_method"]),
        "lbo_output_dtype": str(settings.get("output_dtype", "auto")),
        "lbo_compression": str(settings.get("compression_mode", "auto")),
        "lbo_energy_target": str(settings.get("energy_target", "")),
        "lbo_seed": str(settings.get("seed", "")),
        "lbo_recipe_json": json.dumps(recipe, ensure_ascii=False, separators=(",", ":")),
    }
    if recipe["sources"] and all(source["architecture_hint"] == "moss_image" for source in recipe["sources"]):
        metadata.update({"ss_base_model_version": "moss_image", "modelspec.architecture": "moss/lora",
                         "lbo_model_family": "moss_image", "lbo_moss_layout": "split_qkv"})
    return metadata


def report_to_dict(report: MergeReport) -> dict:
    return {
        "method": report.method,
        "source_count": report.source_count,
        "warnings": report.warnings,
        "gate_targets": report.gate_targets,
        "gate_conflicts": report.gate_conflicts,
        "rank_range": list(report.rank_range()),
        "layers": [
            {
                "target": x.target,
                "contributors": x.contributors,
                "method": x.method,
                "device": x.device,
                "input_rank_sum": x.input_rank_sum,
                "max_input_rank": x.max_input_rank,
                "output_rank": x.output_rank,
                "captured_energy": x.captured_energy,
                "exact": x.exact,
                "unmatched": x.unmatched,
                "gate_conflicts": x.gate_conflicts,
                "warnings": x.warnings,
            }
            for x in report.layer_reports
        ],
    }
