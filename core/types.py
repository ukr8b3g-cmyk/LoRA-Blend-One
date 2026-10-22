from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class FactorStyle:
    name: str
    down_suffix: str
    up_suffix: str


@dataclass(frozen=True)
class FactorComponent:
    down_key: str
    up_key: str
    scale: float
    row_start: int
    row_count: int
    columns: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class LayerSpec:
    target: str
    style: FactorStyle
    down_key: str
    up_key: str
    alpha_key: Optional[str]
    rank: int
    down_shape: tuple[int, ...]
    up_shape: tuple[int, ...]
    dtype_name: str
    effective_scale: float
    layout: str
    out_dim: int
    in_flat: int
    down_tail: tuple[int, ...]
    up_tail: tuple[int, ...]
    components: tuple[FactorComponent, ...] = ()


@dataclass
class SourceInfo:
    path: Path
    name: str
    layers: dict[str, LayerSpec]
    metadata: dict[str, str] = field(default_factory=dict)
    adapter_config: dict[str, Any] = field(default_factory=dict)
    unsupported: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    factor_dtypes: set[str] = field(default_factory=set)
    architecture_hint: str = "unknown"
    sha256: Optional[str] = None


@dataclass
class MergeSettings:
    method: str = "Weighted"
    compression_mode: str = "auto"
    output_rank: Optional[int] = None
    energy_target: float = 0.999
    auto_rank_cap: int = 256
    output_dtype: str = "auto"
    device: str = "auto"
    fallback_policy: str = "auto_cpu"
    unmatched_policy: str = "scaled_carry"
    density: float = 0.90
    sign_consensus: str = "auto"
    della_half_width: float = 0.05
    seed: int = 0
    output_scale: float = 1.0
    gate_conflict_threshold: float = 0.05
    krea2_gate_strength: float = 0.5
    krea2_opposite_strategy: str = "pick_strongest"
    krea2_opposite_threshold: float = 0.05
    krea2_gate_rank: int = 32
    krea2_balance_mode: str = "off"
    krea2_gate_compression: str = "attn_to_gate_only"
    krea2_normal_compression: str = "off"


@dataclass
class LayerReport:
    target: str
    contributors: int
    method: str
    device: str
    input_rank_sum: int
    max_input_rank: int
    output_rank: int
    captured_energy: Optional[float]
    exact: bool
    unmatched: bool
    gate_conflicts: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class MergeReport:
    method: str
    source_count: int
    layer_reports: list[LayerReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gate_targets: int = 0
    gate_conflicts: int = 0

    def rank_range(self) -> tuple[int, int]:
        ranks = [r.output_rank for r in self.layer_reports if r.output_rank > 0]
        return (min(ranks), max(ranks)) if ranks else (0, 0)
