from __future__ import annotations

import hashlib
import math
from typing import Iterable

import torch

from .types import LayerSpec


def flatten_factors(up: torch.Tensor, down: torch.Tensor, spec: LayerSpec, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    up = up.to(device=device, dtype=torch.float32)
    down = down.to(device=device, dtype=torch.float32)
    # Standard LoRA up factors are [out, rank] or [out, rank, 1, ...].
    # Reshape works for both without special-casing a 2D flatten start dimension.
    up2 = up.reshape(spec.out_dim, spec.rank)
    down2 = down.reshape(spec.rank, spec.in_flat)
    return up2.contiguous(), down2.contiguous()


def restore_factor_layout(up2: torch.Tensor, down2: torch.Tensor, template: LayerSpec) -> tuple[torch.Tensor, torch.Tensor]:
    rank = int(down2.shape[0])
    if template.layout == "linear":
        return up2.reshape(template.out_dim, rank), down2.reshape(rank, template.in_flat)
    up = up2.reshape(template.out_dim, rank, *([1] * len(template.up_tail)))
    down = down2.reshape(rank, *template.down_tail)
    return up, down


def reconstruct_delta(up2: torch.Tensor, down2: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return (up2 @ down2).mul_(float(scale))


def choose_energy_rank(s: torch.Tensor, energy_target: float, hard_cap: int | None = None, *, total_energy: torch.Tensor | None = None) -> tuple[int, float]:
    if s.numel() == 0:
        return 1, 1.0
    e = s.float().square()
    denom = float(total_energy.item()) if total_energy is not None else float(e.sum().item())
    if denom <= 1e-30:
        return 1, 1.0
    cum = torch.cumsum(e, dim=0) / denom
    threshold = torch.tensor(float(energy_target), device=cum.device, dtype=cum.dtype)
    r = int(torch.searchsorted(cum, threshold).item()) + 1
    r = max(1, min(r, s.numel()))
    if hard_cap is not None:
        r = min(r, int(hard_cap))
    captured = float(e[:r].sum().item() / denom)
    return r, min(max(captured, 0.0), 1.0)


def exact_core_svd(
    ups: list[torch.Tensor],
    downs: list[torch.Tensor],
    scales: list[float],
    *,
    rank_mode: str,
    fixed_rank: int | None,
    energy_target: float,
    rank_cap: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, float, bool]:
    if not ups:
        raise ValueError("No factors to merge")
    b = torch.cat([u * float(sc) for u, sc in zip(ups, scales)], dim=1)
    a = torch.cat(downs, dim=0)
    full_concat_rank = int(a.shape[0])

    if rank_mode == "none":
        return b.contiguous(), a.contiguous(), full_concat_rank, 1.0, True

    qb, rb = torch.linalg.qr(b, mode="reduced")
    qa, ra = torch.linalg.qr(a.T, mode="reduced")
    core = rb @ ra.T
    u, s, vh = torch.linalg.svd(core, full_matrices=False)
    total_energy = s.float().square().sum()
    max_rank = int(s.numel())

    if float(total_energy.item()) <= 1e-30:
        up = torch.zeros((b.shape[0], 1), dtype=torch.float32, device=b.device)
        down = torch.zeros((1, a.shape[1]), dtype=torch.float32, device=b.device)
        return up, down, 1, 1.0, False

    if rank_mode == "energy":
        rank, captured = choose_energy_rank(s, energy_target, hard_cap=rank_cap)
    elif rank_mode == "fixed":
        if fixed_rank is None:
            raise ValueError("fixed rank requested without output rank")
        rank = max(1, min(int(fixed_rank), max_rank))
        captured = float(s[:rank].float().square().sum().item() / total_energy.item())
    else:
        raise ValueError(f"Unknown rank mode: {rank_mode}")

    root = s[:rank].clamp_min(0).sqrt()
    up = (qb @ u[:, :rank]) * root.unsqueeze(0)
    down = root.unsqueeze(1) * (vh[:rank, :] @ qa.T)
    exact = rank >= max_rank
    return up.contiguous(), down.contiguous(), rank, min(max(captured, 0.0), 1.0), exact


def stable_seed(base_seed: int, *parts: object) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(base_seed)).encode("utf-8"))
    for part in parts:
        h.update(b"\0")
        h.update(str(part).encode("utf-8", "replace"))
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFFFFFFFFFF


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    gen.manual_seed(int(seed))
    return gen


def _hist_keep_threshold(values_abs: torch.Tensor, keep_fraction: float, bins: int = 8192) -> float:
    n = values_abs.numel()
    keep = max(0, min(n, int(round(float(keep_fraction) * n))))
    if keep <= 0:
        return float(values_abs.max().item()) + 1.0
    if keep >= n:
        return 0.0
    hi = float(values_abs.max().item())
    if hi <= 0:
        return 0.0
    hist = torch.histc(values_abs.float(), bins=bins, min=0.0, max=hi)
    csum = torch.cumsum(hist.flip(0), dim=0)
    idx = int(torch.searchsorted(csum, torch.tensor(float(keep), device=csum.device)).item())
    idx = min(max(idx, 0), bins - 1)
    return hi * (1.0 - (idx + 1) / bins)


def ties_sparsify(delta: torch.Tensor, density: float) -> torch.Tensor:
    if density >= 1.0:
        return delta
    if density <= 0.0:
        return torch.zeros_like(delta)
    abs_delta = delta.abs()
    # Exact selection for modest tensors, histogram threshold for large layers.
    if delta.numel() <= 2_000_000:
        n = delta.numel()
        keep = max(1, min(n, int(round(density * n))))
        kth = n - keep + 1
        tau = float(torch.kthvalue(abs_delta.flatten().float(), kth).values.item())
    else:
        tau = _hist_keep_threshold(abs_delta, density)
    delta.mul_(abs_delta >= tau)
    return delta


def dare_sparsify(delta: torch.Tensor, density: float, *, generator: torch.Generator, chunk_rows: int = 256) -> torch.Tensor:
    if density >= 1.0:
        return delta
    if not 0.0 < density <= 1.0:
        raise ValueError("DARE density must be in (0, 1]")
    rows = delta.shape[0]
    inv = 1.0 / float(density)
    for lo in range(0, rows, chunk_rows):
        hi = min(rows, lo + chunk_rows)
        blk = delta[lo:hi]
        rnd = torch.rand(blk.shape, device=blk.device, generator=generator, dtype=torch.float32)
        blk.mul_(rnd < density).mul_(inv)
    return delta


def della_sparsify(
    delta: torch.Tensor,
    density: float,
    half_width: float,
    *,
    generator: torch.Generator,
    chunk_rows: int = 128,
) -> torch.Tensor:
    if density >= 1.0 and half_width == 0.0:
        return delta
    lo_p = float(density) - float(half_width)
    hi_p = float(density) + float(half_width)
    if not (0.0 < lo_p <= hi_p <= 1.0):
        raise ValueError("DELLA requires 0 < density-half_width <= density+half_width <= 1")
    rows, cols = delta.shape
    if cols <= 1:
        p = float(density)
        for lo in range(0, rows, chunk_rows):
            hi = min(rows, lo + chunk_rows)
            blk = delta[lo:hi]
            rnd = torch.rand(blk.shape, device=blk.device, generator=generator, dtype=torch.float32)
            blk.mul_(rnd < p).div_(p)
        return delta

    pos = torch.arange(cols, device=delta.device, dtype=torch.int32).unsqueeze(0)
    denom = float(cols - 1)
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        blk = delta[start:end]
        idx = torch.argsort(blk.abs(), dim=1, descending=False)
        ranks = torch.empty((end - start, cols), device=delta.device, dtype=torch.int32)
        ranks.scatter_(1, idx, pos.expand(end - start, cols))
        probs = lo_p + (ranks.float() / denom) * (2.0 * half_width)
        rnd = torch.rand(probs.shape, device=delta.device, generator=generator, dtype=torch.float32)
        keep = rnd < probs
        blk.mul_(keep).div_(probs)
    return delta


def randomized_svd_factors(
    matrix: torch.Tensor,
    *,
    rank_mode: str,
    fixed_rank: int | None,
    energy_target: float,
    auto_rank_cap: int,
    seed: int,
    oversample: int = 8,
    niter: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    if matrix.dim() != 2:
        raise ValueError("randomized_svd_factors expects a 2D matrix")
    m, n = matrix.shape
    min_dim = min(m, n)
    if min_dim < 1:
        raise ValueError("Cannot factor an empty matrix")
    total_energy = matrix.float().square().sum()
    if float(total_energy.item()) <= 1e-30:
        return (
            torch.zeros((m, 1), device=matrix.device, dtype=torch.float32),
            torch.zeros((1, n), device=matrix.device, dtype=torch.float32),
            1,
            1.0,
        )

    if rank_mode == "fixed":
        if fixed_rank is None:
            raise ValueError("fixed rank requested without output rank")
        target_cap = max(1, min(int(fixed_rank), min_dim))
    elif rank_mode == "energy":
        target_cap = max(1, min(int(auto_rank_cap), min_dim))
    else:
        raise ValueError(f"Unsupported rank mode for dense merge: {rank_mode}")

    q = min(min_dim, target_cap + max(0, int(oversample)))
    gen = make_generator(matrix.device, seed)
    omega = torch.randn((n, q), device=matrix.device, dtype=torch.float32, generator=gen)
    y = matrix @ omega
    for _ in range(max(0, int(niter))):
        # Re-orthogonalize between power iterations for numerical stability.
        qy = torch.linalg.qr(y, mode="reduced").Q
        y = matrix @ (matrix.T @ qy)
    qmat = torch.linalg.qr(y, mode="reduced").Q
    small = qmat.T @ matrix
    u_small, s, vh = torch.linalg.svd(small, full_matrices=False)
    u = qmat @ u_small

    if rank_mode == "energy":
        rank, captured = choose_energy_rank(s, energy_target, hard_cap=target_cap, total_energy=total_energy)
    else:
        rank = min(target_cap, s.numel())
        captured = float(s[:rank].float().square().sum().item() / total_energy.item())

    root = s[:rank].clamp_min(0).sqrt()
    up = u[:, :rank] * root.unsqueeze(0)
    down = root.unsqueeze(1) * vh[:rank]
    return up.contiguous(), down.contiguous(), rank, min(max(captured, 0.0), 1.0)


def factor_fro_dot(up_a: torch.Tensor, down_a: torch.Tensor, up_b: torch.Tensor, down_b: torch.Tensor) -> torch.Tensor:
    return ((up_a.T @ up_b) * (down_a @ down_b.T)).sum()


def factor_fro_norm(up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(factor_fro_dot(up, down, up, down), min=0.0))


def gate_conflict_count(factors: Iterable[tuple[torch.Tensor, torch.Tensor]], threshold: float) -> int:
    fs = list(factors)
    count = 0
    for i in range(len(fs)):
        ua, da = fs[i]
        na = factor_fro_norm(ua, da)
        if float(na.item()) <= 1e-20:
            continue
        for j in range(i + 1, len(fs)):
            ub, db = fs[j]
            nb = factor_fro_norm(ub, db)
            if float(nb.item()) <= 1e-20:
                continue
            cos = factor_fro_dot(ua, da, ub, db) / (na * nb)
            if float(cos.item()) < -float(threshold):
                count += 1
    return count


def torch_dtype_from_name(name: str) -> torch.dtype:
    table = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return table[name.lower()]
    except KeyError:
        raise ValueError(f"Unsupported output dtype: {name}")
