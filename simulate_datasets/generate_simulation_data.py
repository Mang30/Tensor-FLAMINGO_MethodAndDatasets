#!/usr/bin/env python3
"""
Generate Tensor-FLAMINGO-style simulation distance matrices.

Pipeline:
  consensus 3D coordinates -> dense GT distance/contact matrices
  -> downsample observed entries Omega -> delta = min(distance on Omega)
  -> add Gaussian noise only on Omega -> sparse distance/contact matrices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


STRUCTURE_PARAMS = {
    "n_beads": 500,
    "n_consensus": 3,
    "n_cells_per_consensus": 10,
    "coord_generation_method": "sarw",
    "seed": 42,
}

SIMILARITY_PARAMS = {
    "W_values": [0.6, 0.7, 0.8],
}

NOISE_PARAMS = {
    "noise_levels": ["level_0", "level_1", "level_2"],
    "delta_calculation": "min_downsampled_nonzero_distance",
}

DOWNSAMPLING_PARAMS = {
    "retention_rates": [0.005],
    "strategy": "random_uniform",
    "preserve_symmetry": True,
    "zero_diagonal": True,
}

OUTPUT_PARAMS = {
    "output_base_dir": "./simulation_generated",
    "benchmark_dir": "benchmark_consensus_structure",
    "gt_distance_dir": "gt_distance_matrices",
    "gt_contact_dir": "gt_contact_matrices",
    "downsampled_dir": "downsampled_data",
    "downsampled_contact_dir": "downsampled_contact",
    "consensus_filename": "consensus_{idx}.txt",
    "gt_distance_filename": "consensus_{idx}_distance.txt",
    "gt_contact_filename": "consensus_{idx}_contact.txt",
    "sparse_filename": "consensus_{c_idx}_slice_{s_idx}.txt",
    "matrix_format": "dense_txt",
    "delimiter": "\t",
    "precision": "%.6f",
    "alpha": 0.25,
    "overwrite": True,
}

NOISE_LEVELS = {
    "level_0": (0.0, 0.0),
    "level_1": (1.0, 1.0),
    "level_2": (2.0, 1.0),
}


def _merge_params(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    params = deepcopy(defaults)
    if overrides:
        params.update(overrides)
    return params


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    coords = coords - coords.min(axis=0)
    span = coords.max(axis=0) - coords.min(axis=0)
    span[span == 0] = 1.0
    return coords / span


def _validate_square_matrix(matrix: np.ndarray, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")


def generate_consensus_structure(
    n_beads: int = 500,
    method: str = "sarw",
    seed: int | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Generate one consensus 3D structure.

    Supported methods:
      - sarw: self-avoiding random walk with rejection of near-collisions
      - chromatin_polymer: smooth random walk polymer surrogate
      - load_from_file: load coordinates from ``coord_path``
    """
    if n_beads <= 1:
        raise ValueError("n_beads must be greater than 1")

    if method == "load_from_file":
        coord_path = kwargs.get("coord_path")
        if not coord_path:
            raise ValueError("coord_path is required when method='load_from_file'")
        coords = np.loadtxt(coord_path)
        if coords.shape != (n_beads, 3):
            raise ValueError(f"Loaded coords shape {coords.shape} does not match ({n_beads}, 3)")
        return _normalize_coords(coords)

    rng = np.random.default_rng(seed)
    if method == "sarw":
        step_scale = float(kwargs.get("step_scale", 1.0))
        min_separation = float(kwargs.get("min_separation", 0.35 * step_scale))
        max_attempts = int(kwargs.get("max_attempts", 200))
        coords = np.zeros((n_beads, 3), dtype=float)
        occupied = [coords[0].copy()]
        for i in range(1, n_beads):
            accepted = None
            for _ in range(max_attempts):
                direction = rng.normal(size=3)
                norm = np.linalg.norm(direction)
                if norm == 0:
                    continue
                candidate = coords[i - 1] + step_scale * direction / norm
                recent = np.asarray(occupied[:-3] if len(occupied) > 3 else occupied)
                if recent.size == 0 or np.min(np.linalg.norm(recent - candidate, axis=1)) >= min_separation:
                    accepted = candidate
                    break
            if accepted is None:
                direction = rng.normal(size=3)
                accepted = coords[i - 1] + step_scale * direction / np.linalg.norm(direction)
            coords[i] = accepted
            occupied.append(accepted.copy())
        return _normalize_coords(coords)

    if method == "chromatin_polymer":
        step_scale = float(kwargs.get("step_scale", 1.0))
        persistence = float(kwargs.get("persistence", 0.65))
        coords = np.zeros((n_beads, 3), dtype=float)
        direction = rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)
        for i in range(1, n_beads):
            random_direction = rng.normal(size=3)
            random_direction = random_direction / np.linalg.norm(random_direction)
            direction = persistence * direction + (1.0 - persistence) * random_direction
            direction = direction / np.linalg.norm(direction)
            coords[i] = coords[i - 1] + step_scale * direction
        return _normalize_coords(coords)

    raise ValueError("method must be 'sarw', 'chromatin_polymer', or 'load_from_file'")


def generate_multiple_consensus_structures(
    n_consensus: int = 3,
    n_beads: int = 500,
    W_values: list[float] | None = None,
    base_seed: int = 42,
    method: str = "sarw",
) -> list[np.ndarray]:
    """Generate multiple consensus structures with W-controlled similarity."""
    if n_consensus < 1:
        raise ValueError("n_consensus must be at least 1")
    if W_values is None:
        W_values = [0.7]
    if not W_values:
        raise ValueError("W_values cannot be empty")
    for W in W_values:
        if not 0 < float(W) < 1:
            raise ValueError("Each W value must satisfy 0 < W < 1")

    base = generate_consensus_structure(n_beads=n_beads, method=method, seed=base_seed)
    structures = [base]
    for idx in range(1, n_consensus):
        W = float(W_values[min(idx - 1, len(W_values) - 1)])
        perturbation = generate_consensus_structure(
            n_beads=n_beads,
            method=method,
            seed=base_seed + 10_000 + idx,
        )
        structures.append(_normalize_coords(W * base + (1.0 - W) * perturbation))
    return structures


def coords_to_distance_matrix(coords: np.ndarray, normalize: bool = False) -> np.ndarray:
    """Compute a full pairwise Euclidean distance matrix from 3D coordinates."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_beads, 3)")
    try:
        from scipy.spatial.distance import cdist

        dist_matrix = cdist(coords, coords)
    except Exception:
        diff = coords[:, None, :] - coords[None, :, :]
        dist_matrix = np.sqrt((diff * diff).sum(axis=-1))
    np.fill_diagonal(dist_matrix, 0.0)
    if normalize:
        positive = dist_matrix[dist_matrix > 0]
        if positive.size:
            dist_matrix = dist_matrix / positive.max()
            np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix


def distance_to_contact_matrix(dist_matrix: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """
    Convert distances to IF/contact values with no normalization.

    contact = distance^(-1/alpha) for positive distances; missing/diagonal entries stay 0.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    _validate_square_matrix(dist_matrix, "dist_matrix")
    contact = np.zeros_like(dist_matrix, dtype=float)
    observed = dist_matrix > 0
    contact[observed] = dist_matrix[observed] ** (-1.0 / alpha)
    np.fill_diagonal(contact, 0.0)
    return contact


def coords_to_contact_matrix(coords: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """Compute the raw dense IF/contact matrix from 3D coordinates."""
    return distance_to_contact_matrix(coords_to_distance_matrix(coords), alpha=alpha)


def _sample_upper_triangle_mask(
    dist_matrix: np.ndarray,
    retention_rate: float,
    strategy: str,
    seed: int | None,
) -> np.ndarray:
    if not 0 < retention_rate < 1:
        raise ValueError("retention_rate must satisfy 0 < retention_rate < 1")
    if strategy != "random_uniform":
        raise NotImplementedError("Only strategy='random_uniform' is currently implemented")

    n = dist_matrix.shape[0]
    upper_i, upper_j = np.triu_indices(n, k=1)
    n_upper = upper_i.size
    target_total_nonzero = int(round(retention_rate * n * n))
    target_upper = max(1, min(n_upper, int(round(target_total_nonzero / 2.0))))
    rng = np.random.default_rng(seed)
    selected = rng.choice(n_upper, size=target_upper, replace=False)
    mask = np.zeros((n, n), dtype=bool)
    mask[upper_i[selected], upper_j[selected]] = True
    mask |= mask.T
    return mask


def downsample_distance_matrix(
    dist_matrix: np.ndarray,
    retention_rate: float = 0.005,
    strategy: str = "random_uniform",
    preserve_symmetry: bool = True,
    zero_diagonal: bool = True,
    seed: int | None = None,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Downsample a distance matrix to simulate sparse scHi-C observations."""
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    _validate_square_matrix(dist_matrix, "dist_matrix")

    if preserve_symmetry:
        mask = _sample_upper_triangle_mask(dist_matrix, retention_rate, strategy, seed)
    else:
        if strategy != "random_uniform":
            raise NotImplementedError("Only strategy='random_uniform' is currently implemented")
        rng = np.random.default_rng(seed)
        mask = rng.random(dist_matrix.shape) < retention_rate

    if zero_diagonal:
        np.fill_diagonal(mask, False)
    sparse_dist = np.where(mask, dist_matrix, 0.0)
    if preserve_symmetry:
        sparse_dist = np.maximum(sparse_dist, sparse_dist.T)
        mask = sparse_dist > 0
    if return_mask:
        return sparse_dist, mask
    return sparse_dist


def add_noise_to_observed_distances(
    dist_matrix: np.ndarray,
    observed_mask: np.ndarray,
    noise_level: str = "level_1",
    seed: int | None = None,
    delta: float | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """
    Add Gaussian noise only on observed Omega entries.

    Delta is computed from the downsampled observed distances when not provided.
    Output is a sparse distance matrix; non-observed entries remain 0.
    """
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    observed_mask = np.asarray(observed_mask, dtype=bool)
    _validate_square_matrix(dist_matrix, "dist_matrix")
    if observed_mask.shape != dist_matrix.shape:
        raise ValueError("observed_mask must have the same shape as dist_matrix")
    if noise_level not in NOISE_LEVELS:
        raise ValueError("noise_level must be one of level_0, level_1, level_2")

    positive_observed = observed_mask & (dist_matrix > 0)
    upper_observed = np.triu(positive_observed, k=1)
    observed_values = dist_matrix[upper_observed]
    if observed_values.size == 0:
        raise ValueError("observed_mask selects no positive distances")
    if delta is None:
        delta = float(observed_values.min())

    noisy = np.zeros_like(dist_matrix, dtype=float)
    noisy[upper_observed] = dist_matrix[upper_observed]
    mean_multiplier, std_multiplier = NOISE_LEVELS[noise_level]
    if noise_level != "level_0":
        rng = np.random.default_rng(seed)
        noise = rng.normal(
            loc=mean_multiplier * delta,
            scale=std_multiplier * delta,
            size=observed_values.shape,
        )
        noisy[upper_observed] = observed_values + noise
        noisy[upper_observed] = np.maximum(noisy[upper_observed], np.finfo(float).eps)

    noisy = noisy + noisy.T
    np.fill_diagonal(noisy, 0.0)
    info = {
        "noise_level": noise_level,
        "delta": float(delta),
        "n_observed_matrix_entries": int(np.count_nonzero(noisy)),
        "n_observed_upper_entries": int(np.count_nonzero(np.triu(noisy, k=1))),
    }
    if return_info:
        return noisy, info
    return noisy


def add_noise_to_distance_matrix(
    dist_matrix: np.ndarray,
    noise_level: str = "level_1",
    delta: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Compatibility helper that adds noise to all positive matrix entries."""
    observed_mask = np.asarray(dist_matrix) > 0
    return add_noise_to_observed_distances(
        dist_matrix,
        observed_mask,
        noise_level=noise_level,
        delta=delta,
        seed=seed,
        return_info=False,
    )


def _save_matrix(path: Path, matrix: np.ndarray, delimiter: str, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, delimiter=delimiter, fmt=fmt)


def _write_readme(path: Path, params: dict[str, Any], metadata_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Tensor-FLAMINGO-style Simulation Dataset",
        "",
        "This dataset was generated from consensus 3D coordinates using the pipeline:",
        "",
        "1. Generate consensus 3D coordinates.",
        "2. Compute dense GT Euclidean distance matrices.",
        "3. Compute raw dense IF/contact matrices as distance^(-1/alpha), with no normalization.",
        "4. Downsample Omega from the dense distance matrix.",
        "5. Compute delta as the minimum positive distance on Omega.",
        "6. Add Gaussian noise only to Omega entries.",
        "7. Save sparse distance and sparse raw IF/contact matrices.",
        "",
        "## Directories",
        "",
        "- `benchmark_consensus_structure/`: GT consensus 3D coordinates.",
        "- `gt_distance_matrices/`: dense GT distance matrices.",
        "- `gt_contact_matrices/`: dense raw GT IF/contact matrices.",
        "- `downsampled_data/`: sparse noisy distance matrices for imputation.",
        "- `downsampled_contact/`: sparse raw IF/contact matrices converted from sparse distances.",
        "- `metadata.csv`: per-sample generation metadata.",
        "- `params.json`: full generation parameters.",
        "",
        "## Summary",
        "",
        f"- n_consensus: {params['structure_params']['n_consensus']}",
        f"- n_beads: {params['structure_params']['n_beads']}",
        f"- n_samples: {len(metadata_rows)}",
        f"- alpha: {params['output_params']['alpha']}",
        "",
    ]
    path.write_text("\n".join(lines))


def generate_full_simulation_dataset(
    structure_params: dict[str, Any] | None = None,
    similarity_params: dict[str, Any] | None = None,
    noise_params: dict[str, Any] | None = None,
    downsampling_params: dict[str, Any] | None = None,
    output_params: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate a full Tensor-FLAMINGO-style simulation dataset."""
    sp = _merge_params(STRUCTURE_PARAMS, structure_params)
    sip = _merge_params(SIMILARITY_PARAMS, similarity_params)
    np_params = _merge_params(NOISE_PARAMS, noise_params)
    dp = _merge_params(DOWNSAMPLING_PARAMS, downsampling_params)
    op = _merge_params(OUTPUT_PARAMS, output_params)

    n_beads = int(sp["n_beads"])
    n_consensus = int(sp["n_consensus"])
    n_cells = int(sp["n_cells_per_consensus"])
    seed = int(sp.get("seed", 42))
    method = sp.get("coord_generation_method", "sarw")
    noise_levels = list(np_params["noise_levels"])
    retention_rates = list(dp["retention_rates"])
    alpha = float(op["alpha"])

    out_dir = Path(op["output_base_dir"])
    if out_dir.exists() and op.get("overwrite", True):
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = {
        "benchmark": out_dir / op["benchmark_dir"],
        "gt_distance": out_dir / op["gt_distance_dir"],
        "gt_contact": out_dir / op["gt_contact_dir"],
        "downsampled": out_dir / op["downsampled_dir"],
        "downsampled_contact": out_dir / op["downsampled_contact_dir"],
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    consensus_list = generate_multiple_consensus_structures(
        n_consensus=n_consensus,
        n_beads=n_beads,
        W_values=sip["W_values"],
        base_seed=seed,
        method=method,
    )

    files_generated: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    sample_index = 0
    for c_idx, coords in enumerate(consensus_list, start=1):
        coord_path = dirs["benchmark"] / op["consensus_filename"].format(idx=c_idx)
        _save_matrix(coord_path, coords, op["delimiter"], op["precision"])
        files_generated.append(str(coord_path))

        dense_dist = coords_to_distance_matrix(coords, normalize=False)
        dense_contact = distance_to_contact_matrix(dense_dist, alpha=alpha)
        dist_path = dirs["gt_distance"] / op["gt_distance_filename"].format(idx=c_idx)
        contact_path = dirs["gt_contact"] / op["gt_contact_filename"].format(idx=c_idx)
        _save_matrix(dist_path, dense_dist, op["delimiter"], op["precision"])
        _save_matrix(contact_path, dense_contact, op["delimiter"], op["precision"])
        files_generated.extend([str(dist_path), str(contact_path)])

        for s_idx in range(1, n_cells + 1):
            sample_index += 1
            noise_level = noise_levels[(sample_index - 1) % len(noise_levels)]
            retention_rate = float(retention_rates[(sample_index - 1) % len(retention_rates)])
            sample_seed = seed + c_idx * 100_000 + s_idx
            _, mask = downsample_distance_matrix(
                dense_dist,
                retention_rate=retention_rate,
                strategy=dp["strategy"],
                preserve_symmetry=bool(dp["preserve_symmetry"]),
                zero_diagonal=bool(dp["zero_diagonal"]),
                seed=sample_seed,
                return_mask=True,
            )
            sparse_dist, noise_info = add_noise_to_observed_distances(
                dense_dist,
                mask,
                noise_level=noise_level,
                seed=sample_seed + 1,
                return_info=True,
            )
            sparse_contact = distance_to_contact_matrix(sparse_dist, alpha=alpha)

            sparse_name = op["sparse_filename"].format(c_idx=c_idx, s_idx=s_idx)
            sparse_path = dirs["downsampled"] / sparse_name
            sparse_contact_path = dirs["downsampled_contact"] / sparse_name
            _save_matrix(sparse_path, sparse_dist, op["delimiter"], op["precision"])
            _save_matrix(sparse_contact_path, sparse_contact, op["delimiter"], op["precision"])
            files_generated.extend([str(sparse_path), str(sparse_contact_path)])

            observed_entries = int(np.count_nonzero(sparse_dist))
            total_entries = int(sparse_dist.size)
            metadata_rows.append(
                {
                    "sample_id": f"consensus_{c_idx}_slice_{s_idx}",
                    "consensus_id": c_idx,
                    "slice_id": s_idx,
                    "noise_level": noise_level,
                    "retention_rate": retention_rate,
                    "delta": noise_info["delta"],
                    "n_beads": n_beads,
                    "n_observed": observed_entries,
                    "observed_fraction": observed_entries / total_entries,
                    "missing_fraction": 1.0 - observed_entries / total_entries,
                    "sparse_distance_path": str(sparse_path),
                    "sparse_contact_path": str(sparse_contact_path),
                    "gt_distance_path": str(dist_path),
                    "gt_contact_path": str(contact_path),
                    "coords_path": str(coord_path),
                    "alpha": alpha,
                }
            )

    metadata_path = out_dir / "metadata.csv"
    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)
    files_generated.append(str(metadata_path))

    full_params = {
        "structure_params": sp,
        "similarity_params": sip,
        "noise_params": np_params,
        "downsampling_params": dp,
        "output_params": op,
    }
    params_path = out_dir / "params.json"
    params_path.write_text(json.dumps(full_params, indent=2, sort_keys=True))
    files_generated.append(str(params_path))

    readme_path = out_dir / "README.md"
    _write_readme(readme_path, full_params, metadata_rows)
    files_generated.append(str(readme_path))

    info = {
        "n_consensus": n_consensus,
        "n_beads": n_beads,
        "n_total_cells": n_consensus * n_cells,
        "n_noise_levels": len(noise_levels),
        "n_downsample_rates": len(retention_rates),
        "output_dir": str(out_dir),
        "files_generated": files_generated,
    }
    if verbose:
        print(json.dumps(info, indent=2))
    return info


def validate_generated_data(output_dir: str | Path) -> None:
    """Validate generated files against the expected dense/sparse matrix contract."""
    output_dir = Path(output_dir)
    params = json.loads((output_dir / "params.json").read_text())
    n_beads = int(params["structure_params"]["n_beads"])
    retention_rates = params["downsampling_params"]["retention_rates"]

    gt = np.loadtxt(output_dir / "benchmark_consensus_structure" / "consensus_1.txt")
    if gt.shape != (n_beads, 3):
        raise AssertionError(f"GT shape mismatch: {gt.shape}")

    dense_dist = np.loadtxt(output_dir / "gt_distance_matrices" / "consensus_1_distance.txt")
    dense_contact = np.loadtxt(output_dir / "gt_contact_matrices" / "consensus_1_contact.txt")
    sparse = np.loadtxt(output_dir / "downsampled_data" / "consensus_1_slice_1.txt")
    sparse_contact = np.loadtxt(output_dir / "downsampled_contact" / "consensus_1_slice_1.txt")

    for name, matrix in {
        "dense_dist": dense_dist,
        "dense_contact": dense_contact,
        "sparse": sparse,
        "sparse_contact": sparse_contact,
    }.items():
        if matrix.shape != (n_beads, n_beads):
            raise AssertionError(f"{name} shape mismatch: {matrix.shape}")
        if not np.allclose(matrix, matrix.T):
            raise AssertionError(f"{name} is not symmetric")
        if not np.allclose(np.diag(matrix), 0.0):
            raise AssertionError(f"{name} diagonal is not zero")
        if np.any(matrix < 0):
            raise AssertionError(f"{name} contains negative values")

    actual_retention = np.count_nonzero(sparse) / sparse.size
    expected = float(retention_rates[0])
    if abs(actual_retention - expected) > 0.001:
        raise AssertionError(f"Retention mismatch: actual {actual_retention:.6f}, expected {expected:.6f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./simulation_generated")
    parser.add_argument("--n_beads", type=int, default=500)
    parser.add_argument("--n_consensus", type=int, default=3)
    parser.add_argument("--n_cells_per_consensus", type=int, default=10)
    parser.add_argument("--retention_rate", type=float, default=0.005)
    parser.add_argument("--noise_levels", default="level_0,level_1,level_2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    info = generate_full_simulation_dataset(
        structure_params={
            "n_beads": args.n_beads,
            "n_consensus": args.n_consensus,
            "n_cells_per_consensus": args.n_cells_per_consensus,
            "seed": args.seed,
        },
        noise_params={"noise_levels": [x.strip() for x in args.noise_levels.split(",") if x.strip()]},
        downsampling_params={"retention_rates": [args.retention_rate]},
        output_params={"output_base_dir": args.out},
    )
    if args.validate:
        validate_generated_data(info["output_dir"])
        print("Validation passed.")


if __name__ == "__main__":
    main()
