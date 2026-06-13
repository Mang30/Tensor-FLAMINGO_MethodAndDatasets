#!/usr/bin/env python3
"""t-SVD tensor completion for Tensor-FLAMINGO simulation multicondition data."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyfftw
from scipy import sparse
from scipy.stats import spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
SCHIC_ROOT = BASE_DIR.parents[2]
DEFAULT_DATA_ROOT = (
    SCHIC_ROOT
    / "1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/simulation_multicondition"
)
DEFAULT_INPUT_ROOT = BASE_DIR / "input"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output"

LOGGER = logging.getLogger("flamingo_simulation_completion")


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    beads: int
    w: str
    level: int
    source_dir: str
    sim_npz: str
    true_npz: str


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def lower_triangle_size_to_n(feature_count: int) -> int:
    n_float = (1.0 + math.sqrt(1.0 + 8.0 * feature_count)) / 2.0
    n = int(round(n_float))
    if n * (n - 1) // 2 != feature_count:
        raise ValueError(f"Invalid lower-triangle feature count: {feature_count}")
    return n


def discover_datasets(data_root: Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for sim_npz in sorted(data_root.glob("beads_*/W_*/level_*/1_lower_tri_feature/npz/ALL/*_ALL_sim.npz")):
        true_npz = sim_npz.with_name(sim_npz.name.replace("_sim.npz", "_true.npz"))
        if not true_npz.exists():
            raise FileNotFoundError(f"Missing truth npz for {sim_npz}: {true_npz}")
        level_dir = sim_npz.parents[3]
        w_dir = level_dir.parent
        beads_dir = w_dir.parent
        beads = int(beads_dir.name.split("_", 1)[1])
        w = w_dir.name.split("_", 1)[1]
        level = int(level_dir.name.split("_", 1)[1])
        dataset_id = f"beads{beads}_W{w}_level{level}"
        records.append(
            DatasetRecord(
                dataset_id=dataset_id,
                beads=beads,
                w=w,
                level=level,
                source_dir=str(level_dir),
                sim_npz=str(sim_npz),
                true_npz=str(true_npz),
            )
        )
    return records


def write_manifest(records: list[DatasetRecord], manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()), delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_manifest(manifest: Path) -> list[DatasetRecord]:
    with manifest.open(newline="") as handle:
        return [
            DatasetRecord(
                dataset_id=row["dataset_id"],
                beads=int(row["beads"]),
                w=row["w"],
                level=int(row["level"]),
                source_dir=row["source_dir"],
                sim_npz=row["sim_npz"],
                true_npz=row["true_npz"],
            )
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def select_record(records: list[DatasetRecord], dataset: str | None, task_id: int | None) -> DatasetRecord:
    if dataset is not None:
        matches = [record for record in records if record.dataset_id == dataset]
        if not matches:
            raise ValueError(f"Dataset {dataset} not found in manifest")
        return matches[0]
    if task_id is None:
        raise ValueError("Either --dataset or --task-id is required")
    if task_id < 0 or task_id >= len(records):
        raise IndexError(f"Task id {task_id} outside manifest range 0-{len(records) - 1}")
    return records[task_id]


def load_feature_npz(path: Path) -> sparse.csr_matrix:
    matrix = sparse.load_npz(path).tocsr().astype(np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{path} is not a two-dimensional sparse matrix")
    matrix.data[~np.isfinite(matrix.data)] = 0.0
    matrix.data[matrix.data < 0] = 0.0
    matrix.eliminate_zeros()
    return matrix


def feature_matrix_to_tensor(features: sparse.spmatrix | np.ndarray, n_beads: int | None = None) -> np.ndarray:
    dense = features.toarray() if sparse.issparse(features) else np.asarray(features)
    if dense.ndim != 2:
        raise ValueError(f"Feature matrix must be 2D, got {dense.shape}")
    if n_beads is None:
        n_beads = lower_triangle_size_to_n(dense.shape[0])
    expected = n_beads * (n_beads - 1) // 2
    if dense.shape[0] != expected:
        raise ValueError(f"Feature matrix has {dense.shape[0]} rows, expected {expected} for {n_beads} beads")
    tril_i, tril_j = np.tril_indices(n_beads, k=-1)
    n_cells = dense.shape[1]
    tensor = np.zeros((n_cells, n_beads, n_beads), dtype=np.float64)
    for cell_idx in range(n_cells):
        values = dense[:, cell_idx]
        tensor[cell_idx, tril_i, tril_j] = values
        tensor[cell_idx, tril_j, tril_i] = values
    return tensor


def tensor_to_feature_matrix(tensor: np.ndarray) -> np.ndarray:
    if tensor.ndim != 3 or tensor.shape[1] != tensor.shape[2]:
        raise ValueError(f"Tensor must have shape cells x beads x beads, got {tensor.shape}")
    tril_i, tril_j = np.tril_indices(tensor.shape[1], k=-1)
    return tensor[:, tril_i, tril_j].T


def prepare_input(record: DatasetRecord, input_root: Path, force: bool = False) -> Path:
    input_dir = input_root / record.dataset_id
    raw_dir = input_dir / "highres_contact_maps_transformed_original"
    complete_marker = input_dir / ".complete"
    if complete_marker.exists() and not force:
        return input_dir

    sim_features = load_feature_npz(Path(record.sim_npz))
    true_features = load_feature_npz(Path(record.true_npz))
    if sim_features.shape != true_features.shape:
        raise ValueError(f"Shape mismatch for {record.dataset_id}: {sim_features.shape} vs {true_features.shape}")
    if record.beads != lower_triangle_size_to_n(sim_features.shape[0]):
        raise ValueError(f"Manifest bead count does not match feature count for {record.dataset_id}")

    observed = feature_matrix_to_tensor(sim_features, n_beads=record.beads)
    truth = feature_matrix_to_tensor(true_features, n_beads=record.beads)

    raw_dir.mkdir(parents=True, exist_ok=True)
    for cell_idx, matrix in enumerate(observed, start=1):
        np.savetxt(raw_dir / f"RawCount_Cell_{cell_idx:03d}.txt", matrix, fmt="%.10g", delimiter="\t")

    sparse.save_npz(input_dir / "observed_features.npz", sim_features)
    sparse.save_npz(input_dir / "truth_features.npz", true_features)
    np.save(input_dir / "observed_if_tensor.npy", observed)
    np.save(input_dir / "truth_if_tensor.npy", truth)
    write_cell_index(input_dir / "input_file_index.csv", observed.shape[0])
    with (input_dir / "metadata.json").open("w") as handle:
        json.dump(asdict(record) | {"n_cells": observed.shape[0], "n_beads": observed.shape[1]}, handle, indent=2)
    complete_marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    LOGGER.info("Prepared %s input at %s", record.dataset_id, input_dir)
    return input_dir


def write_cell_index(path: Path, n_cells: int) -> None:
    cell_types = ["T1"] * 10 + ["T2"] * 10 + ["T3"] * 10
    rows = []
    for idx in range(n_cells):
        rows.append(
            {
                "cell_idx": idx,
                "cell_number": idx + 1,
                "cell_type": cell_types[idx] if idx < len(cell_types) else "NA",
                "input_file": f"RawCount_Cell_{idx + 1:03d}.txt",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


class FFTWPlans:
    def __init__(self, shape: tuple[int, int, int], n_threads: int):
        self.fft_in = pyfftw.empty_aligned(shape, dtype="complex128")
        self.fft_out = pyfftw.empty_aligned(shape, dtype="complex128")
        self.ifft_in = pyfftw.empty_aligned(shape, dtype="complex128")
        self.ifft_out = pyfftw.empty_aligned(shape, dtype="complex128")
        self.fft = pyfftw.FFTW(
            self.fft_in,
            self.fft_out,
            axes=(0,),
            threads=n_threads,
            flags=("FFTW_ESTIMATE",),
        )
        self.ifft = pyfftw.FFTW(
            self.ifft_in,
            self.ifft_out,
            axes=(0,),
            direction="FFTW_BACKWARD",
            threads=n_threads,
            flags=("FFTW_ESTIMATE",),
        )

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        self.fft_in[:, :, :] = tensor
        return self.fft().copy()

    def backward(self, tensor: np.ndarray) -> np.ndarray:
        self.ifft_in[:, :, :] = tensor
        return self.ifft().copy()


def _serial_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.linalg.svd(matrix, hermitian=True)


def prox_tnn(y_tensor: np.ndarray, threshold: float, plans: FFTWPlans) -> np.ndarray:
    n_cells = y_tensor.shape[0]
    x_fft = np.zeros(y_tensor.shape, dtype=np.complex128)
    y_fft = plans.forward(y_tensor.astype(np.complex128, copy=False))

    svd_indices = [0] + list(range(1, (n_cells + 1) // 2))
    for i in svd_indices:
        u, singular_values, vh = _serial_svd(y_fft[i])
        rank = int(np.sum(singular_values > threshold))
        if rank:
            shrunk = singular_values[:rank] - threshold
            x_fft[i] = (u[:, :rank] * shrunk) @ vh[:rank, :]
        if i != 0:
            x_fft[n_cells - i] = x_fft[i].conjugate()

    if n_cells % 2 == 0:
        i = n_cells // 2
        u, singular_values, vh = _serial_svd(y_fft[i])
        rank = int(np.sum(singular_values > threshold))
        if rank:
            shrunk = singular_values[:rank] - threshold
            x_fft[i] = (u[:, :rank] * shrunk) @ vh[:rank, :]

    return plans.backward(x_fft)


def complete_tensor(
    observed: np.ndarray,
    max_iter: int,
    tol: float,
    mu: float,
    max_mu: float,
    rho: float,
    n_threads: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    x_tensor = np.zeros(observed.shape, dtype=np.complex128)
    omega = observed > 0
    x_tensor[omega] = observed[omega]
    e_tensor = np.zeros(observed.shape, dtype=np.complex128)
    y_tensor = np.zeros(observed.shape, dtype=np.complex128)
    plans = FFTWPlans(observed.shape, n_threads=n_threads)
    log_rows = []
    total_size = int(np.prod(observed.shape))

    for iteration in range(max_iter):
        start = time.time()
        prev_x = x_tensor.copy()
        prev_e = e_tensor.copy()
        x_tensor = prox_tnn(-e_tensor + observed + y_tensor / mu, 1.0 / mu, plans=plans)
        e_tensor = observed - x_tensor + y_tensor / mu
        e_tensor[omega] = 0.0
        residual = observed - x_tensor - e_tensor
        change_x = float(np.max(np.abs(prev_x.reshape(total_size) - x_tensor.reshape(total_size))))
        change_e = float(np.max(np.abs(prev_e.reshape(total_size) - e_tensor.reshape(total_size))))
        residual_max = float(np.max(np.abs(residual.reshape(total_size))))
        change = max(change_x, change_e, residual_max)
        residual_norm = float(np.linalg.norm(residual.reshape(total_size)))
        log_rows.append(
            {
                "iteration": iteration,
                "residual_norm": residual_norm,
                "change": change,
                "mu": mu,
                "seconds": time.time() - start,
            }
        )
        LOGGER.info(
            "Iter:%03d residual=%.6g change=%.6g mu=%.6g sec=%.2f",
            iteration,
            residual_norm,
            change,
            mu,
            log_rows[-1]["seconds"],
        )
        if change < tol:
            break
        y_tensor = y_tensor + mu * residual
        mu = min(rho * mu, max_mu)
    return postprocess_tensor(x_tensor), pd.DataFrame(log_rows)


def postprocess_tensor(completed: np.ndarray) -> np.ndarray:
    completed = np.real(completed).astype(np.float64, copy=False)
    completed[~np.isfinite(completed)] = 0.0
    completed[completed < 0] = 0.0
    for idx in range(completed.shape[0]):
        completed[idx] = np.maximum(completed[idx], completed[idx].T)
        np.fill_diagonal(completed[idx], 0.0)
    return completed


def if_to_pd(values: np.ndarray, threshold: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    pd_values = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values) & (values >= threshold)
    pd_values[mask] = np.power(values[mask], -0.25)
    return pd_values


def _safe_pcc(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    corr = spearmanr(x, y, nan_policy="omit").correlation
    return float(corr) if corr is not None else float("nan")


def _metrics_for_mask(prefix: str, completed: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(completed) & np.isfinite(truth)
    x = completed[valid].astype(np.float64)
    y = truth[valid].astype(np.float64)
    if x.size == 0:
        return {
            f"n_{prefix}": 0,
            f"pcc_{prefix}": float("nan"),
            f"spearman_{prefix}": float("nan"),
            f"mae_{prefix}": float("nan"),
            f"rmse_{prefix}": float("nan"),
            f"relative_error_{prefix}": float("nan"),
        }
    diff = x - y
    denom = np.linalg.norm(y)
    return {
        f"n_{prefix}": int(x.size),
        f"pcc_{prefix}": _safe_pcc(x, y),
        f"spearman_{prefix}": _safe_spearman(x, y),
        f"mae_{prefix}": float(np.mean(np.abs(diff))),
        f"rmse_{prefix}": float(np.sqrt(np.mean(diff**2))),
        f"relative_error_{prefix}": float(np.linalg.norm(diff) / denom) if denom > 0 else float("nan"),
    }


def completion_metrics(completed: np.ndarray, truth: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    truth_mask = np.isfinite(truth) & (truth > 0)
    observed_mask = np.isfinite(observed) & (observed > 0) & truth_mask
    missing_mask = (~observed_mask) & truth_mask
    metrics = {}
    metrics.update(_metrics_for_mask("all", completed, truth, truth_mask))
    metrics.update(_metrics_for_mask("observed", completed, truth, observed_mask))
    metrics.update(_metrics_for_mask("missing", completed, truth, missing_mask))
    return metrics


def evaluate_completion(completed: np.ndarray, truth: np.ndarray, observed: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_types = ["T1"] * 10 + ["T2"] * 10 + ["T3"] * 10
    completed_features = tensor_to_feature_matrix(completed)
    truth_features = tensor_to_feature_matrix(truth)
    observed_features = tensor_to_feature_matrix(observed)
    rows = []
    for cell_idx in range(completed.shape[0]):
        if_metrics = completion_metrics(
            completed_features[:, cell_idx],
            truth_features[:, cell_idx],
            observed_features[:, cell_idx],
        )
        pd_metrics = completion_metrics(
            if_to_pd(completed_features[:, cell_idx], threshold=1.0),
            if_to_pd(truth_features[:, cell_idx], threshold=1.0),
            observed_features[:, cell_idx],
        )
        row = {
            "cell_idx": cell_idx,
            "cell_number": cell_idx + 1,
            "cell_type": cell_types[cell_idx] if cell_idx < len(cell_types) else "NA",
        }
        row.update({f"if_{key}": value for key, value in if_metrics.items()})
        row.update({f"pd_{key}": value for key, value in pd_metrics.items()})
        rows.append(row)
    cell_df = pd.DataFrame(rows)
    summary_df = summarize_metrics(cell_df)
    return cell_df, summary_df


def summarize_metrics(cell_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [col for col in cell_df.columns if col not in {"cell_idx", "cell_number", "cell_type"}]
    rows = []
    for label, group in [("all_cells", cell_df), *cell_df.groupby("cell_type")]:
        row = {"group": label, "n_cells": int(len(group))}
        for col in numeric_cols:
            row[f"{col}_mean"] = float(group[col].mean(skipna=True))
            row[f"{col}_std"] = float(group[col].std(skipna=True))
        rows.append(row)
    return pd.DataFrame(rows)


def write_completion_outputs(
    output_dir: Path,
    completed: np.ndarray,
    truth: np.ndarray,
    observed: np.ndarray,
    log: pd.DataFrame,
    record: DatasetRecord,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = output_dir / "completed_matrices"
    flamingo_dir = output_dir / "high_res_contact_maps_FLAMINGO"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    flamingo_dir.mkdir(parents=True, exist_ok=True)

    completed_features = tensor_to_feature_matrix(completed)
    sparse.save_npz(output_dir / "completed_features.npz", sparse.csr_matrix(completed_features))
    np.save(output_dir / "completed_tensor.npy", completed)
    np.save(output_dir / "high_resolution.npy", completed)
    for idx, matrix in enumerate(completed, start=1):
        np.savetxt(matrix_dir / f"Completed_Cell_{idx:03d}.txt", matrix, fmt="%.6g", delimiter="\t")
        if_matrix = matrix.astype(np.float64, copy=True)
        if_matrix[if_matrix < 1.0] = np.nan
        pd_matrix = if_to_pd(if_matrix, threshold=1.0)
        np.savetxt(flamingo_dir / f"IF_Cell_{idx:03d}.txt", if_matrix, fmt="%.6g", delimiter="\t")
        np.savetxt(flamingo_dir / f"PD_Cell_{idx:03d}.txt", pd_matrix, fmt="%.6g", delimiter="\t")

    cell_df, summary_df = evaluate_completion(completed, truth, observed)
    cell_df.insert(0, "dataset_id", record.dataset_id)
    summary_df.insert(0, "dataset_id", record.dataset_id)
    cell_df.to_csv(output_dir / "cell_level_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    log.to_csv(output_dir / "completion_log.tsv", sep="\t", index=False)
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(asdict(record) | {"n_cells": completed.shape[0], "n_beads": completed.shape[1]}, handle, indent=2)


def run_one(
    record: DatasetRecord,
    input_root: Path,
    output_root: Path,
    max_iter: int,
    tol: float,
    mu: float,
    max_mu: float,
    rho: float,
    n_threads: int,
    force_prepare: bool,
    keep_observed: bool,
) -> None:
    input_dir = prepare_input(record, input_root=input_root, force=force_prepare)
    output_dir = output_root / record.dataset_id
    observed = np.load(input_dir / "observed_if_tensor.npy").astype(np.float64)
    truth = np.load(input_dir / "truth_if_tensor.npy").astype(np.float64)
    positive_observed = observed[np.isfinite(observed) & (observed > 0)]
    input_scale = float(np.max(positive_observed)) if positive_observed.size else 1.0
    if not np.isfinite(input_scale) or input_scale <= 0:
        input_scale = 1.0
    scaled_observed = observed / input_scale
    LOGGER.info(
        "Dataset %s tensor=%s nnz=%d input_scale=%.6g mu=%.6g",
        record.dataset_id,
        observed.shape,
        int(np.count_nonzero(observed)),
        input_scale,
        mu,
    )
    completed, log = complete_tensor(
        observed=scaled_observed,
        max_iter=max_iter,
        tol=tol,
        mu=mu,
        max_mu=max_mu,
        rho=rho,
        n_threads=n_threads,
    )
    completed = completed * input_scale
    if keep_observed:
        completed[observed > 0] = observed[observed > 0]
    write_completion_outputs(output_dir, completed, truth, observed, log, record)
    with (output_dir / "run_parameters.json").open("w") as handle:
        json.dump(
            {
                "input_scale": input_scale,
                "max_iter": max_iter,
                "tol": tol,
                "mu": mu,
                "max_mu": max_mu,
                "rho": rho,
                "n_threads": n_threads,
                "keep_observed": keep_observed,
                "optimization_input": "observed_if_tensor / input_scale",
            },
            handle,
            indent=2,
        )
    LOGGER.info("Wrote %s", output_dir)


def combine_summaries(output_root: Path, manifest: Path) -> None:
    records = read_manifest(manifest)
    cell_frames = []
    summary_frames = []
    for record in records:
        dataset_dir = output_root / record.dataset_id
        cell_path = dataset_dir / "cell_level_metrics.csv"
        summary_path = dataset_dir / "summary_metrics.csv"
        if cell_path.exists():
            cell_frames.append(pd.read_csv(cell_path))
        if summary_path.exists():
            summary_frames.append(pd.read_csv(summary_path))
    if cell_frames:
        pd.concat(cell_frames, ignore_index=True).to_csv(output_root / "all_cell_level_metrics.csv", index=False)
    if summary_frames:
        pd.concat(summary_frames, ignore_index=True).to_csv(output_root / "all_summary_metrics.csv", index=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("write-manifest")
    manifest_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    manifest_parser.add_argument("--manifest", type=Path, default=DEFAULT_INPUT_ROOT / "manifest.tsv")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--manifest", type=Path, default=DEFAULT_INPUT_ROOT / "manifest.tsv")
    prepare_parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    prepare_parser.add_argument("--dataset", default=None)
    prepare_parser.add_argument("--task-id", type=int, default=None)
    prepare_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run-one")
    run_parser.add_argument("--manifest", type=Path, default=DEFAULT_INPUT_ROOT / "manifest.tsv")
    run_parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run_parser.add_argument("--dataset", default=None)
    run_parser.add_argument("--task-id", type=int, default=None)
    run_parser.add_argument("--max-iter", type=int, default=500)
    run_parser.add_argument("--tol", type=float, default=1e-4)
    run_parser.add_argument("--mu", type=float, default=1e-4)
    run_parser.add_argument("--max-mu", type=float, default=1e10)
    run_parser.add_argument("--rho", type=float, default=1.1)
    run_parser.add_argument("--n-threads", type=int, default=8)
    run_parser.add_argument("--force-prepare", action="store_true")
    run_parser.add_argument("--no-keep-observed", action="store_true")

    combine_parser = subparsers.add_parser("combine-summaries")
    combine_parser.add_argument("--manifest", type=Path, default=DEFAULT_INPUT_ROOT / "manifest.tsv")
    combine_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    if args.command == "write-manifest":
        records = discover_datasets(args.data_root.resolve())
        if not records:
            raise FileNotFoundError(f"No simulation datasets found under {args.data_root}")
        write_manifest(records, args.manifest.resolve())
        LOGGER.info("Wrote %d datasets to %s", len(records), args.manifest.resolve())
    elif args.command == "prepare":
        records = read_manifest(args.manifest.resolve())
        record = select_record(records, args.dataset, args.task_id)
        prepare_input(record, args.input_root.resolve(), force=args.force)
    elif args.command == "run-one":
        records = read_manifest(args.manifest.resolve())
        record = select_record(records, args.dataset, args.task_id)
        run_one(
            record=record,
            input_root=args.input_root.resolve(),
            output_root=args.output_root.resolve(),
            max_iter=args.max_iter,
            tol=args.tol,
            mu=args.mu,
            max_mu=args.max_mu,
            rho=args.rho,
            n_threads=args.n_threads,
            force_prepare=args.force_prepare,
            keep_observed=not args.no_keep_observed,
        )
    elif args.command == "combine-summaries":
        combine_summaries(args.output_root.resolve(), args.manifest.resolve())


if __name__ == "__main__":
    main(sys.argv[1:])
