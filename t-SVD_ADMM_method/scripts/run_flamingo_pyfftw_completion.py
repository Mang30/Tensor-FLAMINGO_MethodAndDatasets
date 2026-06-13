#!/usr/bin/env python3
"""Run first-version FLAMINGO-style tensor completion on RawCount matrices."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfftw


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
SCHIC_ROOT = BENCHMARK_ROOT.parents[1]
DEFAULT_INPUT_ROOT = SCHIC_ROOT / "5_baseline/9_FLAMINGO/input"
DEFAULT_OUTPUT_ROOT = SCHIC_ROOT / "5_baseline/9_FLAMINGO/output"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger("flamingo_pyfftw")


def _serial_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.linalg.svd(matrix, hermitian=True)


def _ray_svd_batch(matrices: list[np.ndarray], n_threads: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    import ray

    if not ray.is_initialized():
        ray.init(address="local", num_cpus=n_threads, include_dashboard=False, ignore_reinit_error=True)

    @ray.remote
    def _remote_svd(matrix):
        return np.linalg.svd(matrix, hermitian=True)

    return ray.get([_remote_svd.remote(matrix) for matrix in matrices])


def manifest_datasets(manifest: Path) -> list[str]:
    datasets: list[str] = []
    seen: set[str] = set()
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            dataset = row["dataset"]
            if dataset not in seen:
                datasets.append(dataset)
                seen.add(dataset)
    return datasets


def load_tensor(input_dir: Path) -> tuple[np.ndarray, list[str]]:
    files = sorted(input_dir.glob("RawCount_Cell_*.txt"), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"No RawCount_Cell_*.txt files found in {input_dir}")
    matrices = []
    expected_n = None
    for path in files:
        matrix = np.loadtxt(path, delimiter="\t", dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{path} is not square: {matrix.shape}")
        matrix[~np.isfinite(matrix)] = 0.0
        matrix[matrix < 0] = 0.0
        matrix = np.maximum(matrix, matrix.T)
        np.fill_diagonal(matrix, 0.0)
        expected_n = expected_n or matrix.shape[0]
        if matrix.shape != (expected_n, expected_n):
            raise ValueError(f"{path} shape {matrix.shape} differs from {(expected_n, expected_n)}")
        matrices.append(matrix)
    return np.stack(matrices, axis=0), [path.name for path in files]


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


def prox_tnn(y_tensor: np.ndarray, threshold: float, plans: FFTWPlans, svd_backend: str, n_threads: int) -> np.ndarray:
    n_cells, _, _ = y_tensor.shape
    x_fft = np.zeros(y_tensor.shape, dtype=np.complex128)
    y_fft = plans.forward(y_tensor.astype(np.complex128, copy=False))

    u, singular_values, vh = _serial_svd(y_fft[0])
    rank = int(np.sum(singular_values > threshold))
    if rank:
        shrunk = singular_values[:rank] - threshold
        x_fft[0] = (u[:, :rank] * shrunk) @ vh[:rank, :]

    half_n3 = math.ceil(n_cells / 2)
    svd_indices = list(range(2, half_n3 + 1))
    if svd_backend == "ray" and svd_indices:
        svd_results = _ray_svd_batch([y_fft[i] for i in svd_indices], n_threads=n_threads)
    else:
        svd_results = [_serial_svd(y_fft[i]) for i in svd_indices]

    for i, (u, singular_values, vh) in zip(svd_indices, svd_results, strict=True):
        rank = int(np.sum(singular_values > threshold))
        if rank:
            shrunk = singular_values[:rank] - threshold
            x_fft[i] = (u[:, :rank] * shrunk) @ vh[:rank, :]
        x_fft[n_cells - i] = x_fft[i].conjugate()

    if n_cells % 2 == 0:
        i = half_n3 + 1
        if i < n_cells:
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
    svd_backend: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_tensor = np.zeros(observed.shape, dtype=np.complex128)
    plans = FFTWPlans(observed.shape, n_threads=n_threads)
    omega = observed > 0
    x_tensor[omega] = observed[omega]
    e_tensor = np.zeros(observed.shape, dtype=np.complex128)
    y_tensor = np.zeros(observed.shape, dtype=np.complex128)
    log_rows = []
    total_size = int(np.prod(observed.shape))
    best_x = x_tensor.copy()
    best_residual_norm = float("inf")
    best_iteration = -1

    for iteration in range(max_iter):
        start = time.time()
        prev_x = x_tensor.copy()
        prev_e = e_tensor.copy()
        x_tensor = prox_tnn(
            -e_tensor + observed + y_tensor / mu,
            1.0 / mu,
            plans=plans,
            svd_backend=svd_backend,
            n_threads=n_threads,
        )
        e_tensor = observed - x_tensor + y_tensor / mu
        e_tensor[omega] = 0.0
        residual = observed - x_tensor - e_tensor
        change_x = float(np.max(np.abs(prev_x.reshape(total_size) - x_tensor.reshape(total_size))))
        change_e = float(np.max(np.abs(prev_e.reshape(total_size) - e_tensor.reshape(total_size))))
        residual_max = float(np.max(np.abs(residual.reshape(total_size))))
        change = max(change_x, change_e, residual_max)
        residual_norm = float(np.linalg.norm(residual.reshape(total_size)))
        improved = residual_norm < best_residual_norm
        if improved:
            best_residual_norm = residual_norm
            best_iteration = iteration
            best_x = x_tensor.copy()
        log_rows.append(
            {
                "iteration": iteration,
                "residual_norm": residual_norm,
                "change": change,
                "mu": mu,
                "seconds": time.time() - start,
                "is_best": improved,
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
    LOGGER.info("Selected best iteration %d with residual %.6g", best_iteration, best_residual_norm)
    return best_x, x_tensor, pd.DataFrame(log_rows)


def select_completed_tensor(best: np.ndarray, final: np.ndarray, selection: str) -> np.ndarray:
    if selection == "best":
        return best
    if selection == "final":
        return final
    raise ValueError(f"Unknown completion selection: {selection}")


def postprocess(completed: np.ndarray, observed: np.ndarray, keep_observed: bool) -> np.ndarray:
    completed = np.real(completed).astype(np.float32, copy=False)
    completed[~np.isfinite(completed)] = 0.0
    completed[completed < 0] = 0.0
    for idx in range(completed.shape[0]):
        completed[idx] = np.maximum(completed[idx], completed[idx].T)
        np.fill_diagonal(completed[idx], 0.0)
    if keep_observed:
        omega = observed > 0
        completed[omega] = observed[omega].astype(np.float32, copy=False)
    return completed


def write_outputs(output_dir: Path, completed: np.ndarray, input_files: list[str], log: pd.DataFrame) -> None:
    matrix_dir = output_dir / "completed_matrices"
    flamingo_dir = output_dir / "high_res_contact_maps_FLAMINGO"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    flamingo_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "completed_tensor.npy", completed)
    np.save(output_dir / "high_resolution.npy", completed)
    for idx, matrix in enumerate(completed, start=1):
        np.savetxt(matrix_dir / f"Completed_Cell_{idx}.txt", matrix, fmt="%.6g", delimiter="\t")
        if_matrix = matrix.copy()
        if_matrix[if_matrix < 1.0] = np.nan
        pd_matrix = np.power(if_matrix, -0.25)
        np.savetxt(flamingo_dir / f"IF_Cell_{idx}.txt", if_matrix, fmt="%.6g", delimiter="\t")
        np.savetxt(flamingo_dir / f"PD_Cell_{idx}.txt", pd_matrix, fmt="%.6g", delimiter="\t")
    pd.DataFrame({"input_file": input_files}).to_csv(output_dir / "input_file_index.csv", index=False)
    log.to_csv(output_dir / "completion_log.tsv", sep="\t", index=False)


def run_dataset(args: argparse.Namespace, dataset: str) -> None:
    input_dir = args.input_root / dataset / args.input_subdir
    output_dir = args.output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    observed, input_files = load_tensor(input_dir)
    LOGGER.info("Dataset %s tensor=%s nnz=%d", dataset, observed.shape, int(np.count_nonzero(observed)))
    best_completed, final_completed, log = complete_tensor(
        observed=observed,
        max_iter=args.max_iter,
        tol=args.tol,
        mu=args.mu,
        max_mu=args.max_mu,
        rho=args.rho,
        n_threads=args.n_threads,
        svd_backend=args.svd_backend,
    )
    completed = select_completed_tensor(best_completed, final_completed, args.selection)
    LOGGER.info("Using %s iteration tensor for output", args.selection)
    completed = postprocess(completed, observed, keep_observed=args.keep_observed)
    write_outputs(output_dir, completed, input_files, log)
    LOGGER.info("Wrote %s", output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--input-subdir", default="highres_contact_maps_transformed_original")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--mu", type=float, default=1e-4)
    parser.add_argument("--max-mu", type=float, default=1e10)
    parser.add_argument("--rho", type=float, default=1.1)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--svd-backend", choices=("serial", "ray"), default="serial")
    parser.add_argument("--selection", choices=("best", "final"), default="best")
    parser.add_argument("--keep-observed", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args(sys.argv[1:])
    try:
        args.input_root = args.input_root.resolve()
        args.output_root = args.output_root.resolve()
        args.output_root.mkdir(parents=True, exist_ok=True)
        if args.datasets:
            datasets = args.datasets
        else:
            manifest = (args.manifest or (args.input_root / "manifest.tsv")).resolve()
            datasets = manifest_datasets(manifest)
        for dataset in datasets:
            run_dataset(args, dataset)
    finally:
        if args.svd_backend == "ray":
            import ray

            if ray.is_initialized():
                ray.shutdown()


if __name__ == "__main__":
    main()
