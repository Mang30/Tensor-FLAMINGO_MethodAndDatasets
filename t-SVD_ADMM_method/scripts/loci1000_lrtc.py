#!/usr/bin/env python3
"""t-SVD LRTC validation on loci_1000 paper data.

Stack downsampled+noisy distance matrices into a tensor,
run t-SVD low-rank tensor completion, evaluate against no_noise ground truth.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfftw
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger("loci1000_lrtc")

DATA_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC"
    "/1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/1_RawData/simulation_05FLAMINGO/loci_1000"
)

DOWNSAMPLE_RATES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NOISE_LEVELS = [0, 1, 2]


def load_distance_matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter="\t", dtype=np.float64)
    matrix[~np.isfinite(matrix)] = 0.0
    matrix[matrix < 0] = 0.0
    matrix = np.maximum(matrix, matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def load_ground_truth() -> np.ndarray:
    return load_distance_matrix(DATA_DIR / "Distance_matrix_no_noise.txt")


def build_condition_id(noise_level: int, downsample_rate: float) -> str:
    return f"noise{noise_level}_ds{downsample_rate}"


def build_tensor(noise_level: int, downsample_rates: list[float]) -> tuple[np.ndarray, list[str]]:
    slices = []
    labels = []
    for ds in downsample_rates:
        if noise_level == 0:
            fname = f"Distance_matrix_no_noise_downsampled_{ds}.txt"
        else:
            fname = f"Distance_matrix_noise_level_{noise_level}_downsampled_{ds}.txt"
        path = DATA_DIR / fname
        m = load_distance_matrix(path)
        slices.append(m)
        labels.append(build_condition_id(noise_level, ds))
    return np.stack(slices, axis=0), labels


class FFTWPlans:
    def __init__(self, shape: tuple[int, int, int], n_threads: int):
        self.fft_in = pyfftw.empty_aligned(shape, dtype="complex128")
        self.fft_out = pyfftw.empty_aligned(shape, dtype="complex128")
        self.ifft_in = pyfftw.empty_aligned(shape, dtype="complex128")
        self.ifft_out = pyfftw.empty_aligned(shape, dtype="complex128")
        self.fft = pyfftw.FFTW(
            self.fft_in, self.fft_out, axes=(0,),
            threads=n_threads, flags=("FFTW_ESTIMATE",),
        )
        self.ifft = pyfftw.FFTW(
            self.ifft_in, self.ifft_out, axes=(0,),
            direction="FFTW_BACKWARD", threads=n_threads, flags=("FFTW_ESTIMATE",),
        )

    def forward(self, tensor: np.ndarray) -> np.ndarray:
        self.fft_in[:, :, :] = tensor
        return self.fft().copy()

    def backward(self, tensor: np.ndarray) -> np.ndarray:
        self.ifft_in[:, :, :] = tensor
        return self.ifft().copy()


def prox_tnn(y_tensor: np.ndarray, threshold: float, plans: FFTWPlans) -> np.ndarray:
    n_cells = y_tensor.shape[0]
    x_fft = np.zeros(y_tensor.shape, dtype=np.complex128)
    y_fft = plans.forward(y_tensor.astype(np.complex128, copy=False))
    svd_indices = [0] + list(range(1, (n_cells + 1) // 2))
    for i in svd_indices:
        u, singular_values, vh = np.linalg.svd(y_fft[i], hermitian=True)
        rank = int(np.sum(singular_values > threshold))
        if rank:
            shrunk = singular_values[:rank] - threshold
            x_fft[i] = (u[:, :rank] * shrunk) @ vh[:rank, :]
        if i != 0 and i < n_cells:
            x_fft[n_cells - i] = x_fft[i].conjugate()
    if n_cells % 2 == 0:
        i = n_cells // 2
        u, singular_values, vh = np.linalg.svd(y_fft[i], hermitian=True)
        rank = int(np.sum(singular_values > threshold))
        if rank:
            shrunk = singular_values[:rank] - threshold
            x_fft[i] = (u[:, :rank] * shrunk) @ vh[:rank, :]
    return plans.backward(x_fft)


def complete_tensor(
    observed: np.ndarray, max_iter: int, tol: float,
    mu: float, max_mu: float, rho: float, n_threads: int,
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
        log_rows.append({
            "iteration": iteration, "residual_norm": residual_norm,
            "change": change, "mu": mu, "seconds": time.time() - start,
        })
        LOGGER.info(
            "Iter:%03d residual=%.6g change=%.6g mu=%.6g sec=%.2f",
            iteration, residual_norm, change, mu, log_rows[-1]["seconds"],
        )
        if change < tol:
            break
        y_tensor = y_tensor + mu * residual
        mu = min(rho * mu, max_mu)

    completed = np.real(x_tensor).astype(np.float64)
    completed[~np.isfinite(completed)] = 0.0
    completed[completed < 0] = 0.0
    for idx in range(completed.shape[0]):
        completed[idx] = np.maximum(completed[idx], completed[idx].T)
        np.fill_diagonal(completed[idx], 0.0)
    return completed, pd.DataFrame(log_rows)


def _safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    value = fn(x, y)
    if hasattr(value, "statistic"):
        return float(value.statistic)
    if isinstance(value, tuple):
        return float(value[0])
    return float(value)


def evaluate_slice(
    completed_slice: np.ndarray, observed_slice: np.ndarray, truth: np.ndarray
) -> dict[str, float]:
    n = truth.shape[0]
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)
    truth_mask = lower & (truth > 0)
    observed_mask = truth_mask & (observed_slice > 0)
    heldout_mask = truth_mask & (observed_slice == 0)
    result = {"n_truth": int(truth_mask.sum())}
    for prefix, mask in [("all", truth_mask), ("observed", observed_mask), ("heldout", heldout_mask)]:
        valid = mask & np.isfinite(completed_slice) & np.isfinite(truth)
        pred = completed_slice[valid]
        gt = truth[valid]
        if valid.sum() == 0:
            result.update({f"pcc_{prefix}": float("nan"), f"spearman_{prefix}": float("nan"),
                          f"mae_{prefix}": float("nan"), f"rmse_{prefix}": float("nan")})
            continue
        diff = pred - gt
        result[f"pcc_{prefix}"] = _safe_corr(pearsonr, pred, gt)
        result[f"spearman_{prefix}"] = _safe_corr(spearmanr, pred, gt)
        result[f"mae_{prefix}"] = float(np.mean(np.abs(diff)))
        result[f"rmse_{prefix}"] = float(np.sqrt(np.mean(diff**2)))
    return result


def run_condition(args: argparse.Namespace, noise_level: int, downsample_rates: list[float]) -> None:
    condition_id = build_condition_id(noise_level, downsample_rates[0]) if len(downsample_rates) == 1 else f"noise{noise_level}_ds{downsample_rates[0]}-{downsample_rates[-1]}"
    output_dir = Path(args.output_root) / condition_id
    output_dir.mkdir(parents=True, exist_ok=True)

    truth = load_ground_truth()
    observed_tensor, labels = build_tensor(noise_level, downsample_rates)
    LOGGER.info("Condition %s: tensor=%s nnz=%d", condition_id, observed_tensor.shape, int(np.count_nonzero(observed_tensor)))

    positive = observed_tensor[np.isfinite(observed_tensor) & (observed_tensor > 0)]
    input_scale = float(np.max(positive)) if positive.size else 1.0
    if not np.isfinite(input_scale) or input_scale <= 0:
        input_scale = 1.0
    scaled_observed = observed_tensor / input_scale
    LOGGER.info("input_scale=%.6g", input_scale)

    completed, log = complete_tensor(
        observed=scaled_observed, max_iter=args.max_iter, tol=args.tol,
        mu=args.mu, max_mu=args.max_mu, rho=args.rho, n_threads=args.n_threads,
    )
    completed = completed * input_scale

    np.save(output_dir / "completed_tensor.npy", completed)
    np.save(output_dir / "observed_tensor.npy", observed_tensor)
    np.save(output_dir / "truth.npy", truth)
    log.to_csv(output_dir / "completion_log.tsv", sep="\t", index=False)

    rows = []
    for idx, label in enumerate(labels):
        metrics = evaluate_slice(completed[idx], observed_tensor[idx], truth)
        rows.append({"condition_id": condition_id, "slice_idx": idx, "slice_label": label, **metrics})

    cell_df = pd.DataFrame(rows)
    cell_df.to_csv(output_dir / "slice_metrics.csv", index=False)
    with (output_dir / "run_parameters.json").open("w") as f:
        json.dump({
            "noise_level": noise_level, "downsample_rates": downsample_rates,
            "input_scale": input_scale, "max_iter": args.max_iter, "tol": args.tol,
            "mu": args.mu, "max_mu": args.max_mu, "rho": args.rho, "n_threads": args.n_threads,
        }, f, indent=2)
    LOGGER.info("Wrote %s", output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent.parent / "output_loci1000")
    p.add_argument("--noise-level", type=int, default=None, help="Single noise level (0=no_noise, 1, 2)")
    p.add_argument("--downsample", type=float, default=None, help="Single downsample rate, or omit for all rates stacked")
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--mu", type=float, default=1e-4)
    p.add_argument("--max-mu", type=float, default=1e10)
    p.add_argument("--rho", type=float, default=1.1)
    p.add_argument("--n-threads", type=int, default=8)
    p.add_argument("--mode", choices=["single", "all_rates", "all_conditions"], default="single",
                   help="single=one rate, all_rates=stack all rates for one noise level, all_conditions=run all")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.mode == "all_conditions":
        for nl in NOISE_LEVELS:
            for ds in DOWNSAMPLE_RATES:
                run_condition(args, nl, [ds])
        for nl in NOISE_LEVELS:
            run_condition(args, nl, DOWNSAMPLE_RATES)
    elif args.mode == "all_rates":
        nl = args.noise_level if args.noise_level is not None else 1
        run_condition(args, nl, DOWNSAMPLE_RATES)
    else:
        if args.noise_level is None or args.downsample is None:
            raise ValueError("--noise-level and --downsample required for mode=single")
        run_condition(args, args.noise_level, [args.downsample])


if __name__ == "__main__":
    main(sys.argv[1:])