#!/usr/bin/env python3
"""Prepare and evaluate raw-contact LRTC inputs for simulation_multicondition."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
SCHIC_ROOT = BASE_DIR.parents[2]
DEFAULT_RAW_ROOT = (
    SCHIC_ROOT
    / "1_Dataset/5-Tensor-FLAMINGO_Simulation_Data/1_RawData/simulation_multicondition"
)
DEFAULT_INPUT_ROOT = BASE_DIR / "input_raw_contact"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output_raw_contact"
DEFAULT_MANIFEST = DEFAULT_INPUT_ROOT / "manifest.tsv"


@dataclass(frozen=True)
class RawDataset:
    dataset_id: str
    beads: int
    w: str
    level: int
    raw_dir: str
    metadata_csv: str


def natural_sample_sort_key(name: str | Path) -> tuple[int, int]:
    match = re.search(r"consensus_(\d+)_slice_(\d+)", str(name))
    if not match:
        raise ValueError(f"Cannot parse consensus/slice ids from {name}")
    return int(match.group(1)), int(match.group(2))


def discover_raw_datasets(raw_root: Path) -> list[RawDataset]:
    records: list[RawDataset] = []
    for metadata in sorted(raw_root.glob("beads_*/W_*/level_*/metadata.csv")):
        level_dir = metadata.parent
        w_dir = level_dir.parent
        beads_dir = w_dir.parent
        beads = int(beads_dir.name.split("_", 1)[1])
        w = w_dir.name.split("_", 1)[1]
        level = int(level_dir.name.split("_", 1)[1])
        records.append(
            RawDataset(
                dataset_id=f"beads{beads}_W{w}_level{level}",
                beads=beads,
                w=w,
                level=level,
                raw_dir=str(level_dir),
                metadata_csv=str(metadata),
            )
        )
    return records


def write_manifest(records: list[RawDataset], manifest: Path) -> None:
    if not records:
        raise ValueError("No raw datasets found")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()), delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_manifest(manifest: Path) -> list[RawDataset]:
    with manifest.open(newline="") as handle:
        return [
            RawDataset(
                dataset_id=row["dataset_id"],
                beads=int(row["beads"]),
                w=row["w"],
                level=int(row["level"]),
                raw_dir=row["raw_dir"],
                metadata_csv=row["metadata_csv"],
            )
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def select_record(records: list[RawDataset], dataset: str | None, task_id: int | None) -> RawDataset:
    if dataset is not None:
        for record in records:
            if record.dataset_id == dataset:
                return record
        raise ValueError(f"Dataset not found: {dataset}")
    if task_id is None:
        raise ValueError("Either --dataset or --task-id is required")
    if task_id < 0 or task_id >= len(records):
        raise IndexError(f"task id {task_id} outside 0-{len(records) - 1}")
    return records[task_id]


def load_matrix(path: Path) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter="\t", dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{path} is not a square matrix: {matrix.shape}")
    matrix[~np.isfinite(matrix)] = 0.0
    matrix[matrix < 0] = 0.0
    matrix = np.maximum(matrix, matrix.T)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def prepare_one(record: RawDataset, input_root: Path, force: bool = False) -> Path:
    dataset_input = input_root / record.dataset_id
    matrix_dir = dataset_input / "contact_matrices"
    complete_marker = dataset_input / ".complete"
    if complete_marker.exists() and not force:
        return dataset_input

    metadata = pd.read_csv(record.metadata_csv)
    metadata = metadata.sort_values(["consensus_id", "slice_id"]).reset_index(drop=True)
    required = {"sparse_contact_path", "gt_contact_path"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"{record.metadata_csv} is missing columns: {sorted(missing)}")

    matrix_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell_idx, row in metadata.iterrows():
        src = Path(row["sparse_contact_path"])
        dst_name = f"RawCount_Cell_{cell_idx + 1:03d}.txt"
        shutil.copy2(src, matrix_dir / dst_name)
        rows.append(
            {
                "cell_idx": cell_idx,
                "cell_number": cell_idx + 1,
                "input_file": dst_name,
                "sample_id": row["sample_id"],
                "consensus_id": int(row["consensus_id"]),
                "slice_id": int(row["slice_id"]),
                "sparse_contact_path": row["sparse_contact_path"],
                "gt_contact_path": row["gt_contact_path"],
                "observed_fraction": float(row["observed_fraction"]),
                "missing_fraction": float(row["missing_fraction"]),
            }
        )

    index_df = pd.DataFrame(rows)
    index_df.to_csv(dataset_input / "input_file_index.csv", index=False)
    with (dataset_input / "metadata.json").open("w") as handle:
        json.dump(asdict(record) | {"n_cells": len(index_df)}, handle, indent=2)
    complete_marker.write_text("complete\n")
    return dataset_input


def _safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    value = fn(x, y)
    if hasattr(value, "statistic"):
        return float(value.statistic)
    if isinstance(value, tuple):
        return float(value[0])
    return float(value)


def _metrics(prefix: str, pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(pred) & np.isfinite(truth)
    x = pred[valid]
    y = truth[valid]
    if x.size == 0:
        return {
            f"n_{prefix}": 0,
            f"pcc_{prefix}": float("nan"),
            f"spearman_{prefix}": float("nan"),
            f"mae_{prefix}": float("nan"),
            f"rmse_{prefix}": float("nan"),
            f"relative_error_{prefix}": float("nan"),
            f"log1p_pcc_{prefix}": float("nan"),
            f"log1p_spearman_{prefix}": float("nan"),
            f"log1p_mae_{prefix}": float("nan"),
            f"log1p_rmse_{prefix}": float("nan"),
        }

    diff = x - y
    denom = np.linalg.norm(y)
    log_x = np.log1p(np.maximum(x, 0.0))
    log_y = np.log1p(np.maximum(y, 0.0))
    log_diff = log_x - log_y
    return {
        f"n_{prefix}": int(x.size),
        f"pcc_{prefix}": _safe_corr(pearsonr, x, y),
        f"spearman_{prefix}": _safe_corr(spearmanr, x, y),
        f"mae_{prefix}": float(np.mean(np.abs(diff))),
        f"rmse_{prefix}": float(np.sqrt(np.mean(diff**2))),
        f"relative_error_{prefix}": float(np.linalg.norm(diff) / denom) if denom > 0 else float("nan"),
        f"log1p_pcc_{prefix}": _safe_corr(pearsonr, log_x, log_y),
        f"log1p_spearman_{prefix}": _safe_corr(spearmanr, log_x, log_y),
        f"log1p_mae_{prefix}": float(np.mean(np.abs(log_diff))),
        f"log1p_rmse_{prefix}": float(np.sqrt(np.mean(log_diff**2))),
    }


def contact_metrics_for_cell(completed: np.ndarray, observed: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if completed.shape != observed.shape or completed.shape != truth.shape:
        raise ValueError(f"Shape mismatch: completed={completed.shape}, observed={observed.shape}, truth={truth.shape}")
    n = completed.shape[0]
    lower = np.tril(np.ones((n, n), dtype=bool), k=-1)
    truth_mask = lower & np.isfinite(truth) & (truth > 0)
    observed_mask = truth_mask & (observed > 0)
    heldout_mask = truth_mask & (observed == 0)
    out = {}
    out.update(_metrics("all", completed, truth, truth_mask))
    out.update(_metrics("observed", completed, truth, observed_mask))
    out.update(_metrics("heldout", completed, truth, heldout_mask))
    return out


def evaluate_one(record: RawDataset, input_root: Path, output_root: Path, npy_name: str = "completed_tensor.npy") -> None:
    input_dir = input_root / record.dataset_id
    output_dir = output_root / record.dataset_id
    completed_path = output_dir / npy_name
    if not completed_path.exists():
        alt = output_dir / f"{record.dataset_id}.npy"
        if alt.exists():
            completed_path = alt
        else:
            raise FileNotFoundError(f"Missing completed tensor for {record.dataset_id}: {completed_path}")
    completed = np.real(np.load(completed_path)).astype(np.float64)
    index_df = pd.read_csv(input_dir / "input_file_index.csv")
    rows = []
    for _, row in index_df.iterrows():
        cell_idx = int(row["cell_idx"])
        observed = load_matrix(input_dir / "contact_matrices" / row["input_file"])
        truth = load_matrix(Path(row["gt_contact_path"]))
        metrics = contact_metrics_for_cell(completed[cell_idx], observed, truth)
        rows.append(
            {
                "dataset_id": record.dataset_id,
                "cell_idx": cell_idx,
                "cell_number": int(row["cell_number"]),
                "sample_id": row["sample_id"],
                "consensus_id": int(row["consensus_id"]),
                "slice_id": int(row["slice_id"]),
                **metrics,
            }
        )

    cell_df = pd.DataFrame(rows)
    summary_rows = []
    for label, group in [("all_cells", cell_df), *cell_df.groupby("consensus_id")]:
        summary = {"dataset_id": record.dataset_id, "group": str(label), "n_cells": int(len(group))}
        for col in cell_df.columns:
            if col in {"dataset_id", "cell_idx", "cell_number", "sample_id", "consensus_id", "slice_id"}:
                continue
            summary[f"{col}_mean"] = float(group[col].mean(skipna=True))
            summary[f"{col}_std"] = float(group[col].std(skipna=True))
        summary_rows.append(summary)
    cell_df.to_csv(output_dir / "contact_cell_level_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "contact_summary_metrics.csv", index=False)


def combine(output_root: Path, manifest: Path) -> None:
    frames = []
    summaries = []
    for record in read_manifest(manifest):
        cell_path = output_root / record.dataset_id / "contact_cell_level_metrics.csv"
        summary_path = output_root / record.dataset_id / "contact_summary_metrics.csv"
        if cell_path.exists():
            frames.append(pd.read_csv(cell_path))
        if summary_path.exists():
            summaries.append(pd.read_csv(summary_path))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(output_root / "all_contact_cell_level_metrics.csv", index=False)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(output_root / "all_contact_summary_metrics.csv", index=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("write-manifest")
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    p = sub.add_parser("prep")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--dataset", default=None)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("eval")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--dataset", default=None)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--npy-name", default="completed_tensor.npy")

    p = sub.add_parser("combine")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "write-manifest":
        records = discover_raw_datasets(args.raw_root.resolve())
        write_manifest(records, args.manifest.resolve())
        print(f"Wrote {len(records)} datasets to {args.manifest.resolve()}")
    elif args.command == "prep":
        record = select_record(read_manifest(args.manifest.resolve()), args.dataset, args.task_id)
        print(prepare_one(record, args.input_root.resolve(), force=args.force))
    elif args.command == "eval":
        record = select_record(read_manifest(args.manifest.resolve()), args.dataset, args.task_id)
        evaluate_one(record, args.input_root.resolve(), args.output_root.resolve(), npy_name=args.npy_name)
    elif args.command == "combine":
        combine(args.output_root.resolve(), args.manifest.resolve())


if __name__ == "__main__":
    main(sys.argv[1:])
