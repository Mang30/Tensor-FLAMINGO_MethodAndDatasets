#!/usr/bin/env python3
"""Generate multi-condition Tensor-FLAMINGO-style simulation datasets.

This is a thin batch driver around generate_simulation_data.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from generate_simulation_data import generate_full_simulation_dataset, validate_generated_data


DEFAULT_OUT_ROOT = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/"
    "5-Tensor-FLAMINGO_Simulation_Data/1_RawData/simulation_multicondition"
)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def format_w(value: float) -> str:
    return f"{value:.3g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--beads", default="300,500")
    parser.add_argument("--w-values", default="0.5,0.7,0.9")
    parser.add_argument("--noise-levels", default="level_1,level_2")
    parser.add_argument("--n-consensus", type=int, default=3)
    parser.add_argument("--cells-per-type", type=int, default=10)
    parser.add_argument("--retention-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--validate", action="store_true", default=True)
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    beads_values = parse_csv_ints(args.beads)
    w_values = parse_csv_floats(args.w_values)
    noise_levels = parse_csv_strings(args.noise_levels)
    args.out_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    condition_index = 0
    for n_beads in beads_values:
        for W in w_values:
            for noise_level in noise_levels:
                condition_index += 1
                out_dir = args.out_root / f"beads_{n_beads}" / f"W_{format_w(W)}" / noise_level
                seed = args.seed + condition_index * 1000
                info = generate_full_simulation_dataset(
                    structure_params={
                        "n_beads": n_beads,
                        "n_consensus": args.n_consensus,
                        "n_cells_per_consensus": args.cells_per_type,
                        "coord_generation_method": "sarw",
                        "seed": seed,
                    },
                    similarity_params={"W_values": [W] * max(args.n_consensus - 1, 1)},
                    noise_params={"noise_levels": [noise_level]},
                    downsampling_params={"retention_rates": [args.retention_rate]},
                    output_params={
                        "output_base_dir": str(out_dir),
                        "overwrite": args.overwrite,
                    },
                    verbose=False,
                )
                if args.validate:
                    validate_generated_data(out_dir)
                manifest_rows.append(
                    {
                        "n_beads": n_beads,
                        "W": W,
                        "noise_level": noise_level,
                        "n_consensus": args.n_consensus,
                        "cells_per_type": args.cells_per_type,
                        "retention_rate": args.retention_rate,
                        "seed": seed,
                        "output_dir": str(out_dir),
                        "n_total_cells": info["n_total_cells"],
                    }
                )
                print(
                    f"generated n_beads={n_beads} W={W} noise={noise_level} "
                    f"cells={info['n_total_cells']} -> {out_dir}"
                )

    manifest_path = args.out_root / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
