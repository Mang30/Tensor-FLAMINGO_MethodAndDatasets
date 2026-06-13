#!/usr/bin/env python3
"""Convert generated dense IF/contact matrices to reference-compatible formats.

Input layout expected from generate_simulation_data.py:

  gt_contact_matrices/consensus_{type}_contact.txt
  downsampled_contact/consensus_{type}_slice_{cell}.txt

Output layout:

  1_lower_tri_feature/{csv,npz,h5ad}
  2_full_matrix/{npz,txt,schicluster_txt,scVI-3D_txt}
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, save_npz, vstack


DEFAULT_RAW_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/"
    "5-Tensor-FLAMINGO_Simulation_Data/1_RawData/simulation_generated"
)
DEFAULT_OUT_DIR = Path(
    "/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/1_Dataset/"
    "5-Tensor-FLAMINGO_Simulation_Data/2_ProcessedData/simulation_generated"
)


@dataclass(frozen=True)
class DatasetConfig:
    raw_dir: Path
    out_dir: Path
    dataset_prefix: str = "FLAMINGO"
    depth_label: str = "simulation_generated"
    cell_types: tuple[int, ...] = (1, 2, 3)
    cells_per_type: int = 10
    n_bins: int | None = None
    chrom: str = "chr19"
    write_h5ad: bool = True
    overwrite: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset-prefix", default="FLAMINGO")
    parser.add_argument("--depth-label", default="simulation_generated")
    parser.add_argument("--cell-types", default="1,2,3", help="Comma-separated consensus/type ids.")
    parser.add_argument("--cells-per-type", type=int, default=10)
    parser.add_argument("--n-bins", type=int, default=None)
    parser.add_argument("--chrom", default="chr19")
    parser.add_argument("--skip-h5ad", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; rerun with --overwrite: {path}")


def dataset_name(config: DatasetConfig, type_id: int) -> str:
    return f"{config.dataset_prefix}_T{type_id}_{config.depth_label}"


def detect_n_bins(raw_dir: Path, first_type: int) -> int:
    path = raw_dir / "gt_contact_matrices" / f"consensus_{first_type}_contact.txt"
    matrix = read_dense_matrix(path)
    return int(matrix.shape[0])


def read_dense_matrix(path: Path, n_bins: int | None = None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    matrix = np.loadtxt(path)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{path} must be a square dense matrix, got {matrix.shape}")
    if n_bins is not None and matrix.shape != (n_bins, n_bins):
        raise ValueError(f"{path} has shape {matrix.shape}; expected {(n_bins, n_bins)}")
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{path} is not symmetric")
    if not np.allclose(np.diag(matrix), 0.0):
        raise ValueError(f"{path} diagonal is not zero")
    if np.any(matrix < 0):
        raise ValueError(f"{path} contains negative values")
    return matrix.astype(np.float64, copy=False)


def lower_triangle_vector(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {matrix.shape}")
    return matrix[np.tril_indices(matrix.shape[0], k=-1)]


def feature_columns(cells_per_type: int) -> list[str]:
    return [f"c_{i}" for i in range(1, cells_per_type + 1)]


def write_feature_csv(path: Path, feature_matrix: np.ndarray, overwrite: bool) -> None:
    ensure_can_write(path, overwrite)
    pd.DataFrame(feature_matrix, columns=feature_columns(feature_matrix.shape[1])).to_csv(path, index=False)


def write_feature_npz(path: Path, feature_matrix: np.ndarray, overwrite: bool) -> None:
    ensure_can_write(path, overwrite)
    save_npz(path, csr_matrix(feature_matrix.T))


def nonzero_lower_contacts(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.tril(matrix, k=-1)
    rows, cols = np.nonzero(lower)
    return rows, cols, lower[rows, cols]


def write_full_matrix_outputs(config: DatasetConfig, type_id: int, cell_id: int, sim_dense: np.ndarray) -> int:
    name = dataset_name(config, type_id)
    full_dir = config.out_dir / "2_full_matrix"
    npz_dir = full_dir / "npz" / name
    txt_dir = full_dir / "txt" / name
    schicluster_dir = full_dir / "schicluster_txt" / name
    scvi_dir = full_dir / "scVI-3D_txt" / name
    for directory in (npz_dir, txt_dir, schicluster_dir, scvi_dir):
        directory.mkdir(parents=True, exist_ok=True)

    npz_path = npz_dir / f"{name}_cell_{cell_id}.npz"
    txt_path = txt_dir / f"{name}_cell_{cell_id}.txt"
    schicluster_path = schicluster_dir / f"cell_{cell_id}_{config.chrom}.txt"
    scvi_path = scvi_dir / f"{name}_cell_{cell_id}.txt"
    for path in (npz_path, txt_path, schicluster_path, scvi_path):
        ensure_can_write(path, config.overwrite)

    save_npz(npz_path, csr_matrix(sim_dense))

    rows, cols, values = nonzero_lower_contacts(sim_dense)
    pd.DataFrame(
        {"chr1": config.chrom, "pos1": rows, "chr2": config.chrom, "pos2": cols, "count": values}
    ).to_csv(txt_path, sep="\t", header=False, index=False)
    pd.DataFrame({"pos1": rows, "pos2": cols, "count": values}).to_csv(
        schicluster_path, sep="\t", header=False, index=False
    )
    pd.DataFrame(
        {"chr1": config.chrom, "pos1": cols, "chr2": config.chrom, "pos2": rows, "count": values}
    ).to_csv(scvi_path, sep="\t", header=False, index=False)
    return int(values.size)


def write_h5ad(path: Path, matrix: csr_matrix, cell_type_labels: Iterable[str], overwrite: bool) -> None:
    ensure_can_write(path, overwrite)
    if importlib.util.find_spec("anndata") is None:
        raise RuntimeError("anndata is not installed; rerun with --skip-h5ad or install anndata.")
    import anndata as ad

    labels = list(cell_type_labels)
    adata = ad.AnnData(matrix)
    adata.obs_names = [str(i) for i in range(adata.n_obs)]
    adata.var_names = [f"BinPair_{i}" for i in range(adata.n_vars)]
    adata.obs = pd.DataFrame(
        {"cell_type": labels, "batch": [0] * adata.n_obs, "n_genes": [adata.n_vars] * adata.n_obs},
        index=adata.obs_names,
    )
    adata.layers["counts"] = adata.X.copy()
    adata.write(path, compression="gzip")


def process_cell_type(config: DatasetConfig, type_id: int, n_bins: int) -> None:
    n_features = n_bins * (n_bins - 1) // 2
    sim_features = np.zeros((n_features, config.cells_per_type), dtype=np.float64)
    true_features = np.zeros((n_features, config.cells_per_type), dtype=np.float64)

    true_path = config.raw_dir / "gt_contact_matrices" / f"consensus_{type_id}_contact.txt"
    true_dense = read_dense_matrix(true_path, n_bins)
    true_vector = lower_triangle_vector(true_dense)
    total_full_nnz = 0

    for col_idx, cell_id in enumerate(range(1, config.cells_per_type + 1)):
        sim_path = config.raw_dir / "downsampled_contact" / f"consensus_{type_id}_slice_{cell_id}.txt"
        sim_dense = read_dense_matrix(sim_path, n_bins)
        sim_features[:, col_idx] = lower_triangle_vector(sim_dense)
        true_features[:, col_idx] = true_vector
        total_full_nnz += write_full_matrix_outputs(config, type_id, cell_id, sim_dense)

    lower_dir = config.out_dir / "1_lower_tri_feature"
    csv_dir = lower_dir / "csv"
    npz_dir = lower_dir / "npz"
    h5ad_dir = lower_dir / "h5ad"
    for directory in (csv_dir, npz_dir):
        directory.mkdir(parents=True, exist_ok=True)

    name = dataset_name(config, type_id)
    write_feature_csv(csv_dir / f"{name}_sim.csv", sim_features, config.overwrite)
    write_feature_csv(csv_dir / f"{name}_true.csv", true_features, config.overwrite)
    write_feature_npz(npz_dir / f"{name}_sim.npz", sim_features, config.overwrite)
    write_feature_npz(npz_dir / f"{name}_true.npz", true_features, config.overwrite)

    if config.write_h5ad:
        h5ad_dir.mkdir(parents=True, exist_ok=True)
        write_h5ad(
            h5ad_dir / f"{name}_sim.h5ad",
            csr_matrix(sim_features.T),
            [f"T{type_id}"] * config.cells_per_type,
            config.overwrite,
        )

    print(
        f"{name}: features={(n_features, config.cells_per_type)} "
        f"sim_sum={sim_features.sum():.6g} true_sum={true_features.sum():.6g} "
        f"full_lower_nonzero={total_full_nnz}"
    )


def write_all_outputs(config: DatasetConfig, n_bins: int) -> None:
    lower_dir = config.out_dir / "1_lower_tri_feature"
    csv_dir = lower_dir / "csv"
    npz_dir = lower_dir / "npz"
    all_csv_dir = csv_dir / "T1_T2_T3_ALL"
    all_npz_dir = npz_dir / "ALL"
    all_csv_dir.mkdir(parents=True, exist_ok=True)
    all_npz_dir.mkdir(parents=True, exist_ok=True)

    all_sim_blocks = []
    all_true_blocks = []
    all_h5ad_blocks = []
    all_labels = []
    for type_id in config.cell_types:
        name = dataset_name(config, type_id)
        sim_df = pd.read_csv(csv_dir / f"{name}_sim.csv")
        true_df = pd.read_csv(csv_dir / f"{name}_true.csv")
        all_sim_blocks.append(sim_df.to_numpy())
        all_true_blocks.append(true_df.to_numpy())
        if config.write_h5ad:
            all_h5ad_blocks.append(csr_matrix(sim_df.to_numpy().T))
            all_labels.extend([f"T{type_id}"] * config.cells_per_type)

    expected_features = n_bins * (n_bins - 1) // 2
    for kind, blocks in (("sim", all_sim_blocks), ("true", all_true_blocks)):
        all_array = np.concatenate(blocks, axis=1)
        expected_shape = (expected_features, config.cells_per_type * len(config.cell_types))
        if all_array.shape != expected_shape:
            raise ValueError(f"ALL {kind} shape is {all_array.shape}; expected {expected_shape}")
        all_csv = all_csv_dir / f"{config.dataset_prefix}_{config.depth_label}_ALL_{kind}.csv"
        all_npz = all_npz_dir / f"{config.dataset_prefix}_{config.depth_label}_ALL_{kind}.npz"
        ensure_can_write(all_csv, config.overwrite)
        ensure_can_write(all_npz, config.overwrite)
        np.savetxt(all_csv, all_array, delimiter=",")
        save_npz(all_npz, coo_matrix(all_array))
        print(f"ALL {kind}: shape={all_array.shape} sum={all_array.sum():.6g}")

    if config.write_h5ad:
        h5ad_dir = lower_dir / "h5ad"
        h5ad_dir.mkdir(parents=True, exist_ok=True)
        all_matrix = vstack(all_h5ad_blocks, format="csr")
        write_h5ad(
            h5ad_dir / f"{config.dataset_prefix}_{config.depth_label}_three_types_sim.h5ad",
            all_matrix,
            all_labels,
            config.overwrite,
        )


def write_chrom_sizes(config: DatasetConfig, n_bins: int) -> None:
    schicluster_dir = config.out_dir / "2_full_matrix" / "schicluster_txt"
    schicluster_dir.mkdir(parents=True, exist_ok=True)
    path = schicluster_dir / f"simu_{config.chrom}.chrom.sizes"
    ensure_can_write(path, config.overwrite)
    path.write_text(f"{config.chrom}\t{n_bins}\n")


def process_dataset(config: DatasetConfig) -> None:
    n_bins = config.n_bins
    if n_bins is None:
        n_bins = detect_n_bins(config.raw_dir, config.cell_types[0])

    write_chrom_sizes(config, n_bins)
    for type_id in config.cell_types:
        process_cell_type(config, type_id, n_bins)
    write_all_outputs(config, n_bins)


def main() -> int:
    args = parse_args()
    cell_types = tuple(int(item) for item in args.cell_types.split(",") if item)
    config = DatasetConfig(
        raw_dir=args.raw_dir.resolve(),
        out_dir=args.out_dir.resolve(),
        dataset_prefix=args.dataset_prefix,
        depth_label=args.depth_label,
        cell_types=cell_types,
        cells_per_type=args.cells_per_type,
        n_bins=args.n_bins,
        chrom=args.chrom,
        write_h5ad=not args.skip_h5ad,
        overwrite=args.overwrite,
    )
    process_dataset(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
