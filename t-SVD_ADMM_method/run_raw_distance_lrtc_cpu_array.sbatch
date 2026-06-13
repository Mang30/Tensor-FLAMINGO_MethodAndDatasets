#!/usr/bin/env bash
#SBATCH -J rawdist_lrtc
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-11
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData/output_distance/logs/slurm_%x_%A_%a.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData/output_distance/logs/slurm_%x_%A_%a.err

set -euo pipefail

export LC_ALL=C
export LANG=C
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData"
PYTHON_BIN="${PYTHON_BIN:-/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python}"
PREP_EVAL="${BASE_DIR}/scripts/raw_distance_lrtc_prep_eval.py"
LRTC_SCRIPT="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/4_ImputationCriteria/benchmark_criteria_master/9_FLAMINGO/run_flamingo_pyfftw_completion.py"
MANIFEST="${BASE_DIR}/input_distance/manifest.tsv"
INPUT_ROOT="${BASE_DIR}/input_distance"
OUTPUT_ROOT="${BASE_DIR}/output_distance"

mkdir -p "${OUTPUT_ROOT}/logs"

echo "task_id=${SLURM_ARRAY_TASK_ID}"
echo "node=$(hostname)"
echo "account=pi_limin_r"
echo "python=${PYTHON_BIN}"

"${PYTHON_BIN}" "${PREP_EVAL}" prep \
  --manifest "${MANIFEST}" \
  --input-root "${INPUT_ROOT}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --force

DATASET="$(awk -F '\t' -v i="$((SLURM_ARRAY_TASK_ID + 2))" 'NR==i {print $1}' "${MANIFEST}")"
echo "dataset=${DATASET}"

"${PYTHON_BIN}" "${LRTC_SCRIPT}" \
  --input-root "${INPUT_ROOT}" \
  --input-subdir distance_matrices \
  --output-root "${OUTPUT_ROOT}" \
  --datasets "${DATASET}" \
  --max-iter "${FLAMINGO_MAX_ITER:-500}" \
  --tol "${FLAMINGO_TOL:-1e-4}" \
  --mu "${FLAMINGO_MU:-1e-4}" \
  --max-mu "${FLAMINGO_MAX_MU:-1e10}" \
  --rho "${FLAMINGO_RHO:-1.1}" \
  --n-threads "${SLURM_CPUS_PER_TASK:-8}" \
  --selection final \
  --keep-observed

"${PYTHON_BIN}" "${PREP_EVAL}" eval \
  --manifest "${MANIFEST}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --dataset "${DATASET}"
