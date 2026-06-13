#!/usr/bin/env bash
#SBATCH -J rawdist_lrtc_summary
#SBATCH -A pi_limin_r
#SBATCH -p cpuQ
#SBATCH -q cpuq
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData/output_distance/logs/slurm_%x_%j.out
#SBATCH --error=/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData/output_distance/logs/slurm_%x_%j.err

set -euo pipefail

export LC_ALL=C
export LANG=C

BASE_DIR="/public/home/hpc254701055/2_projects/10_schicdiff/1_scHiC/5_baseline/9_FLAMINGO/2_FLAMINGOData"
PYTHON_BIN="${PYTHON_BIN:-/public/home/hpc254701055/micromamba/envs/unicorn_and_flamingo_env/bin/python}"

mkdir -p "${BASE_DIR}/output_distance/logs"

"${PYTHON_BIN}" "${BASE_DIR}/scripts/raw_distance_lrtc_prep_eval.py" combine \
  --manifest "${BASE_DIR}/input_distance/manifest.tsv" \
  --output-root "${BASE_DIR}/output_distance"
