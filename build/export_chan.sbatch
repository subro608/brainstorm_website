#!/bin/bash
#SBATCH --job-name=web_chan
#SBATCH --account=torch_pr_60_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/sd5963/AlphaBrain/Alphabrain_staging/website_lfpformer_2026-11-05/build/logs/chan_%A_%a.out
#SBATCH --error=/scratch/sd5963/AlphaBrain/Alphabrain_staging/website_lfpformer_2026-11-05/build/logs/chan_%A_%a.err
# Website export: per-channel seed-1 full-recording predictions, one Fig 3c method per array task.
set -euo pipefail
W=/scratch/sd5963/AlphaBrain/Alphabrain_staging/website_lfpformer_2026-11-05
KEYS=(ours_lin ours_tf fus_lin fus_tf manip vit anynet histgbm knn_band knn_psd ridge_band ridge_psd pcaridge_psd rf_band svm_band)
KEY=${KEYS[$SLURM_ARRAY_TASK_ID]}
PL_ROOT=/scratch/pl2820/Alphabrain_staging; PL_BENCH=$PL_ROOT/Lfp2vec_benchmarks
SING_ENV="PYTHONPATH=$PL_BENCH:$PL_ROOT,PB_BENCH=$PL_BENCH,HF_HUB_OFFLINE=1,PYTHONUNBUFFERED=1,OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,MKL_NUM_THREADS=2,MPLBACKEND=Agg"
echo "key: $KEY"
singularity exec --env "$SING_ENV" /scratch/sd5963/containers/w2v2_cuda_cu128_ibl.sif \
  python3 $W/build/export_chan.py "$KEY"
echo "===== DONE $KEY"
