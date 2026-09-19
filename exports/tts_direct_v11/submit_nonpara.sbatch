#!/usr/bin/env bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:ascend910b3:8
#SBATCH --time=2-00:00:00
#SBATCH -J tts-nonpara-eval
set -euo pipefail
sudo -n slurm-docker-run --pull missing --network host --shm-size 64g \
 --mount /shared/chaolei.liu:/shared/chaolei.liu:ro \
 --mount /shared/chaolei.liu/SURE-Evolve/exports/tts_direct_v11:/shared/chaolei.liu/SURE-Evolve/exports/tts_direct_v11:rw \
 --workdir /shared/chaolei.liu/SURE-Evolve/exports/tts_direct_v11 \
 registry.cluster.local:5000/users/chaolei-liu/sure-f5tts@sha256:a6d87e273b0dd20174a6e30806b18ae8752a4a841ad35f5a81ffe608caeb2dfe \
 python /shared/chaolei.liu/SURE-Evolve/exports/tts_direct_v11/nonpara_20260918_r2/evaluate.py pool
