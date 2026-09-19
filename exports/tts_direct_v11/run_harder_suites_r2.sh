#!/usr/bin/env bash
set -euo pipefail
ROOT=/shared/chaolei.liu/SURE-Evolve/exports/tts_direct_v11
python "$ROOT/hardcase_20260918_r2/evaluate.py" pool
python "$ROOT/nonpara_20260918_r2/evaluate.py" pool
