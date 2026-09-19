#!/usr/bin/env bash
set -euo pipefail
ROOT=/shared/chaolei.liu/SURE-Evolve/exports/tts_direct_v11
PY=/shared/chaolei.liu/data/sure_xlab_runtime312/bin/python
$PY "$ROOT/hardcase_20260918/evaluate.py" pool
$PY "$ROOT/nonpara_20260918/evaluate.py" pool
