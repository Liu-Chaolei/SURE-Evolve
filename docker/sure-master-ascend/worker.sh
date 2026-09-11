#!/usr/bin/env bash
# Run only within a genuine Slurm step; the cluster wrapper owns device mapping.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
: "${SURE_EVOLVE_ROOT:?Set SURE_EVOLVE_ROOT}"
: "${SURE_NPU_IMAGE:?Use a registry image pinned by digest}"
case "$SURE_NPU_IMAGE" in *@sha256:*) ;; *) echo 'Image digest required' >&2; exit 2 ;; esac
exec sudo -n slurm-docker-run --pull missing --shm-size 128g \
  --mount /shared/chaolei.liu:/shared/chaolei.liu:rw \
  --workdir "$SURE_EVOLVE_ROOT" --env "PYTHONPATH=$SURE_EVOLVE_ROOT" \
  "$SURE_NPU_IMAGE" "$@"
