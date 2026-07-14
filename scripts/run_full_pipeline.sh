#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
CONFIG_PATH="${RDB_PRIOR_CONFIG:-${PROJECT_DIR}/configs/default.yaml}"

if [[ $# -gt 0 && "${1}" != --* ]]; then
  CONFIG_PATH="$1"
  shift
fi

export PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}"
cd "${WORKSPACE_ROOT}"
"${PYTHON_BIN}" -u -m syn_data.src.rdb_prior.io.full_pipeline --config "${CONFIG_PATH}" "$@"