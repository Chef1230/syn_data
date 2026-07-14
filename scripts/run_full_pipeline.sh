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
# File logging is configured by the YAML logging section. It can be overridden
# with --log-file/--log-level or RDB_PRIOR_LOG_FILE/RDB_PRIOR_LOG_LEVEL.
# DFS continuation flags are passed through unchanged, for example:
#   --resume-dfs --dbinfer-root outputs/dbinfer_for_dfs/syn_v1
"${PYTHON_BIN}" -u -m syn_data.src.rdb_prior.io.full_pipeline --config "${CONFIG_PATH}" "$@"
