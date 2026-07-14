#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

RELBENCH_STATS="${1:-data/statistics/Relbench/schema_graph.json}"
EXISTING_ROOT="${2:-}"
SYNTHETIC_ROOT="${3:-syn_data/outputs/databases/syn_v1}"
OUTPUT_DIR="${4:-syn_data/outputs/analysis/schema_stats}"

export PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}"
cd "${WORKSPACE_ROOT}"

ARGS=(
  --relbench-stats-json "${RELBENCH_STATS}"
  --synthetic-root "${SYNTHETIC_ROOT}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${EXISTING_ROOT}" && "${EXISTING_ROOT}" != "-" ]]; then
  ARGS+=(--existing-root "${EXISTING_ROOT}")
fi

"${PYTHON_BIN}" -m syn_data.src.rdb_prior.analysis.schema_stats "${ARGS[@]}"
