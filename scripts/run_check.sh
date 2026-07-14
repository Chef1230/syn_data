#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

export PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}"
cd "${WORKSPACE_ROOT}"
"${PYTHON_BIN}" -m compileall -q syn_data/src/rdb_prior syn_data/tests
"${PYTHON_BIN}" -m unittest discover syn_data/tests