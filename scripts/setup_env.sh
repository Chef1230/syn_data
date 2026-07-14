#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

cd "${WORKSPACE_ROOT}"
"${PYTHON_BIN}" -m pip install -r "${PROJECT_DIR}/requirements.txt"

echo "Environment dependencies installed for syn_data."