#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/volume/demo/xlzhuang/zh/miniconda3/bin/conda}"
ENV_NAME="${1:-math11}"
PY_VER="${PY_VER:-3.10}"

echo "[setup] root=${ROOT_DIR}"
echo "[setup] env=${ENV_NAME}, python=${PY_VER}"

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "[setup] conda not found: ${CONDA_BIN}" >&2
  exit 1
fi

if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] env ${ENV_NAME} already exists, skip create"
else
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" "python=${PY_VER}"
fi

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -U pip setuptools wheel
"${CONDA_BIN}" run -n "${ENV_NAME}" pip install -r "${ROOT_DIR}/requirement-latest.txt"

echo "[setup] verify imports"
"${CONDA_BIN}" run -n "${ENV_NAME}" python - <<'PY'
import vllm, transformers, torch, ray
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
print("torch", torch.__version__)
print("ray", ray.__version__)
PY

echo "[setup] done"
