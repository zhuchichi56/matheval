#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/volume/demo/xlzhuang/zh/miniconda3/bin/conda}"
ENV_NAME="${1:-math11}"
OUT_FILE="${ROOT_DIR}/requirement-latest.txt"

TMP_FILE="$(mktemp)"
cleanup() { rm -f "${TMP_FILE}"; }
trap cleanup EXIT

echo "# Synced from env: ${ENV_NAME}" > "${TMP_FILE}"
echo "# Self-contained latest requirements for one-shot setup" >> "${TMP_FILE}"
echo "" >> "${TMP_FILE}"

PKGS=(
  numpy
  tqdm
  regex
  multiprocess
  python-dateutil
  sympy
  antlr4-python3-runtime
  latex2sympy2
  word2number
  Pebble
  timeout-decorator
  vllm
  transformers
  tokenizers
  torch
  torchaudio
  torchvision
  ray
  datasets
  accelerate
)

for p in "${PKGS[@]}"; do
  ver="$("${CONDA_BIN}" run -n "${ENV_NAME}" python -c "import importlib.metadata as m; print(m.version('${p}'))" 2>/dev/null || true)"
  if [[ -z "${ver}" ]]; then
    echo "[warn] package missing in ${ENV_NAME}: ${p}" >&2
    continue
  fi
  echo "${p}==${ver}" >> "${TMP_FILE}"
done

mv "${TMP_FILE}" "${OUT_FILE}"
echo "[update] wrote ${OUT_FILE}"
