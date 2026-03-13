#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-/volume/demo/xlzhuang/zh/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-math11}"

PROMPT_TYPE="${1:-mathstral}"
MODEL_NAME_OR_PATH="${2:-}"
OUTPUT_DIR="${3:-}"
N_SAMPLING="${4:-1}"
TEMPERATURE="${5:-0}"
DATA_NAME="${6:-aime24,amc23}"
SPLIT="${7:-test}"
NUM_TEST_SAMPLE="${8:--1}"
STRICT_VLLM="${STRICT_VLLM:-1}"  # 1: fail fast, 0: allow manual fallback

if [[ -z "${MODEL_NAME_OR_PATH}" ]]; then
  echo "Usage: $0 PROMPT_TYPE MODEL_PATH [OUTPUT_DIR] [N_SAMPLING] [TEMPERATURE] [DATA_NAME] [SPLIT] [NUM_TEST_SAMPLE]" >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${ROOT_DIR}/eval/matheval/outputs/$(basename "${MODEL_NAME_OR_PATH}")-latest"
fi

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

RUN_VLLM=(
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -u inference.py
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --data_names "${DATA_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --split "${SPLIT}"
  --prompt_type "${PROMPT_TYPE}"
  --num_test_sample "${NUM_TEST_SAMPLE}"
  --seed 0
  --temperature "${TEMPERATURE}"
  --n_sampling "${N_SAMPLING}"
  --top_p 1
  --start 0
  --end -1
  --use_vllm
  --use_safetensors
  --save_rank_outputs
)

echo "[run] vLLM inference start"
if ! "${RUN_VLLM[@]}"; then
  echo "[run] vLLM inference failed"
  if [[ "${STRICT_VLLM}" == "1" ]]; then
    echo "[run] STRICT_VLLM=1, stop without HF fallback" >&2
    exit 2
  fi
  echo "[run] STRICT_VLLM=0, you can run HF path manually if needed" >&2
  exit 3
fi

echo "[run] evaluation start"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -u eval_only.py \
  --data_names "${DATA_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --prompt_type "${PROMPT_TYPE}" \
  --overwrite

echo "[run] done: ${OUTPUT_DIR}"
