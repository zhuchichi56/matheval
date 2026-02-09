#!/bin/bash
# conda activate /volume/pt-train/users/wzhang/ghchen/laip/miniconda3/envs/evalplus

set -e

# # === 自动激活 conda 环境 ===
# CONDA_ENV="/volume/pt-train/users/wzhang/ghchen/laip/miniconda3/envs/evalplus"

# # 1) 加载 conda 基础脚本
# source "/volume/pt-train/users/wzhang/ghchen/laip/miniconda3/etc/profile.d/conda.sh"

# # 2) 激活指定环境
# conda activate "$CONDA_ENV"

# echo ">>> 已自动激活环境: $CONDA_ENV"

PROMPT_TYPE="alpaca"
N_SAMPLING=16
TEMPERATURE=1
MODEL_DIRS=(
"/mnt/msranlphot_intern/zhuhe/models/Qwen2.5-7B-Instruct"
)

for MODEL_PATH in "${MODEL_DIRS[@]}"; do
    # 自动生成输出目录
    NAME=$(basename "$MODEL_PATH")
    OUTPUT_DIR="eval/matheval/outputs/${NAME}"

    echo "=============================================="
    echo "Evaluating MODEL: $MODEL_PATH"
    echo "Saving to OUTPUT: $OUTPUT_DIR"
    echo "=============================================="

    bash sh/eval.sh \
        $PROMPT_TYPE \
        "$MODEL_PATH" \
        "$OUTPUT_DIR" \
        $N_SAMPLING \
        $TEMPERATURE
done



