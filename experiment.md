# ASFT Math Eval Experiments (整理于 2026-02-15)

## 1) 实验配置（`eval_full_8gpu_20260214_173854`）

- 工作目录: `/volume/demo/xlzhuang/zh/matheval`
- 模型根目录: `/volume/demo/xlzhuang/zh/test_math/ASFT/output`
- 评测脚本: `bash sh/eval.sh`
- Prompt template: `alpaca`
- `n_sampling`: `16`
- `temperature`: `1`
- `data_name`: `math_oai,minerva_math,olympiadbench,aime24,aime25,amc23`
- `split`: `test`
- `num_test_sample`: `-1`
- GPU: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`

## 2) 跑通实验（仅保留非-smoke）

| Run | Status | 时间 |
|---|---|---|
| `eval_full_8gpu_20260214_173854` | Completed (`15/15 PASS`) | `2026-02-14 17:38:54` -> `2026-02-14 20:44:11` |

## 3) Full Run 结果排名（15 个 target）

说明: 下表字段均来自 `eval_full_8gpu_20260214_173854/outputs/*/overall_metrics.json`。

| Rank | Target | overall.acc | overall.mean_acc | math_oai | minerva_math | olympiadbench | aime24 | aime25 | amc23 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `sft_mathcfg_ep4_olmo3_7b_gbs256_len2048__checkpoint-1173` | 17.17 | 14.75 | 41.60 | 15.10 | 13.80 | 0.00 | 0.00 | 32.50 |
| 2 | `sft_mathcfg_ep4_olmo3_7b_gbs256_len2048` | 15.97 | 15.02 | 44.80 | 14.00 | 14.50 | 0.00 | 0.00 | 22.50 |
| 3 | `sft_mathcfg_ep4_olmo3_7b_gbs256_len2048__checkpoint-1564` | 14.38 | 15.09 | 43.40 | 12.50 | 12.10 | 0.00 | 3.30 | 15.00 |
| 4 | `sft_mathcfg_ep4_olmo3_7b_gbs256_len2048__checkpoint-782` | 14.15 | 14.26 | 43.60 | 10.30 | 11.00 | 0.00 | 0.00 | 20.00 |
| 5 | `sft_mathcfg_ep4_olmo3_7b_gbs256_len2048__checkpoint-391` | 13.75 | 13.08 | 40.20 | 9.90 | 9.90 | 0.00 | 0.00 | 22.50 |
| 6 | `sft_mathcfg_ep4_llama31_8b_gbs256_len2048__checkpoint-1173` | 10.43 | 9.25 | 28.80 | 11.40 | 7.40 | 0.00 | 0.00 | 15.00 |
| 7 | `sft_mathcfg_ep4_llama31_8b_gbs256_len2048__checkpoint-1564` | 8.90 | 9.32 | 29.60 | 9.20 | 7.10 | 0.00 | 0.00 | 7.50 |
| 8 | `sft_mathcfg_ep4_llama31_8b_gbs256_len2048` | 8.83 | 9.10 | 29.40 | 9.90 | 6.20 | 0.00 | 0.00 | 7.50 |
| 9 | `sft_mathcfg_ep4_llama31_8b_gbs256_len2048__checkpoint-782` | 7.42 | 8.28 | 25.60 | 9.60 | 4.30 | 0.00 | 0.00 | 5.00 |
| 10 | `sft_mathcfg_ep4_llama31_8b_gbs256_len2048__checkpoint-391` | 4.85 | 5.70 | 18.00 | 7.00 | 4.10 | 0.00 | 0.00 | 0.00 |
| 11 | `sft_mathcfg_ep4_llama2_7b_gbs256_len2048__checkpoint-1564` | 2.53 | 2.02 | 4.80 | 1.10 | 1.80 | 0.00 | 0.00 | 7.50 |
| 12 | `sft_mathcfg_ep4_llama2_7b_gbs256_len2048` | 2.50 | 1.82 | 5.80 | 0.70 | 1.00 | 0.00 | 0.00 | 7.50 |
| 13 | `sft_mathcfg_ep4_llama2_7b_gbs256_len2048__checkpoint-391` | 2.15 | 1.29 | 2.60 | 1.80 | 1.00 | 0.00 | 0.00 | 7.50 |
| 14 | `sft_mathcfg_ep4_llama2_7b_gbs256_len2048__checkpoint-1173` | 1.60 | 1.61 | 5.00 | 1.50 | 0.60 | 0.00 | 0.00 | 2.50 |
| 15 | `sft_mathcfg_ep4_llama2_7b_gbs256_len2048__checkpoint-782` | 1.53 | 1.45 | 2.80 | 0.70 | 0.70 | 0.00 | 0.00 | 5.00 |

## 4) 聚合视角（Full Run）

### 4.1 按模型家族聚合

| Family | Count | mean(overall.acc) | best | worst |
|---|---:|---:|---:|---:|
| llama2_7b | 5 | 2.06 | 2.53 | 1.53 |
| llama31_8b | 5 | 8.09 | 10.43 | 4.85 |
| olmo3_7b | 5 | 15.08 | 17.17 | 13.75 |

### 4.2 按训练阶段聚合

| Checkpoint | Count | mean(overall.acc) | best | worst |
|---|---:|---:|---:|---:|
| base | 3 | 9.10 | 15.97 | 2.50 |
| checkpoint-391 | 3 | 6.92 | 13.75 | 2.15 |
| checkpoint-782 | 3 | 7.70 | 14.15 | 1.53 |
| checkpoint-1173 | 3 | 9.73 | 17.17 | 1.60 |
| checkpoint-1564 | 3 | 8.61 | 14.38 | 2.53 |

## 5) Base 成绩（单独）

### 5.1 ASFT 三个 base（来自 Full Run）

| Model (base) | overall.acc | math_oai | minerva_math | olympiadbench | aime24 | aime25 | amc23 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sft_mathcfg_ep4_llama2_7b_gbs256_len2048` | 2.50 | 5.80 | 0.70 | 1.00 | 0.00 | 0.00 | 7.50 |
| `sft_mathcfg_ep4_llama31_8b_gbs256_len2048` | 8.83 | 29.40 | 9.90 | 6.20 | 0.00 | 0.00 | 7.50 |
| `sft_mathcfg_ep4_olmo3_7b_gbs256_len2048` | 15.97 | 44.80 | 14.00 | 14.50 | 0.00 | 0.00 | 22.50 |

### 5.2 历史通用 Base 模型（非-smoke）

| Model | Prompt | overall.acc | math_oai | minerva_math | olympiadbench | aime24 | aime25 | amc23 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-7B` | default | 23.65 | 54.00 | 12.10 | 22.50 | 10.00 | 3.30 | 40.00 |
| `Qwen3-8B-Base` | default | 25.05 | 58.00 | 11.80 | 24.70 | 10.00 | 3.30 | 42.50 |
| `Qwen2.5-7B` | alpaca | 10.35 | 26.60 | 9.20 | 8.00 | 3.30 | 0.00 | 15.00 |
| `Qwen3-8B-Base` | alpaca | 23.57 | 44.00 | 16.90 | 23.00 | 10.00 | 10.00 | 37.50 |

## 6) 数据来源路径（本文件使用）

- `/volume/demo/xlzhuang/zh/test_math/ASFT/eval_full_8gpu_20260214_173854/status.tsv`
- `/volume/demo/xlzhuang/zh/test_math/ASFT/eval_full_8gpu_20260214_173854/outputs/*/overall_metrics.json`
- `/volume/demo/xlzhuang/zh/matheval/eval/matheval/outputs/Qwen2.5-7B/overall_metrics.json`
- `/volume/demo/xlzhuang/zh/matheval/eval/matheval/outputs/Qwen3-8B-Base/overall_metrics.json`
- `/volume/demo/xlzhuang/zh/matheval/eval/matheval/outputs_alpaca/Qwen2.5-7B/overall_metrics.json`
- `/volume/demo/xlzhuang/zh/matheval/eval/matheval/outputs_alpaca/Qwen3-8B-Base/overall_metrics.json`
