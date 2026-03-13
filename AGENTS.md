# Evaluation Defaults (Must Read First)

Scope: `/volume/demo/xlzhuang/zh/matheval`

For testing models/checkpoints under:
- `/volume/demo/xlzhuang/zh/test_math/ASFT/output`

Machine-specific note:
- "Current machine" means this host has these exact absolute paths and they are accessible:
  - `/volume/demo/xlzhuang/zh/matheval`
  - `/volume/demo/xlzhuang/zh/test_math/ASFT/output`
  - `/volume/demo/xlzhuang/zh/miniconda3/bin/conda` (env: `math12`)
- If any of the paths above do not exist, treat it as another machine and replace all absolute paths before running.

Use these defaults unless user explicitly changes them:
- Conda env: `math12`
- Prompt template: `alpaca`
- `n_sampling`: `16`
- `temperature`: `1`
- `data_name`: `math_oai,minerva_math,olympiadbench,aime24,aime25,amc23`
- `split`: `test`
- `num_test_sample`: `-1` (full dataset)
- GPUs: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`

Canonical command shape:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HOME=/tmp/hf_home \
TOKENIZERS_PARALLELISM=false \
/volume/demo/xlzhuang/zh/miniconda3/bin/conda run -n math12 \
bash sh/eval.sh \
  alpaca \
  <MODEL_OR_CHECKPOINT_PATH> \
  <OUTPUT_DIR> \
  16 \
  1 \
  math_oai,minerva_math,olympiadbench,aime24,aime25,amc23 \
  test \
  -1
```

Background execution rule (must follow):
- Long evaluation jobs must run in background and survive shell/session exit.
- Use `setsid nohup ... &` and record PID immediately with `$!` into a pid file.
- PID is captured after start, not before `nohup`.

Recommended launch pattern:

```bash
RUN_DIR=/volume/demo/xlzhuang/zh/test_math/ASFT/runs
mkdir -p "$RUN_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$RUN_DIR/eval_${TS}.log"
PID_FILE="$RUN_DIR/eval_${TS}.pid"

setsid nohup bash -lc '
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HOME=/tmp/hf_home \
TOKENIZERS_PARALLELISM=false \
/volume/demo/xlzhuang/zh/miniconda3/bin/conda run -n math12 \
bash sh/eval.sh alpaca <MODEL_OR_CHECKPOINT_PATH> <OUTPUT_DIR> 16 1 \
math_oai,minerva_math,olympiadbench,aime24,aime25,amc23 test -1
' >"$LOG_FILE" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
```

Useful checks:
- `ps -fp "$(cat <PID_FILE>)"`
- `tail -f <LOG_FILE>`
