#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


DATASET_KEYS = [
    "math_oai",
    "minerva_math",
    "olympiadbench",
    "aime24",
    "aime25",
    "amc23",
]


def find_latest_eval_root(base_glob: str) -> str:
    roots = sorted(glob.glob(base_glob), key=os.path.getmtime, reverse=True)
    if not roots:
        raise FileNotFoundError(f"No eval root matched: {base_glob}")
    return roots[0]


def parse_stage(name: str):
    # e.g. model__checkpoint-782 or model
    m = re.match(r"^(.*)__checkpoint-(\d+)$", name)
    if m:
        base = m.group(1)
        step = int(m.group(2))
        return base, f"checkpoint-{step}", step, False
    return name, "final", 10**9, True


def load_stage_metrics(eval_root: str):
    model_to_items = defaultdict(list)
    metric_files = glob.glob(os.path.join(eval_root, "outputs", "*", "overall_metrics.json"))
    if not metric_files:
        raise FileNotFoundError(f"No overall_metrics.json found under {eval_root}/outputs")

    for p in metric_files:
        folder = os.path.basename(os.path.dirname(p))
        model, stage_label, stage_order, is_final = parse_stage(folder)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        row = {k: float(data.get(k, {}).get("acc", np.nan)) for k in DATASET_KEYS}
        row["avg"] = float(data.get("avg", {}).get("acc", np.nan))
        model_to_items[model].append(
            {
                "stage_label": stage_label,
                "stage_order": stage_order,
                "is_final": is_final,
                "metrics": row,
                "src": p,
            }
        )
    return model_to_items


def plot_for_model(model: str, items: list, output_dir: str):
    # Plot only checkpoints by default (exclude "final")
    items = [x for x in items if not x["is_final"]]
    items = sorted(items, key=lambda x: (x["stage_order"], 1 if x["is_final"] else 0))
    if not items:
        return None

    x_labels = [x["stage_label"] for x in items]
    x = np.arange(len(x_labels))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(13, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(DATASET_KEYS)))

    for key in DATASET_KEYS:
        y = [i["metrics"].get(key, np.nan) for i in items]
        color = colors[DATASET_KEYS.index(key)]
        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            linestyle="--",
            color=color,
            alpha=0.95,
            label=key,
        )

    y_avg = [i["metrics"].get("avg", np.nan) for i in items]
    plt.plot(
        x,
        y_avg,
        marker="D",
        markersize=7,
        linewidth=3.2,
        linestyle="-",
        color="#111111",
        label="avg",
    )

    plt.xticks(x, x_labels, rotation=25)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Stage")
    plt.title(f"{model} - Stage Metrics (Checkpoints Only)")

    all_vals = []
    for key in DATASET_KEYS + ["avg"]:
        all_vals.extend([i["metrics"].get(key, np.nan) for i in items])
    all_vals = [v for v in all_vals if not np.isnan(v)]
    if all_vals:
        ymin = max(0, min(all_vals) - 2)
        ymax = min(100, max(all_vals) + 3)
        plt.ylim(ymin, ymax)

    plt.grid(alpha=0.35, linestyle=":", linewidth=0.8)
    plt.legend(ncol=4, fontsize=9, frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"{model}_stage_metrics.png")
    plt.savefig(out_png, dpi=180)
    plt.close()
    return out_png


def write_summary_csv(model_to_items, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "stage_metrics_summary.csv")
    headers = ["model", "stage"] + DATASET_KEYS + ["avg"]
    lines = [",".join(headers)]
    for model, items in sorted(model_to_items.items()):
        items = [x for x in items if not x["is_final"]]
        items = sorted(items, key=lambda x: (x["stage_order"], 1 if x["is_final"] else 0))
        for i in items:
            vals = [f"{i['metrics'].get(k, np.nan):.6f}" for k in DATASET_KEYS]
            line = [model, i["stage_label"], *vals, f"{i['metrics'].get('avg', np.nan):.6f}"]
            lines.append(",".join(line))
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_root",
        default=None,
        help="Evaluation root, e.g. /.../eval_full_8gpu_20260214_173854",
    )
    parser.add_argument(
        "--eval_root_glob",
        default="/volume/demo/xlzhuang/zh/test_math/ASFT/eval_full_8gpu_*",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to save figures; default: eval/plots/<eval_root_name>",
    )
    args = parser.parse_args()

    eval_root = args.eval_root or find_latest_eval_root(args.eval_root_glob)
    eval_root_name = os.path.basename(eval_root.rstrip("/"))
    output_dir = args.output_dir or os.path.join("eval", "plots", eval_root_name)

    model_to_items = load_stage_metrics(eval_root)
    generated = []
    for model, items in sorted(model_to_items.items()):
        out_png = plot_for_model(model, items, output_dir)
        if out_png:
            generated.append(out_png)

    summary_csv = write_summary_csv(model_to_items, output_dir)
    print(f"eval_root={eval_root}")
    print(f"output_dir={output_dir}")
    for p in generated:
        print(f"figure={p}")
    print(f"summary={summary_csv}")


if __name__ == "__main__":
    main()
