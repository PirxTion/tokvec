"""Visualize training metrics from CSV log files.

Usage:
    uv run python visualize.py logs/metrics_*.csv
    uv run python visualize.py logs/metrics_20260312_143000.csv --save plot.png
    uv run python visualize.py run1.csv run2.csv          # overlay multiple runs
"""
from __future__ import annotations
import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: str) -> dict[str, np.ndarray]:
    """Load a metrics CSV into dict of arrays. Blank fields become NaN."""
    rows: dict[str, list[float]] = {
        "step": [], "epoch": [], "loss": [], "lr": [],
        "grad_norm": [], "analogy_acc": [], "elapsed": [],
    }
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in rows:
                val = row.get(key, "").strip()
                rows[key].append(float(val) if val else float("nan"))
    return {k: np.array(v) for k, v in rows.items()}


def plot_single(metrics: dict[str, np.ndarray], label: str, axes: list) -> None:
    """Plot one run's metrics onto the given axes."""
    steps = metrics["step"]

    # Training metrics: rows where loss is not NaN
    train_mask = ~np.isnan(metrics["loss"])
    train_steps = steps[train_mask]

    # Loss
    ax = axes[0]
    ax.plot(train_steps, metrics["loss"][train_mask], alpha=0.3, linewidth=0.5, label=f"{label} (raw)")
    # Smoothed loss (exponential moving average)
    loss_raw = metrics["loss"][train_mask]
    if len(loss_raw) > 1:
        alpha = 0.99
        smoothed = np.zeros_like(loss_raw)
        smoothed[0] = loss_raw[0]
        for i in range(1, len(loss_raw)):
            smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * loss_raw[i]
        ax.plot(train_steps, smoothed, linewidth=1.5, label=f"{label} (EMA)")

    # Learning rate
    ax = axes[1]
    ax.plot(train_steps, metrics["lr"][train_mask], linewidth=1.5, label=label)

    # Gradient norm
    ax = axes[2]
    ax.plot(train_steps, metrics["grad_norm"][train_mask], alpha=0.4, linewidth=0.5, label=f"{label} (raw)")
    gnorm = metrics["grad_norm"][train_mask]
    if len(gnorm) > 1:
        smoothed_g = np.zeros_like(gnorm)
        smoothed_g[0] = gnorm[0]
        for i in range(1, len(gnorm)):
            smoothed_g[i] = alpha * smoothed_g[i - 1] + (1 - alpha) * gnorm[i]
        ax.plot(train_steps, smoothed_g, linewidth=1.5, label=f"{label} (EMA)")

    # Analogy accuracy (eval rows where analogy_acc is not NaN)
    eval_mask = ~np.isnan(metrics["analogy_acc"])
    if eval_mask.any():
        ax = axes[3]
        ax.plot(steps[eval_mask], metrics["analogy_acc"][eval_mask] * 100,
                "o-", linewidth=1.5, markersize=4, label=label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize word2vec training metrics")
    parser.add_argument("csv_files", nargs="+", help="One or more metrics CSV files")
    parser.add_argument("--save", type=str, default="", help="Save figure to path instead of showing")
    args = parser.parse_args()

    has_eval = False
    for path in args.csv_files:
        m = load_metrics(path)
        if (~np.isnan(m["analogy_acc"])).any():
            has_eval = True
            break

    n_plots = 4 if has_eval else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots), sharex=True)
    axes = list(axes)

    for path in args.csv_files:
        label = os.path.splitext(os.path.basename(path))[0]
        metrics = load_metrics(path)
        plot_single(metrics, label, axes)

    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Gradient Norm")
    axes[2].set_title("Gradient Norm")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    if has_eval:
        axes[3].set_ylabel("Accuracy (%)")
        axes[3].set_title("Analogy Accuracy")
        axes[3].legend(fontsize=8)
        axes[3].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
