#!/usr/bin/env python3
"""
Cross-run statistical aggregation for Sanctuary Phase 2 sweeps.

Reads all completed run directories matching a pattern, groups by
protocol, and produces:
  - Per-cell means and standard deviations for each metric
  - Pairwise t-tests between protocol cells
  - Cohen's d effect sizes
  - 95% confidence intervals
  - CSV summary at analysis/<sweep_name>/statistical_summary.csv
  - Matplotlib comparison plots at analysis/<sweep_name>/plots/

Usage:
    python metrics/aggregate.py --sweep-name phase2_pilot
    python metrics/aggregate.py --sweep-name phase2_full --runs-dir runs/phase2_full
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

# Attempt matplotlib import (may fail on headless cluster without display)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Metrics to aggregate (dot-separated paths into metrics.json)
METRIC_PATHS = [
    ("misrepresentation.overall_rate", "Misrepresentation Rate"),
    ("allocative_efficiency.ae", "Allocative Efficiency"),
    ("price_cost_margin.mean_pcm", "Price-Cost Margin"),
    ("market_integrity.price_parallelism_index", "Price Parallelism Index"),
    ("market_integrity.markup_correlation", "Markup Correlation"),
    ("market_integrity.exploitation_rate", "Exploitation Rate"),
    ("market_integrity.trust_persistence", "Trust Persistence"),
]


def _extract_metric(metrics: dict, path: str) -> float | None:
    """Extract a nested metric value by dot-separated path."""
    parts = path.split(".")
    val = metrics
    for part in parts:
        if isinstance(val, dict) and part in val:
            val = val[part]
        else:
            return None
    if isinstance(val, (int, float)):
        return float(val)
    return None


def load_run_metrics(run_dir: Path) -> dict | None:
    """Load metrics.json from a run directory, or None if missing/incomplete."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_run_protocol(run_dir: Path) -> str | None:
    """Determine the protocol from a run directory's manifest or config."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            return manifest.get("protocol", None)
        except (json.JSONDecodeError, OSError):
            pass

    config_path = run_dir / "config_used.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            return cfg.get("protocol", {}).get("system", None)
        except (ImportError, OSError):
            pass

    return None


def collect_runs(runs_dir: Path) -> dict[str, list[dict]]:
    """
    Scan runs directory and group metrics by protocol.

    Returns: {protocol_name: [metrics_dict, ...]}
    """
    grouped: dict[str, list[dict]] = {}

    if not runs_dir.exists():
        return grouped

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        metrics = load_run_metrics(run_dir)
        if metrics is None:
            continue

        # Extract protocol from directory name (run_<protocol>_seed<n>)
        dir_name = run_dir.name
        protocol = None
        if dir_name.startswith("run_") and "_seed" in dir_name:
            # e.g. run_ebay_feedback_seed3 -> ebay_feedback
            after_run = dir_name[4:]  # strip "run_"
            seed_idx = after_run.rfind("_seed")
            if seed_idx > 0:
                protocol = after_run[:seed_idx]

        if protocol is None:
            protocol = load_run_protocol(run_dir) or "unknown"

        grouped.setdefault(protocol, []).append(metrics)

    return grouped


def compute_statistics(
    grouped: dict[str, list[dict]],
) -> tuple[list[dict], list[dict]]:
    """
    Compute per-cell statistics and pairwise comparisons.

    Returns:
      - cell_stats: list of dicts with per-protocol-metric stats
      - pairwise: list of dicts with pairwise t-test results
    """
    cell_stats: list[dict] = []
    protocol_values: dict[str, dict[str, list[float]]] = {}

    # Per-cell statistics
    for protocol, runs in sorted(grouped.items()):
        protocol_values[protocol] = {}
        for metric_path, metric_label in METRIC_PATHS:
            values = []
            for m in runs:
                v = _extract_metric(m, metric_path)
                if v is not None:
                    values.append(v)

            protocol_values[protocol][metric_path] = values

            if len(values) >= 2:
                arr = np.array(values)
                mean = float(np.mean(arr))
                std = float(np.std(arr, ddof=1))
                n = len(values)
                se = std / np.sqrt(n)
                ci_low = mean - 1.96 * se
                ci_high = mean + 1.96 * se
            elif len(values) == 1:
                mean = values[0]
                std = 0.0
                n = 1
                ci_low = mean
                ci_high = mean
            else:
                mean = std = ci_low = ci_high = float("nan")
                n = 0

            cell_stats.append({
                "protocol": protocol,
                "metric": metric_label,
                "metric_path": metric_path,
                "n": n,
                "mean": mean,
                "std": std,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
            })

    # Pairwise comparisons
    pairwise: list[dict] = []
    protocols = sorted(grouped.keys())
    for p1, p2 in combinations(protocols, 2):
        for metric_path, metric_label in METRIC_PATHS:
            v1 = protocol_values.get(p1, {}).get(metric_path, [])
            v2 = protocol_values.get(p2, {}).get(metric_path, [])

            if len(v1) < 2 or len(v2) < 2:
                continue

            a1, a2 = np.array(v1), np.array(v2)
            t_stat, p_value = stats.ttest_ind(a1, a2, equal_var=False)

            # Cohen's d
            pooled_std = np.sqrt(
                ((len(v1) - 1) * np.var(a1, ddof=1) +
                 (len(v2) - 1) * np.var(a2, ddof=1)) /
                (len(v1) + len(v2) - 2)
            )
            cohens_d = (np.mean(a1) - np.mean(a2)) / pooled_std if pooled_std > 0 else 0.0

            pairwise.append({
                "protocol_1": p1,
                "protocol_2": p2,
                "metric": metric_label,
                "metric_path": metric_path,
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "cohens_d": float(cohens_d),
                "n1": len(v1),
                "n2": len(v2),
                "mean_1": float(np.mean(a1)),
                "mean_2": float(np.mean(a2)),
            })

    return cell_stats, pairwise


def write_csv(
    cell_stats: list[dict],
    pairwise: list[dict],
    output_path: Path,
) -> None:
    """Write statistical summary CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Section 1: Per-cell statistics
        writer.writerow(["=== Per-Cell Statistics ==="])
        writer.writerow(["protocol", "metric", "n", "mean", "std",
                         "ci_95_low", "ci_95_high"])
        for row in cell_stats:
            writer.writerow([
                row["protocol"], row["metric"], row["n"],
                f"{row['mean']:.4f}", f"{row['std']:.4f}",
                f"{row['ci_95_low']:.4f}", f"{row['ci_95_high']:.4f}",
            ])

        writer.writerow([])

        # Section 2: Pairwise comparisons
        writer.writerow(["=== Pairwise Comparisons ==="])
        writer.writerow(["protocol_1", "protocol_2", "metric",
                         "t_statistic", "p_value", "cohens_d",
                         "n1", "n2", "mean_1", "mean_2"])
        for row in pairwise:
            writer.writerow([
                row["protocol_1"], row["protocol_2"], row["metric"],
                f"{row['t_statistic']:.4f}", f"{row['p_value']:.6f}",
                f"{row['cohens_d']:.4f}",
                row["n1"], row["n2"],
                f"{row['mean_1']:.4f}", f"{row['mean_2']:.4f}",
            ])


def generate_plots(
    cell_stats: list[dict],
    grouped: dict[str, list[dict]],
    plots_dir: Path,
) -> None:
    """Generate matplotlib comparison plots."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plots.", file=sys.stderr)
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    for metric_path, metric_label in METRIC_PATHS:
        protocols = []
        means = []
        stds = []
        ci_lows = []
        ci_highs = []

        for row in cell_stats:
            if row["metric_path"] == metric_path and row["n"] > 0:
                protocols.append(row["protocol"])
                means.append(row["mean"])
                stds.append(row["std"])
                ci_lows.append(row["ci_95_low"])
                ci_highs.append(row["ci_95_high"])

        if not protocols:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(protocols))
        errors = [
            [m - cl for m, cl in zip(means, ci_lows)],
            [ch - m for m, ch in zip(means, ci_highs)],
        ]
        bars = ax.bar(x, means, yerr=errors, capsize=5, alpha=0.8,
                       color=plt.cm.Set2(np.linspace(0, 1, len(protocols))))
        ax.set_xticks(x)
        ax.set_xticklabels(protocols, rotation=45, ha="right")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} by Protocol (95% CI)")

        # Add n labels
        for i, (bar, n) in enumerate(zip(bars, [r["n"] for r in cell_stats if r["metric_path"] == metric_path and r["n"] > 0])):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"n={n}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        safe_name = metric_label.lower().replace(" ", "_").replace("-", "_")
        plt.savefig(plots_dir / f"{safe_name}.png", dpi=150)
        plt.close()

    print(f"Plots saved to: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate metrics across Sanctuary sweep runs.",
    )
    parser.add_argument(
        "--sweep-name",
        required=True,
        help="Name of the sweep (used for directory naming)",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Path to runs directory (default: runs/<sweep_name>)",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir) if args.runs_dir else Path(f"runs/{args.sweep_name}")

    print(f"Scanning: {runs_dir}")
    grouped = collect_runs(runs_dir)

    if not grouped:
        print(f"No completed runs found in {runs_dir}", file=sys.stderr)
        sys.exit(1)

    total_runs = sum(len(v) for v in grouped.values())
    print(f"Found {total_runs} completed runs across {len(grouped)} protocols:")
    for protocol, runs in sorted(grouped.items()):
        print(f"  {protocol}: {len(runs)} runs")
    print()

    cell_stats, pairwise = compute_statistics(grouped)

    # Write CSV
    analysis_dir = Path(f"analysis/{args.sweep_name}")
    csv_path = analysis_dir / "statistical_summary.csv"
    write_csv(cell_stats, pairwise, csv_path)
    print(f"CSV summary: {csv_path}")

    # Generate plots
    plots_dir = analysis_dir / "plots"
    generate_plots(cell_stats, grouped, plots_dir)

    # Print summary table
    print("\n=== Summary ===")
    current_protocol = None
    for row in cell_stats:
        if row["protocol"] != current_protocol:
            current_protocol = row["protocol"]
            print(f"\n{current_protocol} (n={row['n']}):")
        if row["n"] > 0:
            print(f"  {row['metric']:30s}  {row['mean']:.4f} +/- {row['std']:.4f}  "
                  f"[{row['ci_95_low']:.4f}, {row['ci_95_high']:.4f}]")

    if pairwise:
        print("\n=== Significant Comparisons (p < 0.05) ===")
        sig = [p for p in pairwise if p["p_value"] < 0.05]
        if sig:
            for p in sig:
                print(f"  {p['protocol_1']} vs {p['protocol_2']}: "
                      f"{p['metric']} (p={p['p_value']:.4f}, d={p['cohens_d']:.2f})")
        else:
            print("  None")


if __name__ == "__main__":
    main()
