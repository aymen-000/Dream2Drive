"""Plot inference-ablation results from results.json, summary.csv, or runs.csv."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito colorblind-safe palette, assigned consistently by variant order.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
           "#56B4E9", "#F0E442", "#000000"]


def combine_summaries(map_summaries: dict) -> dict:
    episodes = sum(int(m.get("episodes") or 0) for m in map_summaries.values())
    completed = sum(int(m.get("completed_episodes") or 0) for m in map_summaries.values())
    weighted_times = []
    for metrics in map_summaries.values():
        mean = metrics.get("successful_lap_time_mean_sec")
        count = int(metrics.get("completed_episodes") or 0)
        if mean is not None and count:
            weighted_times.extend([float(mean)] * count)
    return {
        "episodes": episodes,
        "completed_episodes": completed,
        "success_rate": completed / episodes if episodes else None,
        "successful_lap_time_mean_sec": float(np.mean(weighted_times)) if weighted_times else None,
        "successful_lap_time_std_sec": float(np.std(weighted_times, ddof=1))
        if len(weighted_times) > 1 else 0.0 if weighted_times else None,
    }


def load_json(path: str):
    with open(path) as f:
        report = json.load(f)
    if "summary_by_variant" in report:
        return report["summary_by_variant"]
    if "runs" in report:
        grouped = defaultdict(list)
        runs = report["runs"]
        # Both ablation-run JSON files and evaluate_all JSON use dictionaries
        # of lists for their per-episode records.
        if isinstance(runs, dict):
            for rows in runs.values():
                for row in rows:
                    variant = row.get("variant", row.get("method", "unknown"))
                    grouped[variant].append(row)
        else:
            for row in runs:
                grouped[row.get("variant", row.get("method", "unknown"))].append(row)
        result = {}
        for variant, rows in grouped.items():
            successes = [r for r in rows if bool(r.get("success", False))]
            times = [float(t) for row in successes for t in row.get("lap_times_sec", [])]
            result[variant] = {
                "episodes": len(rows), "completed_episodes": len(successes),
                "success_rate": len(successes) / len(rows) if rows else None,
                "successful_lap_time_mean_sec": float(np.mean(times)) if times else None,
                "successful_lap_time_std_sec": (float(np.std(times, ddof=1))
                                                  if len(times) > 1 else 0.0 if times else None),
            }
        if result:
            return result
    if "summary_by_variant_and_map" in report:
        return {variant: combine_summaries(maps)
                for variant, maps in report["summary_by_variant_and_map"].items()}
    raise ValueError("JSON does not contain summary_by_variant_and_map")


def load_summary_csv(path: str):
    grouped = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            metrics = {}
            for key in ("episodes", "completed_episodes", "success_rate",
                        "successful_lap_time_mean_sec", "successful_lap_time_std_sec",
                        "collision_rate"):
                value = row.get(key, "")
                metrics[key] = None if value in (None, "") else float(value)
            grouped[row["variant"]] = metrics
    return grouped


def load_runs_csv(path: str):
    runs = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["success"] = str(row["success"]).lower() in ("1", "true", "yes")
            try:
                row["lap_times_sec"] = json.loads(row.get("lap_times_sec", "[]"))
            except json.JSONDecodeError:
                row["lap_times_sec"] = []
            runs[row["variant"]].append(row)

    grouped = {}
    for variant, rows in runs.items():
        successes = [r for r in rows if r["success"]]
        lap_times = [float(t) for r in successes for t in r["lap_times_sec"]]
        grouped[variant] = {
            "episodes": len(rows),
            "completed_episodes": len(successes),
            "success_rate": len(successes) / len(rows) if rows else None,
            "successful_lap_time_mean_sec": float(np.mean(lap_times)) if lap_times else None,
            "successful_lap_time_std_sec": (float(np.std(lap_times, ddof=1))
                                               if len(lap_times) > 1 else 0.0 if lap_times else None),
            "collision_rate": float(np.mean([
                str(r.get("collision", "false")).lower() in ("1", "true", "yes")
                for r in rows
            ])) if rows else None,
        }
    return dict(grouped)


def load_summary(path: str):
    extension = os.path.splitext(path)[1].lower()
    if extension == ".json":
        return load_json(path)
    if extension == ".csv":
        with open(path, newline="") as f:
            columns = set(next(csv.reader(f), []))
        if {"variant", "map", "success_rate"}.issubset(columns):
            return load_summary_csv(path)
        if {"variant", "map", "success"}.issubset(columns):
            return load_runs_csv(path)
    raise ValueError("Input must be results.json, summary.csv, or runs.csv")


def wilson_interval(successes: int, episodes: int, z: float = 1.96):
    if episodes == 0:
        return 0.0, 0.0
    p = successes / episodes
    den = 1.0 + z * z / episodes
    center = (p + z * z / (2.0 * episodes)) / den
    half = z * np.sqrt(p * (1.0 - p) / episodes + z * z / (4.0 * episodes**2)) / den
    return max(0.0, center - half), min(1.0, center + half)


def plot_metric(summary, variants, metric, ylabel, title, out_path,
                error_mode: str):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })
    x = np.arange(len(variants))
    vals, lower, upper = [], [], []
    for variant in variants:
        metrics = summary[variant]
        value = metrics.get(metric)
        vals.append(np.nan if value is None else float(value))
        if error_mode == "wilson" and value is not None:
            lo, hi = wilson_interval(int(metrics.get("completed_episodes") or 0),
                                     int(metrics.get("episodes") or 0))
            lower.append(float(value) - lo)
            upper.append(hi - float(value))
        elif error_mode == "std" and value is not None:
            sd = float(metrics.get("successful_lap_time_std_sec") or 0.0)
            lower.append(sd)
            upper.append(sd)
        else:
            lower.append(0.0)
            upper.append(0.0)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    bars = ax.bar(x, vals, width=0.68,
                  color=[PALETTE[i % len(PALETTE)] for i in range(len(variants))],
                  edgecolor="#333333", linewidth=0.5,
                  yerr=np.asarray([lower, upper]),
                  error_kw={"elinewidth": 0.9, "capsize": 2.5, "capthick": 0.9}, zorder=3)
    ax.set_xticks(x, variants, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if error_mode == "wilson":
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
        for bar, variant in zip(bars, variants):
            metrics = summary[variant]
            label = f"{int(metrics.get('completed_episodes') or 0)}/{int(metrics.get('episodes') or 0)}"
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.splitext(out_path)[0] + "." + ext,
                    dpi=300 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="Ablation results.json, summary.csv, or runs.csv")
    parser.add_argument("--out_dir", default=None,
                        help="Output folder (default: <input directory>/plots)")
    args = parser.parse_args()

    summary = load_summary(args.input)
    variants = list(summary)
    if not variants:
        raise ValueError("No variant results found in input")
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.input)), "plots")
    os.makedirs(out_dir, exist_ok=True)
    aggregate_path = os.path.join(out_dir, "aggregate_summary.csv")
    with open(aggregate_path, "w", newline="") as f:
        fields = ["variant", "episodes", "completed_episodes", "completed_over_episodes",
                  "success_rate", "successful_laps", "successful_lap_time_mean_sec",
                  "successful_lap_time_std_sec"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for variant, metrics in summary.items():
            episodes = int(metrics.get("episodes") or 0)
            completed = int(metrics.get("completed_episodes") or 0)
            writer.writerow({
                "variant": variant,
                "episodes": episodes,
                "completed_episodes": completed,
                "completed_over_episodes": f"{completed}/{episodes}",
                "success_rate": metrics.get("success_rate"),
                "successful_laps": metrics.get("successful_laps", completed),
                "successful_lap_time_mean_sec": metrics.get("successful_lap_time_mean_sec"),
                "successful_lap_time_std_sec": metrics.get("successful_lap_time_std_sec"),
            })
    plot_metric(summary, variants, "success_rate", "Completed episodes",
                "Lap completion rate", os.path.join(out_dir, "success_rate"), "wilson")
    plot_metric(summary, variants, "successful_lap_time_mean_sec", "Lap time (s)",
                "Successful lap time (mean ± SD)",
                os.path.join(out_dir, "lap_time"), "std")
    print(f"[ablation-plots] saved aggregate_summary.csv and success_rate/lap_time PNG/PDF files in {out_dir}")


if __name__ == "__main__":
    main()
