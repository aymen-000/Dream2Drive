"""Matched evaluation for saved PPO, PPO-LSTM, SAC, and IQL checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from scripts.config_utils import load_config
from scripts.evaluate_all import (
    discover_available_maps,
    discover_baseline_checkpoints,
    evaluate_iql_episode,
    evaluate_ppo_episode,
    load_baseline,
    summarize,
)
from utils.track import discover_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoints", "--ppo_models", dest="checkpoints", nargs="+", default=None,
        help="Saved PPO/PPO-LSTM/SAC .zip and IQL .pt checkpoints; type is detected per file.",
    )
    parser.add_argument("--checkpoint_dirs", nargs="+", default=None,
                        help="Discover saved PPO/PPO-LSTM/SAC/IQL checkpoints in these folders.")
    parser.add_argument("--algorithm", choices=("auto", "ppo", "ppo_lstm", "sac", "iql"),
                        default="auto", help="Use auto for mixed checkpoint types.")
    parser.add_argument("--eval_maps", nargs="+", default=None)
    parser.add_argument("--include_train_maps", action="store_true")
    parser.add_argument("--episodes_per_map", type=int, default=5)
    parser.add_argument("--max_episode_seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output_dir", default="outputs/rl_comparison")
    args = parser.parse_args()
    if args.episodes_per_map <= 0:
        parser.error("episodes_per_map must be positive")
    args.checkpoints = list(args.checkpoints or [])
    if args.checkpoint_dirs:
        try:
            args.checkpoints.extend(discover_baseline_checkpoints(args.checkpoint_dirs))
        except FileNotFoundError as exc:
            parser.error(str(exc))
    args.checkpoints = sorted(set(args.checkpoints))
    if not args.checkpoints:
        parser.error("Provide --checkpoints or --checkpoint_dirs")

    cfg = load_config(args.config)
    training_maps = set(cfg.collection.map_names)
    checkpoint_info = []
    for path in args.checkpoints:
        if not os.path.isfile(path):
            parser.error(f"Checkpoint does not exist: {path}")
        sidecar_path = os.path.splitext(path)[0] + ".json"
        metadata = {}
        if os.path.isfile(sidecar_path):
            with open(sidecar_path) as sidecar_file:
                metadata = json.load(sidecar_file)
        elif path.lower().endswith(".pt"):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        training_maps.update(metadata.get("training_maps", []))
        training_maps.update(metadata.get("dataset_maps", []))
        checkpoint_info.append((path, metadata))

    if args.eval_maps:
        maps = list(args.eval_maps)
    else:
        available = discover_available_maps(cfg.data.maps_root)
        maps = available if args.include_train_maps else [
            name for name in available if name not in training_maps
        ]
    if not maps:
        parser.error("No evaluation maps found; pass --eval_maps or check training maps.")
    for name in maps:
        try:
            discover_map(cfg.data.maps_root, name)
        except Exception as exc:
            parser.error(f"Invalid evaluation map {name!r}: {exc}")
    overlap = sorted(set(maps) & training_maps)
    if overlap and not args.include_train_maps:
        parser.error(f"Evaluation maps overlap with checkpoint training maps: {overlap}")

    loaded = []
    expected_obs_dim = int(cfg.data.lidar_dim) + 2
    for path, sidecar in checkpoint_info:
        try:
            algorithm, model = load_baseline(path, args.algorithm)
        except Exception as exc:
            parser.error(f"Could not load checkpoint {path}: {exc}")
        if algorithm == "iql":
            obs_dim = int(model.config.obs_dim)
            action_shape = (int(model.config.action_dim),)
        else:
            obs_shape = getattr(model.observation_space, "shape", None)
            obs_dim = int(np.prod(obs_shape)) if obs_shape else None
            action_shape = getattr(model.action_space, "shape", None)
        if obs_dim != expected_obs_dim:
            parser.error(
                f"{path} expects {obs_dim} observation values; the evaluation "
                f"environment provides {expected_obs_dim}."
            )
        if action_shape != (2,):
            parser.error(f"{path} has action shape {action_shape}; expected (2,).")
        metadata = getattr(model, "baseline_metadata", sidecar)
        seed = metadata.get("seed")
        display_algorithm = "PPO-LSTM" if algorithm == "ppo_lstm" else algorithm.upper()
        label = (f"{display_algorithm} seed {seed} ({os.path.basename(path)})"
                 if seed is not None else os.path.basename(path))
        loaded.append((label, algorithm, model, path, metadata))

    results_by_checkpoint = {}
    rows = []
    for label, algorithm, model, path, metadata in loaded:
        checkpoint_runs = []
        for map_idx, map_name in enumerate(maps):
            for episode in range(args.episodes_per_map):
                eval_seed = args.seed + map_idx * args.episodes_per_map + episode
                if algorithm == "iql":
                    row = evaluate_iql_episode(
                        model, args.config, map_name, eval_seed, args.max_episode_seconds
                    )
                else:
                    row = evaluate_ppo_episode(
                        model, args.config, map_name, eval_seed,
                        args.max_episode_seconds, algorithm,
                    )
                row.update({
                    "algorithm": "PPO-LSTM" if algorithm == "ppo_lstm" else algorithm.upper(),
                    "checkpoint": os.path.abspath(path),
                    "training_seed": metadata.get("seed"),
                    "episode": episode,
                })
                checkpoint_runs.append(row)
                rows.append(row)
                print(
                    f"[evaluate_rl] {label} map={map_name} seed={eval_seed} "
                    f"laps={row['laps_completed']} collision={row['collision']}"
                )
        results_by_checkpoint[label] = {
            "algorithm": "PPO-LSTM" if algorithm == "ppo_lstm" else algorithm.upper(),
            "checkpoint": os.path.abspath(path),
            "training_seed": metadata.get("seed"),
            "runs": checkpoint_runs,
            "per_map": {
                name: summarize([run for run in checkpoint_runs if run["map_name"] == name])
                for name in maps
            },
            "summary": summarize(checkpoint_runs),
        }

    algorithm_groups = {}
    for result in results_by_checkpoint.values():
        algorithm_groups.setdefault(result["algorithm"], []).extend(result["runs"])
    report = {
        "metadata": {
            "config": os.path.abspath(args.config),
            "training_maps": sorted(training_maps),
            "evaluation_maps": maps,
            "episodes_per_map": args.episodes_per_map,
            "evaluation_base_seed": args.seed,
            "max_episode_seconds": args.max_episode_seconds,
            "matched_evaluation": (
                "Every checkpoint receives the same map, derived evaluation seed, "
                "centerline start, and episode time limit."
            ),
            "observation": (
                "Full normalized LiDAR followed by normalized speed and steering; "
                "IQL receives the same observation layout used during training."
            ),
            "success_definition": "at least one complete lap in an episode",
        },
        "summary_by_algorithm": {
            name: summarize(group) for name, group in algorithm_groups.items()
        },
        "results_by_checkpoint": results_by_checkpoint,
        "runs": rows,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as output_file:
        json.dump(report, output_file, indent=2, allow_nan=False)

    fields = sorted({key for row in rows for key in row})
    with open(os.path.join(args.output_dir, "evaluation_runs.csv"), "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})

    summary_fields = ["algorithm", "checkpoint", "training_seed", "episodes", "successes",
                      "success_rate", "collision_rate", "max_progress_fraction_mean",
                      "successful_lap_time_mean_sec", "successful_lap_time_std_sec"]
    with open(os.path.join(args.output_dir, "evaluation_summary.csv"), "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        for result in results_by_checkpoint.values():
            writer.writerow({key: result.get(key) for key in summary_fields} | result["summary"])

    algorithm_fields = ["algorithm", "episodes", "successes", "success_rate",
                        "collision_rate", "max_progress_fraction_mean",
                        "successful_lap_time_mean_sec", "successful_lap_time_std_sec"]
    with open(os.path.join(args.output_dir, "evaluation_algorithm_summary.csv"),
              "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=algorithm_fields, extrasaction="ignore")
        writer.writeheader()
        for algorithm, summary in report["summary_by_algorithm"].items():
            writer.writerow({"algorithm": algorithm, **summary})

    print(f"Saved JSON, run-level CSV, and checkpoint summary CSV under {args.output_dir}")


if __name__ == "__main__":
    main()
