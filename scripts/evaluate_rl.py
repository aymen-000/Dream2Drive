import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3 import PPO

from rl_baselines.f1tenth import F1TenthRLTrackEnv, rl_env_config_from_project
from scripts.config_utils import load_config
from scripts.evaluate import discover_available_maps, select_eval_maps


def summarize_runs(runs):
    completed = [r for r in runs if r["laps_completed"] > 0]
    lap_times = [t for r in runs for t in r.get("lap_times_sec", [])]
    episode_times = [r["elapsed_time_sec"] for r in runs
                     if r.get("elapsed_time_sec") is not None]
    return {
        "num_episodes": len(runs),
        "laps_completed": int(sum(r["laps_completed"] for r in runs)),
        "successes": len(completed),
        "success_rate": float(len(completed) / len(runs)) if runs else 0.0,
        "collision_count": int(sum(r["collision"] for r in runs)),
        "collision_rate": float(np.mean([r["collision"] for r in runs])) if runs else 0.0,
        "avg_successful_lap_time_sec": float(np.mean(lap_times)) if lap_times else None,
        "avg_episode_time_sec": float(np.mean(episode_times)) if episode_times else None,
        "avg_speed_mps": float(np.mean([r["avg_speed_mps"] for r in runs
                                       if r.get("avg_speed_mps") is not None]))
        if any(r.get("avg_speed_mps") is not None for r in runs) else None,
        "runs": runs,
    }


def evaluate_policy(model, model_path, config_path, maps, episodes_per_map,
                    max_episode_seconds, seed, deterministic, randomize_start,
                    training_maps, training_seed):
    runs = []
    for map_idx, map_name in enumerate(maps):
        env_cfg = rl_env_config_from_project(
            config_path, map_name, max_episode_seconds,
            randomize_start=randomize_start)
        for episode in range(episodes_per_map):
            episode_seed = seed + map_idx * episodes_per_map + episode
            env = F1TenthRLTrackEnv(env_cfg)
            try:
                obs, _ = env.reset(seed=episode_seed)
                terminated = truncated = False
                episode_return = 0.0
                speed_sum = 0.0
                target_speed_sum = 0.0
                speed_steps = 0
                info = {}
                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    speed_sum += float(info.get("speed_mps", 0.0))
                    target_speed_sum += float(info.get("target_speed_mps", 0.0))
                    speed_steps += 1
                runs.append({
                    "map_name": map_name, "episode": episode,
                    "seed": episode_seed,
                    "laps_completed": int(info.get("laps_completed", 0)),
                    "collision": bool(info.get("collision", False)),
                    "elapsed_time_sec": float(info.get("elapsed_time_sec", 0.0)),
                    "lap_times_sec": list(info.get("lap_times_sec", [])),
                    "episode_return": episode_return,
                    "progress_m": float(info.get("progress_m", 0.0)),
                    "total_track_m": float(info.get("total_track_m", 0.0)),
                    "avg_speed_mps": speed_sum / max(speed_steps, 1),
                    "avg_target_speed_mps": target_speed_sum / max(speed_steps, 1),
                })
                print(f"[evaluate_rl] PPO {map_name} episode={episode} "
                      f"laps={runs[-1]['laps_completed']} collision={runs[-1]['collision']}")
            finally:
                env.close()
    by_map = {name: summarize_runs([r for r in runs if r["map_name"] == name])
              for name in maps}
    return {
        "algorithm": "PPO", "checkpoint": os.path.abspath(model_path),
        "seed": training_seed, "evaluation_seed": seed,
        "training_maps": training_maps, "eval_maps": maps,
        "episodes_per_map": episodes_per_map,
        "deterministic_actions": deterministic,
        "start_policy": "seeded_random_centerline_point" if randomize_start else "first_centerline_point",
        "summary": summarize_runs(runs), "per_map": by_map,
    }


def load_planner_result(path, label, selected_maps):
    with open(path, "r") as f:
        source = json.load(f)
    result = source.get(label)
    if result is None:
        result = next(iter(source.values())) if source else {}
    planning = result.get("planning", {})
    runs = []
    selected = set(selected_maps)
    for row in planning.get("runs", []):
        if row.get("map_name") not in selected:
            continue
        runs.append({
            "map_name": row.get("map_name"),
            "laps_completed": int(row.get("laps_completed", 0)),
            "collision": bool(row.get("collision", False)),
            "lap_times_sec": row.get("lap_times_sec", []),
            "elapsed_time_sec": row.get("elapsed_time_sec"),
        })
    return {
        "algorithm": label, "source_json": os.path.abspath(path),
        "eval_maps": selected_maps, "summary": summarize_runs(runs),
        "timing_available": any(r.get("lap_times_sec") for r in runs),
    }


def summarize_training_seeds(results):
    """Summarize model-level outcomes so episodes are not treated as seeds."""
    metrics = ("success_rate", "collision_rate", "avg_successful_lap_time_sec", "avg_speed_mps")
    summary = {"num_training_seeds": len(results), "per_seed": {}}
    for metric in metrics:
        values = [r["summary"][metric] for r in results
                  if r["summary"][metric] is not None]
        summary["per_seed"][metric] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "values": values,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ppo_models", nargs="+", default=None,
                        help="One or more PPO zip checkpoints, such as independent training seeds.")
    parser.add_argument("--direct_results", default=None)
    parser.add_argument("--cem_results", default=None)
    parser.add_argument("--eval_maps", nargs="+", default=None)
    parser.add_argument("--include_train_maps", action="store_true")
    parser.add_argument("--episodes_per_map", type=int, default=5)
    parser.add_argument("--max_episode_seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=10000,
                        help="First evaluation seed; episodes receive distinct fixed seeds.")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of using deterministic PPO actions.")
    parser.add_argument("--fixed_start", action="store_true",
                        help="Use each map's first centerline point for every episode.")
    parser.add_argument("--output", default="outputs/rl_comparison/evaluation_results.json")
    args = parser.parse_args()
    if args.episodes_per_map <= 0:
        parser.error("episodes_per_map must be positive")
    if not args.ppo_models and not args.direct_results and not args.cem_results:
        parser.error("Pass at least one PPO checkpoint or planner result JSON")

    project_cfg = load_config(args.config)
    checkpoint_metadata = {}
    for path in args.ppo_models or []:
        metadata_path = os.path.splitext(path)[0] + ".json"
        if os.path.isfile(metadata_path):
            with open(metadata_path) as f:
                checkpoint_metadata[path] = json.load(f)
    if checkpoint_metadata:
        actual_train_maps = set()
        for meta in checkpoint_metadata.values():
            actual_train_maps.update(meta.get("training_maps", []))
    else:
        actual_train_maps = set(project_cfg.collection.map_names)
    if args.eval_maps:
        eval_maps = list(args.eval_maps)
    else:
        available = discover_available_maps(project_cfg.data.maps_root)
        eval_maps = (available if args.include_train_maps else
                     [name for name in available if name not in actual_train_maps])
    if not eval_maps:
        parser.error("No evaluation maps selected")
    overlap = sorted(actual_train_maps.intersection(eval_maps))
    if overlap and not args.include_train_maps:
        parser.error(f"Evaluation maps overlap with PPO training maps: {overlap}")
    print(f"[evaluate_rl] maps={eval_maps} episodes_per_map={args.episodes_per_map}")

    report = {"metadata": {
        "evaluation_maps": eval_maps,
        "excluded_training_maps": sorted(actual_train_maps),
        "episodes_per_map": args.episodes_per_map,
        "base_seed": args.seed,
        "start_policy": "first_centerline_point" if args.fixed_start else "seeded_random_centerline_point",
        "success_definition": "at least one complete lap in an episode",
        "lap_time_definition": "simulated seconds from reset to crossing the start line",
        "missing_lap_time_note": "Legacy planner source files may not contain lap times.",
    }, "results": {}}
    ppo_results = []
    for index, path in enumerate(args.ppo_models or []):
        model = PPO.load(path, device="auto")
        meta = checkpoint_metadata.get(path, {})
        training_seed = meta.get("seed")
        label = f"PPO seed {training_seed}" if training_seed is not None else f"PPO model {index + 1}"
        result = evaluate_policy(
            model, path, args.config, eval_maps, args.episodes_per_map,
            args.max_episode_seconds, args.seed, not args.stochastic,
            not args.fixed_start, meta.get("training_maps", list(project_cfg.collection.map_names)),
            training_seed)
        report["results"][label] = result
        ppo_results.append(result)
    if ppo_results:
        report["ppo_seed_aggregate"] = summarize_training_seeds(ppo_results)
    if args.direct_results:
        report["results"]["Direct"] = load_planner_result(
            args.direct_results, "Direct", eval_maps)
    if args.cem_results:
        report["results"]["CEM (MPC)"] = load_planner_result(
            args.cem_results, "CEM (MPC)", eval_maps)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f"[evaluate_rl] wrote {args.output}")


if __name__ == "__main__":
    main()
