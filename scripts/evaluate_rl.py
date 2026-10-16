from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3 import PPO

from scripts.config_utils import load_config
from utils.track import discover_map, progress_fraction


def summarize_runs(runs: list[dict]) -> dict:
    successes = [run for run in runs if run["laps_completed"] > 0]
    lap_times = [time for run in runs for time in run["lap_times_sec"]]
    return {
        "num_episodes": len(runs),
        "successes": len(successes),
        "success_rate": len(successes) / len(runs) if runs else 0.0,
        "collision_count": sum(bool(run["collision"]) for run in runs),
        "collision_rate": float(np.mean([run["collision"] for run in runs])) if runs else 0.0,
        "avg_successful_lap_time_sec": float(np.mean(lap_times)) if lap_times else None,
        "avg_speed_mps": float(np.mean([run["avg_speed_mps"] for run in runs])) if runs else None,
        "avg_target_speed_mps": float(np.mean([run["avg_target_speed_mps"] for run in runs])) if runs else None,
        "max_progress_fraction_mean": float(np.mean([run["max_progress_fraction"] for run in runs])) if runs else None,
        "max_progress_fraction_std": float(np.std([run["max_progress_fraction"] for run in runs], ddof=1))
        if len(runs) > 1 else (0.0 if runs else None),
    }


def evaluate_model(model: PPO, checkpoint: str, config_path: str,
                   eval_maps: list[str], episodes_per_map: int,
                   max_episode_seconds: float, base_seed: int,
                   training_maps: list[str], training_seed):
    # Import the simulator only after PPO.load(): importing F110Env initializes
    # pyglet/OpenGL, which can conflict with PyTorch checkpoint loading.
    from rl_baselines.ppo_env import F1TenthPPOEnv

    runs = []
    for map_idx, map_name in enumerate(eval_maps):
        for episode in range(episodes_per_map):
            eval_seed = base_seed + map_idx * episodes_per_map + episode
            env = F1TenthPPOEnv(
                map_name, randomize_start=True,
                max_episode_seconds=max_episode_seconds,
                config_path=config_path)
            try:
                start_index = int(np.random.default_rng(eval_seed).integers(len(env.points)))
                obs, _ = env.reset(seed=eval_seed, options={"start_index": start_index})
                terminated = truncated = False
                total_reward = speed_sum = target_speed_sum = 0.0
                steps = 0
                max_progress_m = 0.0
                info = {}
                while not (terminated or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    max_progress_m = max(max_progress_m, float(info["progress_m"]))
                    total_reward += float(reward)
                    speed_sum += float(info["speed_mps"])
                    target_speed_sum += float(info["target_speed_mps"])
                    steps += 1
                runs.append({
                    "map_name": map_name,
                    "episode": episode,
                    "seed": eval_seed,
                    "start_index": start_index,
                    "laps_completed": int(info["laps_completed"]),
                    "collision": bool(info["collision"]),
                    "elapsed_time_sec": float(info["elapsed_time_sec"]),
                    "lap_times_sec": list(info["lap_times_sec"]),
                    "episode_return": total_reward,
                    "progress_m": float(info["progress_m"]),
                    "total_track_m": float(info["total_track_m"]),
                    "max_progress_m": max_progress_m,
                    "max_progress_fraction": progress_fraction(
                        max_progress_m, float(info["total_track_m"])),
                    "avg_speed_mps": speed_sum / max(steps, 1),
                    "avg_target_speed_mps": target_speed_sum / max(steps, 1),
                })
                row = runs[-1]
                print(f"[evaluate_rl] PPO {map_name} episode={episode} "
                      f"laps={row['laps_completed']} collision={row['collision']}")
            finally:
                env.close()

    per_map = {name: summarize_runs([run for run in runs if run["map_name"] == name])
               for name in eval_maps}
    return {
        "algorithm": "PPO",
        "training_seed": training_seed,
        "checkpoint": os.path.abspath(checkpoint),
        "training_maps": training_maps,
        "eval_maps": eval_maps,
        "episodes_per_map": episodes_per_map,
        "runs": runs,
        "per_map": per_map,
        "summary": summarize_runs(runs),
    }


def summarize_checkpoints(results: dict[str, dict]) -> dict:
    metrics = ("success_rate", "collision_rate", "avg_successful_lap_time_sec",
               "avg_speed_mps", "avg_target_speed_mps", "max_progress_fraction_mean")
    aggregate = {}
    for metric in metrics:
        values = [result["summary"][metric] for result in results.values()
                  if result["summary"][metric] is not None]
        aggregate[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "per_seed": values,
        }
    return aggregate


def discover_available_maps(maps_root: str) -> list[str]:
    """Find map folders with the expected YAML/image/centerline files."""
    if not os.path.isdir(maps_root):
        raise FileNotFoundError(f"maps_root '{maps_root}' does not exist.")
    names = []
    for name in sorted(os.listdir(maps_root)):
        folder = os.path.join(maps_root, name)
        if not os.path.isdir(folder):
            continue
        try:
            discover_map(maps_root, name)
            names.append(name)
        except FileNotFoundError as exc:
            print(f"[evaluate] skipping '{name}' under '{maps_root}' "
                  f"(not a usable map: {exc})")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing PPO checkpoints on held-out F1TENTH maps.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ppo_models", nargs="+", required=True,
                        help="Existing Stable-Baselines3 PPO .zip checkpoints to evaluate.")
    parser.add_argument("--eval_maps", nargs="+", default=None,
                        help="Optional explicit maps; defaults to available held-out maps.")
    parser.add_argument("--include_train_maps", action="store_true")
    parser.add_argument("--episodes_per_map", type=int, default=5)
    parser.add_argument("--max_episode_seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=10000,
                        help="Base evaluation seed; each map/episode gets a distinct seed.")
    parser.add_argument("--output", default="outputs/rl_comparison/evaluation_results.json")
    args = parser.parse_args()
    if args.episodes_per_map <= 0:
        parser.error("episodes_per_map must be positive")

    project_cfg = load_config(args.config)
    checkpoint_metadata = {}
    for path in args.ppo_models:
        if not os.path.isfile(path):
            parser.error(f"Checkpoint does not exist: {path}")
        metadata_path = os.path.splitext(path)[0] + ".json"
        if os.path.isfile(metadata_path):
            with open(metadata_path) as metadata_file:
                checkpoint_metadata[path] = json.load(metadata_file)

    training_maps = set(project_cfg.collection.map_names)
    if checkpoint_metadata:
        training_maps = set().union(*(
            set(metadata.get("training_maps", []))
            for metadata in checkpoint_metadata.values())) or training_maps

    if args.eval_maps:
        eval_maps = list(args.eval_maps)
        for map_name in eval_maps:
            try:
                discover_map(project_cfg.data.maps_root, map_name)
            except Exception as exc:
                parser.error(f"Invalid evaluation map '{map_name}': {exc}")
    else:
        available = discover_available_maps(project_cfg.data.maps_root)
        eval_maps = (available if args.include_train_maps else
                     [name for name in available if name not in training_maps])
    if not eval_maps:
        raise RuntimeError("No held-out maps found. Check maps/ and training maps.")
    overlap = sorted(training_maps.intersection(eval_maps))
    if overlap and not args.include_train_maps:
        parser.error(f"Evaluation maps overlap with checkpoint training maps: {overlap}")
    map_description = "maps" if args.include_train_maps else "held-out maps"
    print(f"Evaluating {len(eval_maps)} {map_description}: {eval_maps}")

    results = {}
    for index, path in enumerate(args.ppo_models):
        print(f"[evaluate_rl] loading PPO checkpoint with Stable-Baselines3: {path}",
              flush=True)
        try:
            policy = PPO.load(path, device="cpu")
        except Exception as exc:
            parser.error(f"Could not load PPO checkpoint '{path}' with Stable-Baselines3: {exc}")
        expected_dim = int(project_cfg.data.lidar_dim) + 2
        obs_shape = getattr(policy.observation_space, "shape", None)
        obs_dim = int(np.prod(obs_shape)) if obs_shape else None
        if obs_dim != expected_dim:
            parser.error(f"{path}: checkpoint expects observation shape {obs_shape}, but this "
                         f"environment provides {expected_dim} (LiDAR + speed + steering)")
        metadata = checkpoint_metadata.get(path, {})
        training_seed = metadata.get("seed")
        label = f"PPO seed {training_seed}" if training_seed is not None else f"PPO model {index + 1}"
        print(f"[evaluate_rl] loading existing PPO checkpoint {path} with Stable-Baselines3 "
              f"(obs_shape={obs_shape}, device={policy.device})", flush=True)
        results[label] = evaluate_model(
            policy, path, args.config, eval_maps, args.episodes_per_map,
            args.max_episode_seconds, args.seed,
            metadata.get("training_maps", sorted(training_maps)), training_seed)

    report = {
        "metadata": {
            "training_maps": sorted(training_maps),
            "evaluation_maps": eval_maps,
            "episodes_per_map": args.episodes_per_map,
            "evaluation_base_seed": args.seed,
            "success_definition": "at least one complete lap in an episode",
            "start_policy": "index from NumPy default_rng(seed), then centerline tangent",
            "progress_definition": "maximum segment-projected distance from seeded start, modulo track length, divided by track length and capped at one lap",
            "observation": "normalized LiDAR, speed/max_speed in [0, 1], steering/max_steer in [-1, 1]",
        },
        "results": results,
        "aggregate_across_checkpoints": summarize_checkpoints(results),
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as output_file:
        json.dump(report, output_file, indent=2, allow_nan=False)
    print(f"Saved evaluation report: {args.output}")


if __name__ == "__main__":
    main()
