from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import random
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from scripts.config_utils import load_config, resolve_device
from utils.normalization import Normalizer
from utils.track import Centerline, discover_map


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(cfg: Any, checkpoint: str, device: str):
    from models.skill_model import OPOSMSkillModel

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = OPOSMSkillModel(
        state_dim=ckpt.get("state_dim", len(cfg.data.state_features)),
        action_dim=ckpt.get("action_dim", len(cfg.data.action_features)),
        z_dim=ckpt.get("z_dim", cfg.model.z_dim),
        lidar_dim=ckpt.get("lidar_dim", cfg.data.lidar_dim),
        hidden_dim=cfg.model.hidden_dim,
        gru_hidden_dim=cfg.model.gru_hidden_dim,
        gru_bidirectional=cfg.model.gru_bidirectional,
        min_std=cfg.model.min_std,
        obs_embed_dim=ckpt.get("obs_embed_dim", cfg.model.obs_embed_dim),
        lidar_max_range=cfg.data.lidar_max_range,
        tawm_predict_delta=ckpt.get("tawm_predict_delta", cfg.model.tawm_predict_delta),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def make_variants(cfg) -> list[dict[str, Any]]:
    """Baseline plus a few isolated inference-time planner changes."""
    variants = [{"name": "default", "changes": {}}]
    variants.extend([
        {"name": "shorter_plan", "changes": {
            "plan_length": max(1, cfg.planning.plan_length // 2)}},
        {"name": "half_population", "changes": {
            "population_size": max(1, cfg.planning.population_size // 2)}},
        {"name": "half_cem_iterations", "changes": {
            "cem_iters": max(1, cfg.planning.cem_iters // 2)}},
        {"name": "replan_every_2_skills", "changes": {
            "replan_every_skills": 2}},
    ])
    return variants


def start_state_for_seed(centerline: Centerline, seed: int) -> tuple[np.ndarray, int]:
    idx = int(np.random.default_rng(seed).integers(len(centerline.points)))
    p0 = centerline.points[idx]
    p1 = centerline.points[(idx + 1) % len(centerline.points)]
    yaw = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    state = np.array([p0[0], p0[1], 0.0, 0.0, yaw, 0.0, 0.0], dtype=np.float32)
    return state, idx


def evaluate_episode(model, cfg, state_norm, action_norm, device,
                     map_name: str, seed: int, variant: dict[str, Any]) -> dict[str, Any]:
    from envs.f1tenth_env import F1TenthEnvConfig, F1TenthEnvironment
    from planning.cem_planner import CEMPlanner, CEMPlannerConfig, mpc_race
    from planning.reward import LidarStructuralReward

    map_yaml, map_ext, centerline_csv = discover_map(cfg.data.maps_root, map_name)
    centerline = Centerline.from_csv(centerline_csv)
    initial_state, start_index = start_state_for_seed(centerline, seed)
    waypoints = centerline.subsample_waypoints(cfg.planning.waypoint_spacing_m)

    planner_values = {
        "plan_length": cfg.planning.plan_length,
        "population_size": cfg.planning.population_size,
        "elite_frac": cfg.planning.elite_frac,
        "cem_iters": cfg.planning.cem_iters,
        "init_std": cfg.planning.init_std,
        "replan_every_skills": cfg.planning.replan_every_skills,
    }
    planner_values.update(variant["changes"])
    reward = LidarStructuralReward(
        lidar_dim=model.lidar_dim,
        forward_weight=cfg.planning.forward_reward_weight,
        symmetry_weight=cfg.planning.symmetry_penalty_weight,
        collision_penalty=cfg.planning.lidar_collision_penalty,
        safe_distance=cfg.planning.safe_distance_m,
        forward_fraction=cfg.planning.forward_beam_fraction,
    )
    planner_cfg = CEMPlannerConfig(
        plan_length=planner_values["plan_length"],
        population_size=planner_values["population_size"],
        elite_frac=planner_values["elite_frac"],
        cem_iters=planner_values["cem_iters"],
        init_std=planner_values["init_std"],
        device=device,
        save_dir=None,
    )
    planner = CEMPlanner(model, reward, planner_cfg)
    env_cfg = F1TenthEnvConfig(
        map_yaml=map_yaml, map_ext=map_ext,
        timestep=cfg.f1tenth_env.timestep,
        control_hz=cfg.f1tenth_env.control_hz,
        max_steer_rad=cfg.f1tenth_env.max_steer_rad,
        max_speed_mps=cfg.f1tenth_env.max_speed_mps,
        render=False, lidar_dim=model.lidar_dim,
        action_repeat=cfg.data.subsample_stride,
    )
    env = F1TenthEnvironment(model, state_norm, action_norm, env_cfg,
                             initial_state, device=device)
    try:
        result = mpc_race(
            planner, env, waypoints,
            skill_horizon=cfg.data.subtraj_len - 1,
            max_skills=cfg.planning.max_skills_per_episode,
            max_laps=cfg.planning.max_laps,
            replan_every_skills=planner_values["replan_every_skills"],
        )
    finally:
        env.close()

    completed = int(result["laps_completed"])
    lap_times = [float(x) for x in result.get("lap_times_sec", [])]
    return {
        "variant": variant["name"], "map": map_name, "seed": int(seed),
        "start_index": start_index, "laps_completed": completed,
        "success": completed >= int(cfg.planning.max_laps),
        "collision": bool(result.get("collision", False)),
        "lap_times_sec": lap_times,
        "planner_settings": planner_values,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["success"]]
    times = [time for row in successes for time in row["lap_times_sec"]]
    return {
        "episodes": len(rows),
        "completed_episodes": len(successes),
        "completed_over_episodes": f"{len(successes)}/{len(rows)}",
        "success_rate": (len(successes) / len(rows)) if rows else None,
        "successful_laps": len(times),
        "successful_lap_time_mean_sec": float(np.mean(times)) if times else None,
        "successful_lap_time_std_sec": (float(np.std(times, ddof=1))
                                          if len(times) > 1 else (0.0 if times else None)),
        "collision_rate": (float(np.mean([r["collision"] for r in rows])) if rows else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalizer_dir", default=None,
                        help="Defaults to train.checkpoint_dir from the config")
    parser.add_argument("--maps", nargs="+",
                        default=["BrandsHatch", "Budapest", "Hockenheim", "Monza", "Shanghai"],
                        help="OOD maps (defaults to five maps not listed in collection.map_names)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", default="outputs/inference_ablations")
    args = parser.parse_args()
    if not args.seeds:
        parser.error("Provide at least one seed")

    cfg = load_config(args.config)
    training_maps = set(cfg.collection.map_names)
    in_distribution = sorted(training_maps.intersection(args.maps))
    if in_distribution:
        parser.error(f"Ablation maps must be OOD; these are configured training maps: {in_distribution}")
    device = resolve_device(cfg.planning.device)
    normalizer_dir = args.normalizer_dir or cfg.train.checkpoint_dir
    state_norm = Normalizer.load(os.path.join(normalizer_dir, "state_normalizer.json"))
    action_norm = Normalizer.load(os.path.join(normalizer_dir, "action_normalizer.json"))
    model = load_model(cfg, args.checkpoint, device)
    variants = make_variants(cfg)

    all_rows = []
    for variant in variants:
        print(f"[ablation] {variant['name']}: changes={variant['changes'] or 'none'}")
        for map_name in args.maps:
            for seed in args.seeds:
                seed_everything(seed)
                print(f"[ablation] map={map_name} seed={seed}", flush=True)
                row = evaluate_episode(model, cfg, state_norm, action_norm,
                                       device, map_name, seed, variant)
                all_rows.append(row)
                lap_time = row["lap_times_sec"][0] if row["lap_times_sec"] else None
                print(f"[ablation] completed={row['laps_completed']} "
                      f"collision={row['collision']} lap_time={lap_time}", flush=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, stamp)
    os.makedirs(out_dir, exist_ok=True)
    summary_by_variant = {
        variant["name"]: summarize([r for r in all_rows if r["variant"] == variant["name"]])
        for variant in variants
    }
    report = {
        "metadata": {
            "config": os.path.abspath(args.config),
            "checkpoint": os.path.abspath(args.checkpoint),
            "maps": args.maps, "seeds": args.seeds,
            "success_definition": "completed configured max_laps",
            "lap_time_definition": "simulated lap times, averaged over successful episodes",
            "ablations": variants,
        },
        "summary_by_variant": summary_by_variant,
        "runs": all_rows,
    }
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    runs_path = os.path.join(out_dir, "runs.csv")
    with open(runs_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "variant", "map", "seed", "start_index", "laps_completed", "success",
            "collision", "lap_times_sec", "planner_settings",
        ])
        writer.writeheader()
        for row in all_rows:
            csv_row = dict(row)
            csv_row["lap_times_sec"] = json.dumps(csv_row["lap_times_sec"])
            csv_row["planner_settings"] = json.dumps(csv_row["planner_settings"], sort_keys=True)
            writer.writerow(csv_row)

    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        fields = ["variant", "episodes", "completed_episodes", "completed_over_episodes",
                  "success_rate", "successful_laps", "successful_lap_time_mean_sec",
                  "successful_lap_time_std_sec", "collision_rate", "planner_settings"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for variant in variants:
            name = variant["name"]
            writer.writerow({"variant": name, **summary_by_variant[name],
                             "planner_settings": json.dumps(variant["changes"], sort_keys=True)})

    print(f"[ablation] saved results under {out_dir}")


if __name__ == "__main__":
    main()
