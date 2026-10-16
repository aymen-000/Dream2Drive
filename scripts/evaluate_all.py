from __future__ import annotations

import argparse
import json
import os
import random
import sys
import zipfile
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO, SAC

from scripts.config_utils import load_config, resolve_device
from utils.track import discover_map, progress_fraction, relative_progress_m


def discover_available_maps(maps_root: str) -> list[str]:
    """Find valid map folders without importing the simulator or renderer."""
    if not os.path.isdir(maps_root):
        raise FileNotFoundError(f"maps_root '{maps_root}' does not exist.")
    names = []
    for name in sorted(os.listdir(maps_root)):
        if not os.path.isdir(os.path.join(maps_root, name)):
            continue
        try:
            discover_map(maps_root, name)
            names.append(name)
        except FileNotFoundError as exc:
            print(f"[evaluate] skipping '{name}' under '{maps_root}' "
                  f"(not a usable map: {exc})")
    return names


def load_baseline(path: str, requested_algorithm: str):
    """Load an SB3 PPO/SAC archive using the matching SB3 algorithm class."""
    with zipfile.ZipFile(path) as archive:
        saved_data = json.loads(archive.read("data"))
    if requested_algorithm == "auto":
        if "n_steps" in saved_data:
            algorithm = "ppo"
        elif "train_freq" in saved_data:
            algorithm = "sac"
        else:
            raise ValueError(f"Cannot infer PPO/SAC algorithm from checkpoint metadata: {path}")
    else:
        algorithm = requested_algorithm
    model_class = PPO if algorithm == "ppo" else SAC
    return algorithm, model_class.load(path, device="cpu")


def load_oposm(cfg: Any, checkpoint: str, device: str) -> OPOSMSkillModel:
    from models.skill_model import OPOSMSkillModel

    model = OPOSMSkillModel(
        state_dim=len(cfg.data.state_features), action_dim=len(cfg.data.action_features),
        z_dim=cfg.model.z_dim, hidden_dim=cfg.model.hidden_dim,
        gru_hidden_dim=cfg.model.gru_hidden_dim,
        gru_bidirectional=cfg.model.gru_bidirectional, min_std=cfg.model.min_std,
        lidar_dim=cfg.data.lidar_dim, obs_embed_dim=cfg.model.obs_embed_dim,
        lidar_max_range=cfg.data.lidar_max_range,
        tawm_predict_delta=cfg.model.tawm_predict_delta,
    )
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.to(device).eval()
    return model


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_start(centerline: Centerline, seed: int) -> tuple[np.ndarray, float]:
    points = centerline.points
    idx = int(np.random.default_rng(seed).integers(len(points)))
    next_idx = (idx + 1) % len(points)
    direction = points[next_idx] - points[idx]
    yaw = float(np.arctan2(direction[1], direction[0]))
    state = np.array([points[idx, 0], points[idx, 1], 0., 0., yaw, 0., 0.], dtype=np.float32)
    return state, float(idx)


def evaluate_oposm_episode(model, cfg, state_norm, action_norm, device, map_name,
                           seed: int) -> dict[str, Any]:
    from envs.f1tenth_env import F1TenthEnvConfig, F1TenthEnvironment
    from planning.cem_planner import CEMPlanner, CEMPlannerConfig, mpc_race
    from planning.reward import LidarStructuralReward
    from utils.track import Centerline

    map_yaml, map_ext, centerline_csv = discover_map(cfg.data.maps_root, map_name)
    centerline = Centerline.from_csv(centerline_csv)
    start_state, start_index = _seed_start(centerline, seed)
    waypoints = centerline.subsample_waypoints(cfg.planning.waypoint_spacing_m)
    reward = LidarStructuralReward(
        lidar_dim=model.lidar_dim,
        forward_weight=cfg.planning.forward_reward_weight,
        symmetry_weight=cfg.planning.symmetry_penalty_weight,
        collision_penalty=cfg.planning.lidar_collision_penalty,
        safe_distance=cfg.planning.safe_distance_m,
        forward_fraction=cfg.planning.forward_beam_fraction,
    )
    planner_cfg = CEMPlannerConfig(
        plan_length=cfg.planning.plan_length,
        population_size=cfg.planning.population_size,
        elite_frac=cfg.planning.elite_frac, cem_iters=cfg.planning.cem_iters,
        init_std=cfg.planning.init_std, device=device, save_dir=None,
    )
    planner = CEMPlanner(model, reward, planner_cfg)
    env_cfg = F1TenthEnvConfig(
        map_yaml=map_yaml, map_ext=map_ext, timestep=cfg.f1tenth_env.timestep,
        control_hz=cfg.f1tenth_env.control_hz,
        max_steer_rad=cfg.f1tenth_env.max_steer_rad,
        max_speed_mps=cfg.f1tenth_env.max_speed_mps, render=False,
        lidar_dim=cfg.data.lidar_dim, action_repeat=cfg.data.subsample_stride,
    )
    env = F1TenthEnvironment(model, state_norm, action_norm, env_cfg, start_state, device=device)
    try:
        result = mpc_race(planner, env, waypoints, skill_horizon=cfg.data.subtraj_len - 1,
                          max_skills=cfg.planning.max_skills_per_episode,
                          max_laps=cfg.planning.max_laps,
                          replan_every_skills=cfg.planning.replan_every_skills)
        track_len = centerline.total_length
        start_s = centerline.cum_dist[int(start_index)]
        projected = [centerline.project_m(state[:2])[0]
                     for state in result["trajectory_raw"]]
        progress = max(relative_progress_m(s, start_s, track_len) for s in projected)
        lap_times = [float(t) for t in result.get("lap_times_sec", [])]
        return {
            "method": "OPOSM CEM-MPC",
            "map_name": map_name, "seed": seed, "start_index": start_index,
            "laps_completed": int(result["laps_completed"]),
            "success": bool(result["laps_completed"] > 0),
            "collision": bool(result.get("collision", False)),
            "max_progress_fraction": progress_fraction(progress, track_len),
            "max_progress_m": progress, "track_length_m": track_len,
            "lap_times_sec": lap_times,
        }
    finally:
        env.close()


def evaluate_ppo_episode(model, config_path, map_name, seed: int, max_seconds: float) -> dict[str, Any]:
    from rl_baselines.ppo_env import F1TenthPPOEnv

    env = F1TenthPPOEnv(
        map_name, randomize_start=True, max_episode_seconds=max_seconds,
        config_path=config_path)
    try:
        start_index = int(np.random.default_rng(seed).integers(len(env.points)))
        obs, _ = env.reset(seed=seed, options={"start_index": start_index})
        terminated = truncated = False
        info: dict[str, Any] = {}
        max_progress = 0.
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            max_progress = max(max_progress, float(info.get("progress_m", 0.)))
        track_len = float(info.get("total_track_m", 0.))
        laps = int(info.get("laps_completed", 0))
        return {
            "method": "RL baseline", "map_name": map_name, "seed": seed,
            "start_index": start_index,
            "laps_completed": laps, "success": laps > 0,
            "collision": bool(info.get("collision", False)),
            "max_progress_fraction": progress_fraction(max_progress, track_len),
            "max_progress_m": max_progress, "track_length_m": track_len,
            "lap_times_sec": [float(t) for t in info.get("lap_times_sec", [])],
        }
    finally:
        env.close()


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    progress = [r["max_progress_fraction"] for r in runs]
    successes = [r for r in runs if r["success"]]
    lap_times = [t for r in successes for t in r["lap_times_sec"]]
    return {
        "episodes": len(runs),
        "success_rate": float(np.mean([r["success"] for r in runs])) if runs else None,
        "successes": len(successes),
        "max_progress_fraction_mean": float(np.mean(progress)) if progress else None,
        "max_progress_fraction_std": float(np.std(progress, ddof=1)) if len(progress) > 1 else 0.,
        "successful_lap_time_mean_sec": float(np.mean(lap_times)) if lap_times else None,
        "successful_lap_time_std_sec": float(np.std(lap_times, ddof=1)) if len(lap_times) > 1 else None,
        "collision_rate": float(np.mean([r["collision"] for r in runs])) if runs else None,
    }


def _mean_ci(values: list[float]) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if not len(x):
        return float("nan"), float("nan")
    # Normal 95% interval; report one observation as a point without error bars.
    return float(np.mean(x)), float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.


def make_figures(results: dict[str, list[dict[str, Any]]], out_dir: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.labelsize": 10, "legend.fontsize": 9,
                         "figure.dpi": 150, "savefig.dpi": 300,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"RL baseline": "#4C78A8", "OPOSM CEM-MPC": "#54A24B"}
    methods = list(results)
    def save_figure(fig, stem: str) -> None:
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight")
        plt.close(fig)

    # Per episode maximum progress, with individual seed outcomes and mean 95% CI.
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    for i, method in enumerate(methods):
        vals = [r["max_progress_fraction"] for r in results[method]]
        jitter = np.linspace(-0.08, 0.08, max(1, len(vals)))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=16, alpha=.55,
                   color=colors.get(method), edgecolor="none")
        mean, ci = _mean_ci(vals)
        ax.errorbar(i, mean, yerr=ci, fmt="D", color="black", capsize=3, ms=4, zorder=5)
    ax.set_xticks(range(len(methods)), methods, rotation=20, ha="right")
    ax.set_ylabel("Maximum lap progress")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{x:.0%}" for x in np.linspace(0, 1, 6)])
    ax.grid(axis="y", alpha=.25)
    ax.set_title("Maximum lap progress")
    save_figure(fig, "evaluation_progress")

    # Lap time is conditioned on successful episodes and labeled as such.
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    for i, method in enumerate(methods):
        vals = [t for r in results[method] if r["success"] for t in r["lap_times_sec"]]
        if vals:
            mean, ci = _mean_ci(vals)
            ax.errorbar(i, mean, yerr=ci, fmt="o", color=colors.get(method),
                        capsize=4, ms=5, lw=1.5)
            ax.scatter(np.full(len(vals), i) + np.linspace(-.08, .08, len(vals)), vals,
                       s=14, color=colors.get(method), alpha=.45, edgecolor="none")
    ax.set_xticks(range(len(methods)), methods, rotation=20, ha="right")
    ax.set_ylabel("Lap time (s)")
    ax.grid(axis="y", alpha=.25)
    ax.set_title("Successful lap time")
    save_figure(fig, "evaluation_lap_time")

    # Wilson 95% interval for a Bernoulli success rate.
    fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
    for i, method in enumerate(methods):
        runs = results[method]
        n = len(runs)
        if not n:
            continue
        p = sum(r["success"] for r in runs) / n
        z = 1.96
        den = 1 + z*z/n
        center = (p + z*z/(2*n))/den
        half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
        ax.errorbar(i, p, yerr=[[p-max(0., center-half)], [min(1., center+half)-p]],
                    fmt="o", color=colors.get(method), capsize=4, ms=5)
        ax.text(i, min(1.05, p + .09), f"{sum(r['success'] for r in runs)}/{n}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(methods)), methods, rotation=20, ha="right")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{x:.0%}" for x in np.linspace(0, 1, 6)])
    ax.grid(axis="y", alpha=.25)
    ax.set_title("Lap completion")
    save_figure(fig, "evaluation_success_rate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model_checkpoint", required=True,
                        help="OPOSM checkpoint; evaluated with CEM-MPC.")
    parser.add_argument("--normalizer_dir", required=True,
                        help="Directory containing this model's state/action normalizer JSONs.")
    parser.add_argument("--baseline_checkpoints", nargs="+", required=True,
                        help="One or more PPO/SAC .zip baseline checkpoints (checkpoint seeds/models).")
    parser.add_argument("--baseline_algorithm", choices=("auto", "ppo", "sac"), default="auto",
                        help="Expected algorithm of the baseline checkpoints; auto detects it from the weights.")
    parser.add_argument("--eval_maps", nargs="+", default=None)
    parser.add_argument("--include_train_maps", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--max_episode_seconds", type=float, default=120.)
    parser.add_argument("--output_dir", default="outputs/evaluation")
    args = parser.parse_args()
    if not args.seeds:
        parser.error("Provide at least one evaluation seed")

    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)
    from utils.normalization import Normalizer

    state_norm = Normalizer.load(os.path.join(args.normalizer_dir, "state_normalizer.json"))
    action_norm = Normalizer.load(os.path.join(args.normalizer_dir, "action_normalizer.json"))
    train_maps = set(cfg.collection.map_names)
    maps = list(args.eval_maps) if args.eval_maps else [
        m for m in discover_available_maps(cfg.data.maps_root)
        if args.include_train_maps or m not in train_maps]
    if not maps:
        parser.error("No maps selected; pass --eval_maps or configure valid held-out maps")
    os.makedirs(args.output_dir, exist_ok=True)
    results: dict[str, list[dict[str, Any]]] = {"RL baseline": [], "OPOSM CEM-MPC": []}

    # Load SB3 policies before importing the F1TENTH simulator/OpenGL modules.
    baseline_models = []
    for path in args.baseline_checkpoints:
        try:
            algorithm, policy = load_baseline(path, args.baseline_algorithm)
        except Exception as exc:
            parser.error(f"Could not load SB3 baseline '{path}': {exc}")
        expected_obs_dim = cfg.data.lidar_dim + 2
        obs_shape = getattr(policy.observation_space, "shape", None)
        obs_dim = int(np.prod(obs_shape)) if obs_shape else None
        if obs_dim != expected_obs_dim:
            parser.error(
                f"{path}: policy expects observation shape {obs_shape}, but the RL environment "
                f"provides {expected_obs_dim} ({cfg.data.lidar_dim} LiDAR + speed + steering). "
                "Use a checkpoint trained with this observation layout or update the layout "
                "to match its training metadata.")
        print(f"[evaluate_all] loaded {algorithm.upper()} baseline with Stable-Baselines3 "
              f"{path} (obs_shape={obs_shape}, device={policy.device})", flush=True)
        baseline_models.append((algorithm, policy))

    model = load_oposm(cfg, args.model_checkpoint, device)

    for map_name in maps:
        for seed in args.seeds:
            print(f"[evaluate_all] map={map_name} seed={seed}")
            # Multiple baseline checkpoints are pooled as model seeds and
            # over the same evaluation seeds, with checkpoint identity retained.
            for model_idx, (algorithm, baseline) in enumerate(baseline_models):
                _seed_everything(seed)
                row = evaluate_ppo_episode(baseline, args.config, map_name, seed, args.max_episode_seconds)
                row["algorithm"] = algorithm.upper()
                row["checkpoint"] = os.path.abspath(args.baseline_checkpoints[model_idx])
                row["checkpoint_index"] = model_idx
                results["RL baseline"].append(row)
            _seed_everything(seed)
            row = evaluate_oposm_episode(model, cfg, state_norm, action_norm,
                                         device, map_name, seed)
            row["checkpoint"] = os.path.abspath(args.model_checkpoint)
            results["OPOSM CEM-MPC"].append(row)

    report = {
        "metadata": {
            "config": os.path.abspath(args.config),
            "model_checkpoint": os.path.abspath(args.model_checkpoint),
            "baseline_checkpoints": [os.path.abspath(p) for p in args.baseline_checkpoints],
            "maps": maps, "seeds": args.seeds,
            "success_definition": "at least one complete lap in an episode",
            "progress_definition": "maximum centerline-segment-projected distance from the seeded start, wrapped to one lap, divided by track length and capped at one lap; same seeded centerline start for RL and OPOSM",
            "lap_time_definition": "simulated time to complete a lap; aggregated over successful episodes only",
            "intervals": "95% normal confidence intervals for means; Wilson intervals for success rates",
        },
        "summary": {method: summarize(runs) for method, runs in results.items()},
        "per_map_summary": {
            map_name: {
                method: summarize([r for r in runs if r["map_name"] == map_name])
                for method, runs in results.items()
            }
            for map_name in maps
        },
        "runs": results,
    }
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(report, f, indent=2)
    make_figures(results, args.output_dir)
    print(f"[evaluate_all] saved JSON and figures under {args.output_dir}")


if __name__ == "__main__":
    main()
