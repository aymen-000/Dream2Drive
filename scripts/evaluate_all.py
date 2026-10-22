from __future__ import annotations

import argparse
import csv
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
try:
    from sb3_contrib import RecurrentPPO
except ImportError:  # PPO, SAC, and IQL remain usable without the optional contrib package.
    RecurrentPPO = None

from scripts.config_utils import load_config, resolve_device
from utils.track import Centerline, discover_map, progress_fraction, relative_progress_m


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


def discover_baseline_checkpoints(directories: list[str]) -> list[str]:
    """Discover known RL checkpoint naming patterns in one or more folders."""
    prefixes = ("ppo_seed_", "ppo_lstm_seed_", "ppo_f1tenth_seed",
                "ppo_lstm_f1tenth_seed", "iql_seed_", "iql_f1tenth_seed",
                "sac_seed_", "sac_f1tenth_seed")
    found = []
    for directory in directories:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")
        for filename in sorted(os.listdir(directory)):
            lower = filename.lower()
            if lower.endswith((".zip", ".pt")) and lower.startswith(prefixes):
                found.append(os.path.abspath(os.path.join(directory, filename)))
    return sorted(set(found))


def load_baseline(path: str, requested_algorithm: str):
    """Load a feed-forward/recurrent SB3 policy or an IQL actor checkpoint."""
    requested_algorithm = requested_algorithm.lower().replace("-", "_")
    if requested_algorithm == "iql" or (requested_algorithm == "auto" and path.lower().endswith(".pt")):
        from rl_baselines.iql import IQL
        model, metadata = IQL.load(path, device="cpu")
        model.baseline_metadata = metadata
        return "iql", model
    with zipfile.ZipFile(path) as archive:
        saved_data = json.loads(archive.read("data"))
    sidecar_path = os.path.splitext(path)[0] + ".json"
    sidecar = {}
    if os.path.isfile(sidecar_path):
        with open(sidecar_path) as sidecar_file:
            sidecar = json.load(sidecar_file)
    if requested_algorithm == "auto" and sidecar.get("algorithm"):
        algorithm = str(sidecar["algorithm"]).lower().replace("-", "_")
    elif requested_algorithm == "auto":
        policy_class = str(saved_data.get("policy_class", "")).lower()
        if "recurrent" in policy_class or "sb3_contrib" in policy_class:
            algorithm = "ppo_lstm"
        elif "train_freq" in saved_data:
            algorithm = "sac"
        elif "n_steps" in saved_data:
            algorithm = "ppo"
        else:
            raise ValueError(f"Cannot infer PPO/SAC algorithm from checkpoint metadata: {path}")
    else:
        algorithm = requested_algorithm
    model_classes = {"ppo": PPO, "sac": SAC, "ppo_lstm": RecurrentPPO}
    model_class = model_classes[algorithm]
    if model_class is None:
        raise ImportError(
            "PPO-LSTM loading requires sb3-contrib, but it is not installed in "
            "this Python environment. Install the project dependencies with "
            "`python -m pip install -r requirements.txt` and rerun evaluation."
        )
    # Older SB3 SAC archives serialized use_sde both at model and policy level.
    # SAC.__init__ forwards the model value explicitly, so remove the duplicate
    # policy kwarg while keeping the top-level saved value authoritative.
    custom_objects = None
    if algorithm == "sac":
        policy_kwargs = dict(saved_data.get("policy_kwargs", {}))
        policy_kwargs.pop("use_sde", None)
        custom_objects = {"policy_kwargs": policy_kwargs}
    model = model_class.load(path, device="cpu", custom_objects=custom_objects)
    model.baseline_metadata = sidecar
    return algorithm, model


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
                           seed: int, max_seconds: float) -> dict[str, Any]:
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
        skill_horizon = max(1, int(cfg.data.subtraj_len) - 1)
        skill_duration = (skill_horizon * int(cfg.data.subsample_stride)
                          / float(cfg.f1tenth_env.control_hz))
        time_limited_skills = max(1, int(max_seconds / skill_duration))
        result = mpc_race(planner, env, waypoints, skill_horizon=skill_horizon,
                          max_skills=min(cfg.planning.max_skills_per_episode,
                                         time_limited_skills),
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
            "elapsed_time_sec": float(env.get_elapsed_time()),
        }
    finally:
        env.close()


def evaluate_ppo_episode(model, config_path, map_name, seed: int, max_seconds: float,
                         algorithm: str = "ppo") -> dict[str, Any]:
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
        total_reward = speed_sum = target_speed_sum = 0.0
        steps = 0
        recurrent_state = None
        episode_start = np.ones((1,), dtype=bool)
        while not (terminated or truncated):
            if algorithm == "ppo_lstm":
                action, recurrent_state = model.predict(
                    obs, state=recurrent_state, episode_start=episode_start, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_start = np.asarray([terminated or truncated], dtype=bool)
            max_progress = max(max_progress, float(info.get("progress_m", 0.)))
            total_reward += float(reward)
            speed_sum += float(info["speed_mps"])
            target_speed_sum += float(info["target_speed_mps"])
            steps += 1
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
            "elapsed_time_sec": float(info.get("elapsed_time_sec", 0.)),
            "episode_return": total_reward,
            "avg_speed_mps": speed_sum / max(steps, 1),
            "avg_target_speed_mps": target_speed_sum / max(steps, 1),
        }
    finally:
        env.close()


def evaluate_iql_episode(model, config_path, map_name, seed: int,
                         max_seconds: float) -> dict[str, Any]:
    from rl_baselines.ppo_env import F1TenthPPOEnv

    env = F1TenthPPOEnv(map_name, randomize_start=True,
                        max_episode_seconds=max_seconds, config_path=config_path)
    try:
        start_index = int(np.random.default_rng(seed).integers(len(env.points)))
        obs, _ = env.reset(seed=seed, options={"start_index": start_index})
        terminated = truncated = False
        info: dict[str, Any] = {}
        max_progress = 0.
        total_reward = speed_sum = target_speed_sum = 0.0
        steps = 0
        while not (terminated or truncated):
            action = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            max_progress = max(max_progress, float(info.get("progress_m", 0.)))
            total_reward += float(reward)
            speed_sum += float(info["speed_mps"])
            target_speed_sum += float(info["target_speed_mps"])
            steps += 1
        track_len = float(info.get("total_track_m", 0.))
        laps = int(info.get("laps_completed", 0))
        return {"method": "IQL", "map_name": map_name, "seed": seed,
                "start_index": start_index, "laps_completed": laps, "success": laps > 0,
                "collision": bool(info.get("collision", False)),
                "max_progress_fraction": progress_fraction(max_progress, track_len),
                "max_progress_m": max_progress, "track_length_m": track_len,
                "lap_times_sec": [float(t) for t in info.get("lap_times_sec", [])],
                "elapsed_time_sec": float(info.get("elapsed_time_sec", 0.)),
                "episode_return": total_reward,
                "avg_speed_mps": speed_sum / max(steps, 1),
                "avg_target_speed_mps": target_speed_sum / max(steps, 1)}
    finally:
        env.close()


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    progress = [r["max_progress_fraction"] for r in runs]
    successes = [r for r in runs if r["success"]]
    lap_times = [t for r in successes for t in r["lap_times_sec"]]
    def mean_if_present(key: str):
        values = [r[key] for r in runs if key in r and r[key] is not None]
        return float(np.mean(values)) if values else None

    return {
        "episodes": len(runs),
        "success_rate": float(np.mean([r["success"] for r in runs])) if runs else None,
        "successes": len(successes),
        "max_progress_fraction_mean": float(np.mean(progress)) if progress else None,
        "max_progress_fraction_std": float(np.std(progress, ddof=1)) if len(progress) > 1 else 0.,
        "successful_lap_time_mean_sec": float(np.mean(lap_times)) if lap_times else None,
        "successful_lap_time_std_sec": float(np.std(lap_times, ddof=1)) if len(lap_times) > 1 else None,
        "collision_rate": float(np.mean([r["collision"] for r in runs])) if runs else None,
        "avg_elapsed_time_sec": mean_if_present("elapsed_time_sec"),
        "avg_speed_mps": mean_if_present("avg_speed_mps"),
        "avg_target_speed_mps": mean_if_present("avg_target_speed_mps"),
    }


def _mean_ci(values: list[float]) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if not len(x):
        return float("nan"), float("nan")
    # Normal 95% interval; report one observation as a point without error bars.
    return float(np.mean(x)), float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.


def paired_comparisons(checkpoint_runs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Compare checkpoints only on identical map/seed episodes."""
    labels = list(checkpoint_runs)
    comparisons = {}
    for left_idx, left_label in enumerate(labels):
        for right_label in labels[left_idx + 1:]:
            left = {(row["map_name"], row["seed"]): row
                    for row in checkpoint_runs[left_label]}
            right = {(row["map_name"], row["seed"]): row
                     for row in checkpoint_runs[right_label]}
            shared = sorted(left.keys() & right.keys())
            if not shared:
                continue
            success_deltas = [int(right[key]["success"]) - int(left[key]["success"])
                              for key in shared]
            progress_deltas = [right[key]["max_progress_fraction"] -
                               left[key]["max_progress_fraction"] for key in shared]
            comparisons[f"{left_label} vs {right_label}"] = {
                "paired_episodes": len(shared),
                "success_rate_difference_right_minus_left": float(np.mean(success_deltas)),
                "max_progress_fraction_difference_right_minus_left": float(np.mean(progress_deltas)),
            }
    return comparisons


def make_figures(results: dict[str, list[dict[str, Any]]], out_dir: str) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.labelsize": 10, "legend.fontsize": 9,
                         "figure.dpi": 150, "savefig.dpi": 300,
                         "axes.spines.top": False, "axes.spines.right": False})
    methods = list(results)
    palette = ["#4C78A8", "#F58518", "#B279A2", "#E45756", "#54A24B", "#72B7B2"]
    colors = {method: ("#54A24B" if method == "OPOSM CEM-MPC"
                       else palette[i % len(palette)])
              for i, method in enumerate(methods)}
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
    parser.add_argument("--baseline_checkpoints", nargs="+", default=None,
                        help="PPO/PPO-LSTM/SAC .zip or IQL .pt checkpoints; auto detects each file.")
    parser.add_argument("--checkpoint_dirs", nargs="+", default=None,
                        help="Discover RL checkpoints here; defaults to baseline_checkpoints/checkpoints when no explicit checkpoint files are supplied.")
    parser.add_argument("--baseline_algorithm", choices=("auto", "ppo", "ppo_lstm", "sac", "iql"), default="auto",
                        help="Override auto detection when all supplied checkpoints use one algorithm.")
    parser.add_argument("--eval_maps", nargs="+", default=None)
    parser.add_argument("--include_train_maps", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--max_episode_seconds", type=float, default=120.)
    parser.add_argument("--output_dir", default="outputs/evaluation")
    args = parser.parse_args()
    if not args.seeds:
        parser.error("Provide at least one evaluation seed")
    args.baseline_checkpoints = list(args.baseline_checkpoints or [])
    checkpoint_dirs = args.checkpoint_dirs
    if checkpoint_dirs is None and not args.baseline_checkpoints:
        default_checkpoint_dir = "baseline_checkpoints/checkpoints"
        if os.path.isdir(default_checkpoint_dir):
            checkpoint_dirs = [default_checkpoint_dir]
    if checkpoint_dirs:
        try:
            args.baseline_checkpoints.extend(discover_baseline_checkpoints(checkpoint_dirs))
        except FileNotFoundError as exc:
            parser.error(str(exc))
    args.baseline_checkpoints = sorted(set(args.baseline_checkpoints))
    if not args.baseline_checkpoints:
        parser.error("Provide --baseline_checkpoints or --checkpoint_dirs")

    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)
    from utils.normalization import Normalizer

    state_norm = Normalizer.load(os.path.join(args.normalizer_dir, "state_normalizer.json"))
    action_norm = Normalizer.load(os.path.join(args.normalizer_dir, "action_normalizer.json"))
    train_maps = set(cfg.collection.map_names)
    checkpoint_metadata: dict[str, dict[str, Any]] = {}
    for path in args.baseline_checkpoints:
        if not os.path.isfile(path):
            parser.error(f"Baseline checkpoint does not exist: {path}")
        sidecar = os.path.splitext(path)[0] + ".json"
        metadata: dict[str, Any] = {}
        if os.path.isfile(sidecar):
            with open(sidecar) as metadata_file:
                metadata = json.load(metadata_file)
        elif path.lower().endswith(".pt"):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
            except Exception as exc:
                parser.error(f"Could not read IQL checkpoint metadata from {path}: {exc}")
        checkpoint_metadata[path] = metadata
        train_maps.update(metadata.get("training_maps", []))
        train_maps.update(metadata.get("dataset_maps", []))

    maps = list(args.eval_maps) if args.eval_maps else [
        m for m in discover_available_maps(cfg.data.maps_root)
        if args.include_train_maps or m not in train_maps]
    if not maps:
        parser.error("No maps selected; pass --eval_maps or configure valid held-out maps")
    print(f"Evaluating every available held-out map ({len(maps)}): {maps}")
    for map_name in maps:
        try:
            discover_map(cfg.data.maps_root, map_name)
        except Exception as exc:
            parser.error(f"Invalid evaluation map '{map_name}': {exc}")
    overlap = sorted(set(maps) & train_maps)
    if overlap and not args.include_train_maps:
        parser.error(f"Evaluation maps overlap with model training maps: {overlap}")
    os.makedirs(args.output_dir, exist_ok=True)
    results: dict[str, list[dict[str, Any]]] = {"OPOSM CEM-MPC": []}

    # Load SB3 policies before importing the F1TENTH simulator/OpenGL modules.
    baseline_models = []
    for path in args.baseline_checkpoints:
        try:
            algorithm, policy = load_baseline(path, args.baseline_algorithm)
        except Exception as exc:
            parser.error(f"Could not load baseline '{path}': {exc}")
        expected_obs_dim = cfg.data.lidar_dim + 2
        obs_shape = getattr(getattr(policy, "observation_space", None), "shape", None)
        obs_dim = (int(policy.config.obs_dim) if algorithm == "iql" else
                   (int(np.prod(obs_shape)) if obs_shape else None))
        action_shape = ((int(policy.config.action_dim),) if algorithm == "iql" else
                        getattr(getattr(policy, "action_space", None), "shape", None))
        if obs_dim != expected_obs_dim:
            parser.error(
                f"{path}: policy expects observation shape {obs_shape}, but the RL environment "
                f"provides {expected_obs_dim} ({cfg.data.lidar_dim} LiDAR + speed + steering). "
                "Use a checkpoint trained with this observation layout or update the layout "
                "to match its training metadata.")
        if action_shape != (2,):
            parser.error(f"{path}: checkpoint action shape is {action_shape}; expected (2,) "
                         "for normalized steering and target speed.")
        metadata = checkpoint_metadata.get(path, {})
        policy.baseline_metadata = metadata
        if algorithm == "iql":
            print(f"[evaluate_all] loaded IQL policy {path} "
                  f"(obs_dim={obs_dim}, device={policy.device})", flush=True)
        else:
            print(f"[evaluate_all] loaded {algorithm.upper()} baseline {path} "
                  f"(obs_shape={obs_shape}, device={policy.device})", flush=True)
        baseline_models.append((algorithm, policy, path, metadata))

    model = load_oposm(cfg, args.model_checkpoint, device)

    checkpoint_results: dict[str, list[dict[str, Any]]] = {}
    for map_name in maps:
        for seed in args.seeds:
            print(f"[evaluate_all] map={map_name} seed={seed}")
            # Multiple baseline checkpoints are pooled as model seeds and
            # over the same evaluation seeds, with checkpoint identity retained.
            for model_idx, (algorithm, baseline, checkpoint, metadata) in enumerate(baseline_models):
                _seed_everything(seed)
                if algorithm == "iql":
                    row = evaluate_iql_episode(baseline, args.config, map_name, seed,
                                               args.max_episode_seconds)
                    method = "IQL"
                else:
                    row = evaluate_ppo_episode(baseline, args.config, map_name, seed,
                                               args.max_episode_seconds, algorithm)
                    method = "PPO-LSTM" if algorithm == "ppo_lstm" else algorithm.upper()
                row["method"] = method
                row["algorithm"] = "PPO-LSTM" if algorithm == "ppo_lstm" else algorithm.upper()
                row["checkpoint"] = os.path.abspath(checkpoint)
                row["training_seed"] = metadata.get("seed")
                row["checkpoint_index"] = model_idx
                results.setdefault(method, []).append(row)
                checkpoint_label = f"{method}::{os.path.abspath(checkpoint)}"
                checkpoint_results.setdefault(checkpoint_label, []).append(row)
            _seed_everything(seed)
            row = evaluate_oposm_episode(model, cfg, state_norm, action_norm,
                                         device, map_name, seed,
                                         args.max_episode_seconds)
            row["checkpoint"] = os.path.abspath(args.model_checkpoint)
            results["OPOSM CEM-MPC"].append(row)
            checkpoint_results.setdefault(
                f"OPOSM CEM-MPC::{os.path.abspath(args.model_checkpoint)}", []
            ).append(row)

    all_rows = [row for rows in results.values() for row in rows]
    run_fields = sorted({key for row in all_rows for key in row})
    with open(os.path.join(args.output_dir, "evaluation_runs.csv"), "w", newline="") as run_file:
        writer = csv.DictWriter(run_file, fieldnames=run_fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})

    checkpoint_summaries = {
        label: summarize(rows) for label, rows in checkpoint_results.items()
    }
    with open(os.path.join(args.output_dir, "evaluation_summary.csv"), "w", newline="") as summary_file:
        fields = ["method", "checkpoint", "episodes", "successes", "success_rate",
                  "collision_rate", "max_progress_fraction_mean",
                  "successful_lap_time_mean_sec", "successful_lap_time_std_sec"]
        writer = csv.DictWriter(summary_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for label, summary in checkpoint_summaries.items():
            method, checkpoint_name = label.split("::", 1)
            writer.writerow({"method": method, "checkpoint": checkpoint_name, **summary})

    with open(os.path.join(args.output_dir, "evaluation_algorithm_summary.csv"),
              "w", newline="") as summary_file:
        fields = ["method", "episodes", "successes", "success_rate", "collision_rate",
                  "max_progress_fraction_mean", "successful_lap_time_mean_sec",
                  "successful_lap_time_std_sec"]
        writer = csv.DictWriter(summary_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for method, rows in results.items():
            writer.writerow({"method": method, **summarize(rows)})

    report = {
        "metadata": {
            "config": os.path.abspath(args.config),
            "model_checkpoint": os.path.abspath(args.model_checkpoint),
            "baseline_checkpoints": [os.path.abspath(p) for p in args.baseline_checkpoints],
            "maps": maps, "seeds": args.seeds,
            "training_maps": sorted(train_maps),
            "episodes_per_map": len(args.seeds),
            "max_episode_seconds": args.max_episode_seconds,
            "matched_evaluation": "Every policy receives the same map/seed-derived centerline start. RL limits are exact by control step; OPOSM limits are rounded down to whole skills and can exceed the limit by less than one skill duration.",
            "iql_observation": "Uses the same full normalized LiDAR, normalized speed, and normalized steering vector used by IQL training; no evaluation-only crop or transform.",
            "success_definition": "at least one complete lap in an episode",
            "progress_definition": "maximum centerline-segment-projected distance from the seeded start, wrapped to one lap, divided by track length and capped at one lap; same seeded centerline start for RL and OPOSM",
            "lap_time_definition": "simulated time to complete a lap; aggregated over successful episodes only",
            "intervals": "95% normal confidence intervals for means; Wilson intervals for success rates",
        },
        "summary": {method: summarize(runs) for method, runs in results.items()},
        "summary_by_checkpoint": checkpoint_summaries,
        "paired_comparisons": paired_comparisons(checkpoint_results),
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
        json.dump(report, f, indent=2, allow_nan=False)
    make_figures(results, args.output_dir)
    print(f"[evaluate_all] saved JSON and figures under {args.output_dir}")


if __name__ == "__main__":
    main()
