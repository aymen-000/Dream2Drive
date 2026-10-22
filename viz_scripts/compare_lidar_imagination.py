from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from envs.f1tenth_env import F1TenthEnvConfig, F1TenthEnvironment
from models.skill_model import OPOSMSkillModel
from planning.cem_planner import CEMPlanner, CEMPlannerConfig, mpc_race
from planning.reward import LidarStructuralReward
from scripts.config_utils import load_config, resolve_device
from utils.normalization import Normalizer
from utils.track import Centerline, discover_map


def plot_comparison(real_scan: np.ndarray, imagined_scan: np.ndarray,
                    max_range: float, map_name: str, output_path: str) -> None:
    """Save a simple real-versus-imagined scan overlay."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                        "axes.spines.right": False, "legend.frameon": False})
    beams = np.arange(len(real_scan))
    mae = float(np.mean(np.abs(real_scan - imagined_scan)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(beams, real_scan, color="#1f77b4", lw=1.8, label="Real LiDAR")
    ax.plot(beams, imagined_scan, color="#ff7f0e", lw=1.8, ls="--",
            label="CEM / TAWM imagined LiDAR")
    ax.set(title=f"{map_name} — final planned skill (MAE {mae:.2f} m)",
           xlabel="LiDAR beam", ylabel="Range (m)")
    ax.set_ylim(0, max_range)
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_map(map_name, args, cfg, model, state_norm, action_norm, device):
    map_yaml, map_ext, centerline_csv = discover_map(cfg.data.maps_root, map_name)
    centerline = Centerline.from_csv(centerline_csv)
    if not 0 <= args.start_index < len(centerline.points):
        raise ValueError(f"--start_index must be in [0, {len(centerline.points) - 1}] for {map_name}")
    xy0 = centerline.points[args.start_index]
    next_xy = centerline.points[(args.start_index + 1) % len(centerline.points)]
    yaw0 = float(np.arctan2(next_xy[1] - xy0[1], next_xy[0] - xy0[0]))
    initial_state = np.array([xy0[0], xy0[1], 0, 0, yaw0, 0, 0], dtype=np.float32)

    env_cfg = F1TenthEnvConfig(
        map_yaml=map_yaml, map_ext=map_ext, timestep=cfg.f1tenth_env.timestep,
        control_hz=cfg.f1tenth_env.control_hz,
        max_steer_rad=cfg.f1tenth_env.max_steer_rad,
        max_speed_mps=cfg.f1tenth_env.max_speed_mps,
        render=False, lidar_dim=model.lidar_dim,
        action_repeat=cfg.data.subsample_stride,
    )
    reward_fn = LidarStructuralReward(
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
        elite_frac=cfg.planning.elite_frac,
        cem_iters=cfg.planning.cem_iters,
        init_std=cfg.planning.init_std,
        device=device,
        save_dir=None,
    )
    planner = CEMPlanner(model, reward_fn, planner_cfg)
    env = F1TenthEnvironment(model, state_norm, action_norm, env_cfg, initial_state, device=device)
    try:
        result = mpc_race(
            planner, env, centerline.points,
            skill_horizon=cfg.data.subtraj_len - 1,
            max_skills=cfg.planning.max_skills_per_episode,
            max_laps=cfg.planning.max_laps,
            replan_every_skills=cfg.planning.replan_every_skills,
        )
    finally:
        env.close()

    max_range = model.lidar_max_range
    real_scans = np.clip(result["real_lidar_trajectory"], 0.0, max_range)
    imagined_scans = np.clip(result["imagined_lidar_trajectory"], 0.0, max_range)
    poses = result["lidar_poses"]
    executed_zs = result["executed_zs"]
    print(f"[{map_name}] completed {result['laps_completed']} lap(s); "
          f"collision={result['collision']}")
    for step, (real, imagined) in enumerate(zip(real_scans[1:], imagined_scans[1:]), start=1):
        mae = float(np.mean(np.abs(real - imagined)))
        print(f"[{map_name}] skill={step}/{len(executed_zs)}, LiDAR MAE={mae:.3f} m")

    os.makedirs(args.out, exist_ok=True)
    stem = f"{map_name}_cem_lidar_imagination"
    npz_path = os.path.join(args.out, stem + ".npz")
    np.savez_compressed(npz_path, real_scans=real_scans,
                        imagined_scans=imagined_scans,
                        poses=poses,
                        centerline=centerline.points,
                        executed_zs=executed_zs,
                        trajectory=result["trajectory_raw"],
                        laps_completed=np.int64(result["laps_completed"]))
    # Plot the last skill that was actually executed; ignore any unexecuted
    # plan suffix after a collision.
    final_index = len(executed_zs)
    png_path = os.path.join(args.out, stem + ".png")
    plot_comparison(real_scans[final_index], imagined_scans[final_index],
                    model.lidar_max_range, map_name, png_path)
    print(f"[{map_name}] wrote {png_path} and {npz_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--maps", nargs="+", required=True,
                        help="One or more track names, e.g. Spielberg Austin")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Centerline point used as the common initial pose")
    parser.add_argument("--out", default="outputs/lidar_imagination")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.planning.device)
    state_norm = Normalizer.load(os.path.join(cfg.train.checkpoint_dir, "state_normalizer.json"))
    action_norm = Normalizer.load(os.path.join(cfg.train.checkpoint_dir, "action_normalizer.json"))
    ckpt = torch.load(args.checkpoint, map_location=device)
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
        tawm_predict_delta=ckpt.get("tawm_predict_delta", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for map_name in args.maps:
        run_map(map_name, args, cfg, model, state_norm, action_norm, device)


if __name__ == "__main__":
    main()
