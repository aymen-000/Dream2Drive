"""
scripts/make_gifs.py

Render an animated GIF of the CEM-planner / TAWM closed-loop rollout on one
or more tracks, using the exact same planning + execution path as
`scripts/plan_lap.py`. Each frame shows the centerline, the target
waypoints, the trajectory driven so far, and the car (drawn as an oriented
triangle) at its current pose.

Usage
-----
    python scripts/make_gifs.py --config configs/default.yaml \
        --checkpoint checkpoints/oposm_final.pt \
        --maps Spielberg Austin Nuerburgring Catalunya \
        --out figures --fps 20

Notes
-----
- This reuses `F1TenthEnvironment` for closed-loop execution (real LiDAR
  scans from `f1tenth_gym`), exactly like `plan_lap.py`. Set `--render`
  only if you also want to pop up the live pyglet window while frames are
  being recorded; it is not required and will slow things down.
- Frames are subsampled with `--frame-stride` (recorded at `control_hz`
  ticks; a stride of 2 at control_hz=20 gives ~10 rendered frames/sec of
  sim time) and hard-capped with `--max-frames` so GIFs stay a reasonable
  size.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

from envs.f1tenth_env import F1TenthEnvConfig, F1TenthEnvironment
from models.skill_model import OPOSMSkillModel
from planning.cem_planner import CEMPlanner, CEMPlannerConfig, mpc_race
from planning.reward import CenterlineProgressReward
from utils.normalization import Normalizer
from utils.track import Centerline, discover_map
from scripts.config_utils import load_config, resolve_device

# f110_gym's own vehicle footprint constants (rendering.py), used so the
# car glyph in the GIF is drawn to scale with the actual vehicle model.
CAR_LENGTH = 0.58
CAR_WIDTH = 0.31


def load_model(cfg, checkpoint_path: str, device: str) -> OPOSMSkillModel:
    ckpt = torch.load(checkpoint_path, map_location=device)
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
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def plan_and_run(cfg, model, state_norm, action_norm, map_name: str, device: str,
                  render: bool):
    """Runs the exact same zero-shot planning + closed-loop execution as
    `plan_lap.main()` and returns the full per-tick trajectory plus the
    track centerline/waypoints needed to render it."""
    map_yaml, map_ext, centerline_csv = discover_map(cfg.data.maps_root, map_name)
    centerline = Centerline.from_csv(centerline_csv)
    waypoints_raw = centerline.subsample_waypoints(cfg.planning.waypoint_spacing_m)

    heading0 = np.arctan2(*(centerline.points[1] - centerline.points[0])[::-1])
    initial_state_raw = np.array(
        [centerline.points[0, 0], centerline.points[0, 1], 0.0, 0.0, heading0, 0.0, 0.0],
        dtype=np.float32,
    )

    reward_fn = CenterlineProgressReward(
        waypoints_xy=waypoints_raw,
        waypoint_bonus=cfg.planning.waypoint_bonus_reward,
        progress_weight=cfg.planning.progress_reward_weight,
        track_half_width=cfg.planning.track_half_width_m,
        off_track_penalty=cfg.planning.off_track_penalty,
        reward_mode=cfg.planning.reward,
        backward_penalty=cfg.planning.backward_penalty,
        max_progress_per_step=cfg.planning.max_progress_per_skill_m,
        collision_penalty=cfg.planning.collision_penalty,
    )
    planner_cfg = CEMPlannerConfig(
        plan_length=cfg.planning.plan_length, population_size=cfg.planning.population_size,
        elite_frac=cfg.planning.elite_frac, cem_iters=cfg.planning.cem_iters,
        init_std=cfg.planning.init_std, device=device,
    )
    planner = CEMPlanner(model, state_norm, reward_fn, planner_cfg)
    skill_horizon = cfg.data.subtraj_len - 1

    env_cfg = F1TenthEnvConfig(
        map_yaml=map_yaml, map_ext=map_ext, timestep=cfg.f1tenth_env.timestep,
        control_hz=cfg.f1tenth_env.control_hz, max_steer_rad=cfg.f1tenth_env.max_steer_rad,
        max_speed_mps=cfg.f1tenth_env.max_speed_mps,
        render=render,
        lidar_dim=model.lidar_dim,
    )
    env = F1TenthEnvironment(model, state_norm, action_norm, env_cfg, initial_state_raw, device=device)

    print(f"[make_gifs] planning + executing on {map_name} ...")
    try:
        result = mpc_race(
            planner, env, waypoints_raw, skill_horizon=skill_horizon,
            max_skills=cfg.planning.max_skills_per_episode, max_laps=cfg.planning.max_laps,
            replan_every_skills=cfg.planning.replan_every_skills,
        )
    finally:
        if hasattr(env, "close"):
            env.close()

    print(f"[make_gifs] {map_name}: {result['laps_completed']} lap(s), "
          f"{result['executed_zs'].shape[0]} skills, "
          f"{result['trajectory_raw'].shape[0]} ticks recorded.")
    return result, centerline, waypoints_raw


def render_gif(trajectory: np.ndarray, centerline: Centerline, waypoints: np.ndarray,
               map_name: str, out_path: str, fps: int, frame_stride: int,
               max_frames: int, dpi: int, trail: bool):
    """Renders `trajectory` (per-tick [x, y, steer, vel, yaw, ...] rows) as
    an animated GIF: track centerline + waypoints in the background, the
    driven trail so far, and the car glyph at its current pose."""
    idx = np.arange(0, trajectory.shape[0], max(1, frame_stride))
    if idx.shape[0] > max_frames:
        idx = np.linspace(0, trajectory.shape[0] - 1, max_frames).astype(int)
    traj = trajectory[idx]

    x_col, y_col, yaw_col, v_col = 0, 1, 4, 3

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(centerline.points[:, 0], centerline.points[:, 1], color="lightgray",
            linewidth=1.0, zorder=1, label="Centerline")
    ax.scatter(waypoints[:, 0], waypoints[:, 1], color="gold", edgecolor="black",
               s=40, marker="X", zorder=2, label="Waypoints")

    trail_line, = ax.plot([], [], color="crimson", linewidth=2, zorder=3, label="Driven path")

    half_l, half_w = CAR_LENGTH / 2.0, CAR_WIDTH / 2.0
    car_pts_local = np.array([
        [half_l, 0.0],
        [-half_l, half_w],
        [-half_l, -half_w],
    ])
    car_patch = patches.Polygon(car_pts_local, closed=True, facecolor="royalblue",
                                 edgecolor="black", zorder=5)
    ax.add_patch(car_patch)

    speed_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                          fontsize=11, family="monospace",
                          bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    pad = 3.0
    ax.set_xlim(centerline.points[:, 0].min() - pad, centerline.points[:, 0].max() + pad)
    ax.set_ylim(centerline.points[:, 1].min() - pad, centerline.points[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"OPOSM zero-shot closed-loop rollout: {map_name}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    def _car_polygon(x, y, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        return car_pts_local @ rot.T + np.array([x, y])

    def update(frame_i: int):
        row = traj[frame_i]
        x, y, yaw, v = row[x_col], row[y_col], row[yaw_col], row[v_col]
        car_patch.set_xy(_car_polygon(x, y, yaw))
        if trail:
            trail_line.set_data(traj[: frame_i + 1, x_col], traj[: frame_i + 1, y_col])
        speed_text.set_text(f"tick {idx[frame_i]:5d}\nspeed {v:5.2f} m/s")
        return car_patch, trail_line, speed_text

    ani = FuncAnimation(fig, update, frames=traj.shape[0], blit=False, interval=1000 / fps)
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"[make_gifs] saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--maps", type=str, nargs="+", required=True,
                         help="One or more map names, e.g. --maps Spielberg Austin")
    parser.add_argument("--out", type=str, default="figures",
                         help="Output directory for the rendered GIFs")
    parser.add_argument("--render", action="store_true",
                         help="Also render the live pyglet window during execution (slower).")
    parser.add_argument("--fps", type=int, default=20, help="GIF playback frame rate")
    parser.add_argument("--frame-stride", type=int, default=2,
                         help="Keep every Nth recorded control tick as a GIF frame")
    parser.add_argument("--max-frames", type=int, default=400,
                         help="Hard cap on frames per GIF (uniformly re-subsampled if exceeded)")
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--no-trail", action="store_true",
                         help="Don't draw the accumulated driven-path trail")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.planning.device)
    os.makedirs(args.out, exist_ok=True)

    state_norm = Normalizer.load(os.path.join(cfg.train.checkpoint_dir, "state_normalizer.json"))
    action_norm = Normalizer.load(os.path.join(cfg.train.checkpoint_dir, "action_normalizer.json"))
    model = load_model(cfg, args.checkpoint, device)

    for map_name in args.maps:
        result, centerline, waypoints_raw = plan_and_run(
            cfg, model, state_norm, action_norm, map_name, device, render=args.render,
        )
        out_path = os.path.join(args.out, f"rollout_{map_name.lower()}.gif")
        render_gif(
            trajectory=result["trajectory_raw"], centerline=centerline, waypoints=waypoints_raw,
            map_name=map_name, out_path=out_path, fps=args.fps, frame_stride=args.frame_stride,
            max_frames=args.max_frames, dpi=args.dpi, trail=not args.no_trail,
        )


if __name__ == "__main__":
    main()