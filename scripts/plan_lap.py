import argparse
import os
import sys

from envs.f1tenth_env import F1TenthEnvConfig, F1TenthEnvironment
from planning.mpc_controller import KinematicBicycleParams, KinematicMPCController, MPCConfig, classical_mpc_race

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch

from models.skill_model import OPOSMSkillModel
from planning.cem_planner import CEMPlanner, CEMPlannerConfig, mpc_race
from planning.reward import CenterlineProgressReward
from utils.normalization import Normalizer
from utils.track import Centerline, discover_map 
from scripts.config_utils import load_config, resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--map", type=str, default="Spielberg", help="Map name, e.g. Spielberg")
    parser.add_argument("--env", type=str, default="f1tentch", choices=["tawm", "kinematic", "f1tenth"])
    parser.add_argument("--render", action="store_true",
                        help="Render the F1TENTH window (slower; disabled by default).")
    parser.add_argument("--out", type=str, default="outputs/plan_lap")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.planning.device)
    os.makedirs(args.out, exist_ok=True)

    reward_type = cfg.planning.reward
    state_norm = Normalizer.load(os.path.join(cfg.train.checkpoint_dir, "state_normalizer.json"))
    action_norm = Normalizer.load(os.path.join(cfg.train.checkpoint_dir, "action_normalizer.json"))

    map_yaml, map_ext, centerline_csv = discover_map(cfg.data.maps_root, args.map)
    centerline = Centerline.from_csv(centerline_csv)
    waypoints_raw = centerline.subsample_waypoints(cfg.planning.waypoint_spacing_m)

    heading0 = np.arctan2(*(centerline.points[1] - centerline.points[0])[::-1])
    initial_state_raw = np.array(
        [centerline.points[0, 0], centerline.points[0, 1], 0.0, 0.0, heading0, 0.0, 0.0],
        dtype=np.float32,
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = OPOSMSkillModel(
        state_dim=ckpt.get("state_dim", len(cfg.data.state_features)),
        action_dim=ckpt.get("action_dim", len(cfg.data.action_features)),
        z_dim=ckpt.get("z_dim", cfg.model.z_dim),
        lidar_dim=ckpt.get("lidar_dim", cfg.data.lidar_dim),
        hidden_dim=cfg.model.hidden_dim,
        gru_hidden_dim=cfg.model.gru_hidden_dim, gru_bidirectional=cfg.model.gru_bidirectional,
        min_std=cfg.model.min_std, obs_embed_dim=ckpt.get("obs_embed_dim", cfg.model.obs_embed_dim),
        lidar_max_range=cfg.data.lidar_max_range,
        # Old checkpoints predicted absolute state; new residual checkpoints
        # carry this flag so they remain unambiguous at planning time.
        tawm_predict_delta=ckpt.get("tawm_predict_delta", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

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
    # TODO : implment TAL reward function  baseline
    planner = CEMPlanner(model, state_norm, reward_fn, planner_cfg)
    skill_horizon = cfg.data.subtraj_len - 1



    env_cfg = F1TenthEnvConfig(
            map_yaml=map_yaml, map_ext=map_ext, timestep=cfg.f1tenth_env.timestep,
            control_hz=cfg.f1tenth_env.control_hz, max_steer_rad=cfg.f1tenth_env.max_steer_rad,
            max_speed_mps=cfg.f1tenth_env.max_speed_mps,
            render=cfg.f1tenth_env.render or args.render,
            # The live simulator's native scan length is installation
            # dependent.  Match the checkpoint's encoder input exactly.
            lidar_dim=model.lidar_dim,
    )
    env = F1TenthEnvironment(model, state_norm, action_norm, env_cfg, initial_state_raw, device=device)

    print(f"[plan] map={args.map} num_waypoints={waypoints_raw.shape[0]} env={args.env}")
    try:
        result = mpc_race(
            planner, env, waypoints_raw, skill_horizon=skill_horizon,
            max_skills=cfg.planning.max_skills_per_episode, max_laps=cfg.planning.max_laps,
            replan_every_skills=cfg.planning.replan_every_skills,
        )
    finally:
        if hasattr(env, "close"):
            env.close()

    print(f"[plan]({result['laps_completed']} laps) in {result['executed_zs'].shape[0]} skills.")
    if result.get("stopped_for_predicted_collision", False):
        print("[plan] stopped before executing an imagined collision; no safe MPC prefix was found.")
    if getattr(env, "crashed", False):
        print(f"[plan] WARNING: collision during execution (control step {env.crash_step}).")

    traj = result["trajectory_raw"]
    np.savez(os.path.join(args.out, f"{args.map}_plan.npz"),
              trajectory=traj, executed_zs=result["executed_zs"], waypoints=waypoints_raw)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(centerline.points[:, 0], centerline.points[:, 1], color="lightgray",
            linewidth=1, label="Centerline")
    ax.plot(traj[:, 0], traj[:, 1], color="crimson", linewidth=2, marker="o",
            markersize=3, label=f"OPOSM plan ({args.env} execution)")
    ax.scatter(waypoints_raw[:, 0], waypoints_raw[:, 1], color="gold", edgecolor="black",
               s=60, marker="X", zorder=5, label="Waypoints")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"OPOSM zero-shot lap plan: {args.map}")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    out_png = os.path.join(args.out, f"{args.map}_plan.png")
    fig.savefig(out_png, dpi=150)
    print(f"[plan] saved trajectory plot -> {out_png}")



if __name__ == "__main__":
    main()
