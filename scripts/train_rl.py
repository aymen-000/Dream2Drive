import argparse
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_baselines.ppo_env import F1TenthPPOEnv
from scripts.config_utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train_maps", nargs="+", default=None,
                        help="Training maps; defaults to collection.map_names.")
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                        help="Total transitions across all vector environments.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episode_seconds", type=float, default=120.0)
    parser.add_argument("--output_dir", default="checkpoints/rl")
    parser.add_argument("--n_envs", type=int, default=3,
                        help="Number of environments, distributed over training maps.")
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--time_penalty_per_sec", type=float, default=0.2,
                        help="Reward cost per simulated second to discourage stopping.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.timesteps <= 0 or args.n_envs <= 0 or args.n_steps <= 0:
        parser.error("timesteps, n_envs, and n_steps must be positive")
    if args.batch_size <= 0 or args.batch_size > args.n_steps * args.n_envs:
        parser.error("batch_size must be in [1, n_steps * n_envs]")

    cfg = load_config(args.config)
    maps = list(args.train_maps or cfg.collection.map_names)
    if not maps:
        parser.error("At least one training map is required")
    if args.n_envs < len(maps):
        parser.error("n_envs must be at least the number of training maps")
    n_envs = args.n_envs
    selected_maps = [maps[i % len(maps)] for i in range(n_envs)]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    def make_env(map_name):
        def _init():
            return F1TenthPPOEnv(
                map_name, randomize_start=True,
                max_episode_seconds=args.max_episode_seconds,
                config_path=args.config,
                time_penalty_per_sec=args.time_penalty_per_sec,
            )
        return _init

    vec_env = DummyVecEnv([make_env(name) for name in selected_maps])
    vec_env.seed(args.seed)
    model = PPO(
        "MlpPolicy", vec_env, verbose=1, seed=args.seed, device=args.device,
        n_steps=args.n_steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, gamma=args.gamma,
        gae_lambda=args.gae_lambda, clip_range=args.clip_range,
        ent_coef=args.ent_coef, policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=os.path.join(args.output_dir, "tensorboard"),
    )
    print(f"[train_rl] PPO maps={selected_maps} envs={n_envs} "
          f"total_timesteps={args.timesteps} seed={args.seed}")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    stem = f"ppo_f1tenth_seed{args.seed}"
    model.save(os.path.join(args.output_dir, stem))
    metadata = {
        "algorithm": "PPO", "seed": args.seed, "total_timesteps": args.timesteps,
        "training_maps": maps, "environment_maps": selected_maps,
        "randomized_start": True, "max_episode_seconds": args.max_episode_seconds,
        "observation": (f"{cfg.data.lidar_dim} LiDAR ranges scaled to [0, 1], "
                        "followed by speed/max_speed and steering/max_steer in [-1, 1]"),
        "action": "normalized steering and target speed in [-1, 1]",
        "hyperparameters": {
            "n_envs": n_envs, "n_steps": args.n_steps, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "gamma": args.gamma,
            "gae_lambda": args.gae_lambda, "clip_range": args.clip_range,
            "ent_coef": args.ent_coef, "policy_architecture": [256, 256],
        },
        "environment": {
            "lidar_dim": cfg.data.lidar_dim,
            "lidar_max_range_m": cfg.data.lidar_max_range,
            "control_hz": cfg.f1tenth_env.control_hz,
            "timestep_sec": cfg.f1tenth_env.timestep,
            "max_steer_rad": cfg.f1tenth_env.max_steer_rad,
            "max_speed_mps": cfg.f1tenth_env.max_speed_mps,
            "progress_weight": 1.0, "cross_track_weight": 0.15,
            "speed_weight": 0.02,
            "time_penalty_per_sec": args.time_penalty_per_sec,
            "lap_bonus": 50.0,
            "collision_penalty": 50.0,
        },
        "config": os.path.abspath(args.config),
    }
    with open(os.path.join(args.output_dir, stem + ".json"), "w") as f:
        json.dump(metadata, f, indent=2)
    vec_env.close()
    print(f"[train_rl] saved {os.path.join(args.output_dir, stem)}.zip and metadata")


if __name__ == "__main__":
    main()
