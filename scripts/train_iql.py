"""Train IQL offline from the project's fixed Pure Pursuit driving logs."""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch

from rl_baselines.iql import IQL, IQLConfig
from scripts.config_utils import load_config
from utils.track import Centerline, discover_map


def _lidar_features(frame: pd.DataFrame, out_dim: int, max_range: float) -> np.ndarray:
    columns = sorted((c for c in frame.columns if c.startswith("lidar_")),
                     key=lambda c: int(c.split("_")[1]))
    if not columns:
        raise ValueError("Offline CSV has no lidar_* columns; recollect data with the current collector")
    source = np.linspace(0.0, 1.0, len(columns))
    target = np.linspace(0.0, 1.0, out_dim)
    scans = frame[columns].to_numpy(dtype=np.float32)
    scans = np.stack([np.interp(target, source, row) for row in scans])
    return np.clip(scans / max_range, 0.0, 1.0).astype(np.float32)


def load_offline_transitions(config_path: str, maps: list[str] | None = None):
    cfg = load_config(config_path)
    maps = set(maps or cfg.collection.map_names)
    files = sorted(glob.glob(os.path.join(cfg.data.root, "*", "run-*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No run-*.csv logs under {cfg.data.root}; run scripts/generate_offline_data.py first")
    all_obs, all_actions, all_rewards, all_next_obs, all_terminals = [], [], [], [], []
    used_maps = set()
    for file_path in files:
        frame = pd.read_csv(file_path).dropna().reset_index(drop=True)
        if len(frame) < 2:
            continue
        map_name = str(frame["map_name"].iloc[0]) if "map_name" in frame else os.path.basename(os.path.dirname(file_path))
        if map_name not in maps:
            continue
        _, _, centerline_file = discover_map(cfg.data.maps_root, map_name)
        centerline = Centerline.from_csv(centerline_file)
        lidar = _lidar_features(frame, int(cfg.data.lidar_dim), float(cfg.data.lidar_max_range))
        state = np.stack([
            np.clip(frame["vel"].to_numpy(dtype=np.float32) / float(cfg.f1tenth_env.max_speed_mps), 0., 1.),
            np.clip(frame["steer_angle"].to_numpy(dtype=np.float32) / float(cfg.f1tenth_env.max_steer_rad), -1., 1.),
        ], axis=-1)
        obs = np.concatenate([lidar, state], axis=-1).astype(np.float32)
        steer = frame["steer_cmd"].to_numpy(dtype=np.float32) / float(cfg.f1tenth_env.max_steer_rad)
        speed = (frame["speed_cmd"].to_numpy(dtype=np.float32) /
                 float(cfg.f1tenth_env.max_speed_mps)) * 2.0 - 1.0
        actions = np.clip(np.stack([steer, speed], axis=-1), -1., 1.).astype(np.float32)

        xy = frame[["x", "y"]].to_numpy(dtype=np.float32)
        projections = [centerline.project_m(point) for point in xy]
        progress = np.asarray([item[0] for item in projections], dtype=np.float32)
        cross_track = np.asarray([item[1] for item in projections], dtype=np.float32)
        if {"next_x", "next_y"}.issubset(frame.columns):
            next_xy = frame[["next_x", "next_y"]].to_numpy(dtype=np.float32)
            next_projections = [centerline.project_m(point) for point in next_xy]
            next_progress = np.asarray([item[0] for item in next_projections], dtype=np.float32)
            next_cross_track = np.asarray([item[1] for item in next_projections], dtype=np.float32)
            raw_delta = next_progress - progress
            speed_next = (frame["next_vel"].to_numpy(dtype=np.float32)
                          if "next_vel" in frame else np.r_[frame["vel"].to_numpy(dtype=np.float32)[1:],
                                                            frame["vel"].iloc[-1]])
        else:
            # Older logs did not save the post-action pose; use the next row
            # and approximate the final terminal transition.
            raw_delta = np.r_[np.diff(progress), 0.0]
            next_cross_track = np.r_[cross_track[1:], cross_track[-1]]
            speed_next = np.r_[frame["vel"].to_numpy(dtype=np.float32)[1:],
                               frame["vel"].iloc[-1]]
        lap_completed = raw_delta < -0.5 * centerline.total_length
        delta = raw_delta.copy()
        delta[lap_completed] += centerline.total_length
        delta = np.clip(delta, -2.0, 2.0)
        rewards = (delta - .15 * next_cross_track + .02 * np.maximum(speed_next, 0.)
                   - .2 / float(cfg.f1tenth_env.control_hz))
        rewards += lap_completed.astype(np.float32) * 50.0
        collisions = (frame["collision"].to_numpy(dtype=bool) if "collision" in frame
                      else np.zeros(len(frame), dtype=bool))
        rewards = rewards - collisions.astype(np.float32) * 80.0
        terminals = (frame["done"].to_numpy(dtype=bool) if "done" in frame
                     else np.zeros(len(frame), dtype=bool))
        terminals[-1] = True  # every CSV is one finite episode; final next-state is unused.
        all_obs.append(obs)
        all_actions.append(actions)
        all_rewards.append(rewards.astype(np.float32))
        all_next_obs.append(np.concatenate([obs[1:], obs[-1:]], axis=0))
        all_terminals.append(terminals.astype(np.float32))
        used_maps.add(map_name)
    if not all_obs:
        raise ValueError(f"No offline runs matched configured training maps: {sorted(maps)}")
    return (tuple(np.concatenate(parts, axis=0) for parts in
                  (all_obs, all_actions, all_rewards, all_next_obs, all_terminals)),
            sorted(used_maps))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train_maps", nargs="+", default=None)
    parser.add_argument("--updates", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default="checkpoints/offline")
    parser.add_argument("--expectile", type=float, default=.7)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--gamma", type=float, default=.99)
    args = parser.parse_args()
    if args.updates <= 0 or args.batch_size <= 0:
        parser.error("updates and batch_size must be positive")
    cfg = load_config(args.config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    (obs, actions, rewards, next_obs, terminals), used_maps = load_offline_transitions(
        args.config, args.train_maps)
    learner = IQL(IQLConfig(obs_dim=obs.shape[1], gamma=args.gamma,
                            expectile=args.expectile, temperature=args.temperature), device)
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    log_every = max(1, args.updates // 20)
    for step in range(1, args.updates + 1):
        indices = rng.integers(0, len(obs), size=args.batch_size)
        losses = learner.update(obs[indices], actions[indices], rewards[indices],
                                next_obs[indices], terminals[indices])
        if step == 1 or step % log_every == 0:
            print(f"[train_iql] update={step}/{args.updates} "
                  f"value={losses['value_loss']:.4f} critic={losses['critic_loss']:.4f} "
                  f"actor={losses['actor_loss']:.4f}", flush=True)
    stem = f"iql_f1tenth_seed{args.seed}"
    metadata = {"algorithm": "IQL", "seed": args.seed, "updates": args.updates,
                "dataset_root": os.path.abspath(cfg.data.root), "dataset_maps": used_maps,
                "obs_dim": int(obs.shape[1]), "lidar_dim": int(cfg.data.lidar_dim),
                "action_layout": ["normalized_steering", "normalized_target_speed"],
                "config": os.path.abspath(args.config), "gamma": args.gamma,
                "expectile": args.expectile, "temperature": args.temperature,
                "offline_only": True}
    learner.save(os.path.join(args.output_dir, stem + ".pt"), metadata)
    with open(os.path.join(args.output_dir, stem + ".json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[train_iql] saved {os.path.join(args.output_dir, stem)}.pt")


if __name__ == "__main__":
    main()
