import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import gym
from f110_gym.envs.f110_env import F110Env


from data.collector import PurePursuitController, PurePursuitConfig, add_exploration_noise
from utils.track import Centerline, discover_map
from scripts.config_utils import load_config


# --------------------------------------------------------------------------
# LiDAR state representation
# --------------------------------------------------------------------------
# The pose vector (x, y, yaw, ...) is map-specific and not something a real
# car would observe -- it doesn't generalize and leaks map identity into the
# "state". Instead we use the standard LiDAR-beam representation: N evenly
# spaced beams spanning a forward-facing field of view, range-clipped and
# normalized to [0, 1]. Pose is still computed internally (pure pursuit needs
# it to act), it's just no longer part of the state we save for the world
# model.
NUM_BEAMS_OUT = 108
BEAM_FOV_RAD = np.pi          # pi radians, centered on the heading direction
BEAM_MAX_RANGE_M = 10.0       # beams are clipped to this range before scaling
BEAM_COLUMNS = [f"beam_{i:02d}" for i in range(NUM_BEAMS_OUT)]

# f110_gym's default ScanSimulator2D uses 1080 beams over a 270 deg (4.7 rad)
# FOV, symmetric about the heading. We read this from the env if available
# and fall back to these defaults otherwise.
DEFAULT_FULL_NUM_BEAMS = 108
DEFAULT_FULL_FOV_RAD = 4.7


def get_full_scan_geometry(env) -> "tuple[int, float]":
    candidates = [
        lambda: (env.sim.agents[0].scan_simulator.num_beams, env.sim.agents[0].scan_simulator.fov),
        lambda: (env.sim.agents[0].num_beams, env.sim.agents[0].fov),
        lambda: (env.params["num_beams"], env.params["fov"]),
    ]
    for get in candidates:
        try:
            num_beams, fov = get()
            return int(num_beams), float(fov)
        except (AttributeError, KeyError, TypeError):
            continue
    return DEFAULT_FULL_NUM_BEAMS, DEFAULT_FULL_FOV_RAD


def build_beam_sampling_indices(full_num_beams: int, full_fov: float,
                                 out_num_beams: int = NUM_BEAMS_OUT,
                                 out_fov: float = BEAM_FOV_RAD) -> np.ndarray:
    """Precompute, once per env, which indices into the raw full-resolution
    scan correspond to `out_num_beams` evenly spaced beams spanning `out_fov`
    radians centered on the heading direction."""
    full_angles = np.linspace(-full_fov / 2.0, full_fov / 2.0, full_num_beams)
    target_angles = np.linspace(-out_fov / 2.0, out_fov / 2.0, out_num_beams)
    indices = np.array([int(np.argmin(np.abs(full_angles - a))) for a in target_angles])
    return indices


def extract_beams(full_scan: np.ndarray, sample_indices: np.ndarray,
                   max_range: float = BEAM_MAX_RANGE_M) -> np.ndarray:
    """Downsample a raw LiDAR scan and retain ranges in metres.

    ``LidarEncoder`` is the single normalization point shared by training and
    execution.  Saving pre-normalized beams here would make the train-time
    and live-policy inputs differ by a range-scale factor.
    """
    beams = np.asarray(full_scan, dtype=np.float32)[sample_indices]
    beams = np.nan_to_num(beams, nan=max_range, posinf=max_range, neginf=0.0)
    beams = np.clip(beams, 0.0, max_range)
    return beams.astype(np.float32)


def collect_one_run(env, centerline: Centerline, target_speed: float, cfg, rng,
                     num_ticks: int, substeps: int, out_path: str, map_name: str,
                     episode_id: int, randomize_start: bool = True):

    pp_cfg = PurePursuitConfig(
        wheelbase_m=cfg.collection.wheelbase_m, max_steer_rad=cfg.collection.max_steer_rad,
        target_speed_mps=target_speed,
    )
    controller = PurePursuitController(centerline, pp_cfg)

    if randomize_start:
        start_idx = int(rng.randint(0, centerline.num_points))
    else:
        start_idx = 0
    next_idx = (start_idx + 1) % centerline.num_points
    start_xy = centerline.points[start_idx]
    heading = np.arctan2(*(centerline.points[next_idx] - centerline.points[start_idx])[::-1])
    obs, _, done, _ = env.reset(np.array([[start_xy[0], start_xy[1], heading]], dtype=np.float64))

    full_num_beams, full_fov = get_full_scan_geometry(env)
    sample_indices = build_beam_sampling_indices(
        full_num_beams, full_fov, out_num_beams=cfg.data.lidar_dim,
    )
    rows = []
    collided = False
    for step in range(num_ticks):
        try:
            state = np.asarray(env.sim.agents[0].state, dtype=np.float32)
        except AttributeError:
            state = np.array([
                obs["poses_x"][0], obs["poses_y"][0], 0.0,
                obs["linear_vels_x"][0], obs["poses_theta"][0], obs["ang_vels_z"][0], 0.0,
            ], dtype=np.float32)
        scan = np.asarray(obs["scans"][0], dtype=np.float32)  # scan paired with `state` above: both reflect the previous tick's outcome

        x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle = state
        action = controller.act(x, y, yaw_angle, vel)
        if cfg.collection.action_noise:
            action = add_exploration_noise(
                action, rng, steer_std=cfg.collection.steer_noise_std,
                speed_std=cfg.collection.speed_noise_std, max_steer_rad=cfg.collection.max_steer_rad,
            )

        for _ in range(substeps):
            obs, _, done, _ = env.step(np.array([action], dtype=np.float64))
            if done:
                break

        collided = bool(done and obs["collisions"][0])
        is_last_tick = step == (num_ticks - 1)
        step_done = bool(done or is_last_tick)

        row = {
            "map_name": map_name, "episode_id": episode_id, "step": step,
            "x": x, "y": y, "steer_angle": steer_angle, "vel": vel,
            "yaw_angle": yaw_angle, "yaw_rate": yaw_rate, "slip_angle": slip_angle,
            "steer_cmd": action[0], "speed_cmd": action[1],
            "done": step_done, "collision": collided,
        }
        beams = extract_beams(scan, sample_indices, max_range=cfg.data.lidar_max_range)
        for i, beam in enumerate(beams):
            row[f"lidar_{i}"] = float(beam)
        rows.append(row)

        if step_done:
            if collided:
                print(f"  [collect] episode {episode_id}: collision -- ended at tick {len(rows)}")
            else:
                print(f"  [collect] episode {episode_id}: finished cleanly at tick {len(rows)}")
            break

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  [collect] saved {len(df)} rows -> {out_path}")
    return {"episode_id": episode_id, "num_steps": len(rows), "collision": collided}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    rng = np.random.RandomState(cfg.collection.seed)
    substeps = max(1, round(1.0 / (cfg.collection.control_hz * cfg.collection.timestep)))
    run_seconds = cfg.collection.run_seconds
    num_ticks = int(run_seconds * cfg.collection.control_hz)

    # Number of episodes to collect per map. Falls back to laps_per_map for
    # backwards compatibility with older configs.
    num_episodes = getattr(cfg.collection, "episodes_per_map", None) 
    randomize_start = getattr(cfg.collection, "randomize_start", True)

    summary = {}
    for map_name in cfg.collection.map_names:
        map_yaml, map_ext, centerline_csv = discover_map(cfg.data.maps_root, map_name)
        centerline = Centerline.from_csv(centerline_csv)
        out_dir = os.path.join(cfg.collection.out_root, map_name)
        os.makedirs(out_dir, exist_ok=True)

        env = F110Env(map=map_name)

        map_stats = []
        for episode_id in range(num_episodes):
            target_speed = cfg.collection.max_speed_choices[episode_id % len(cfg.collection.max_speed_choices)]
            out_path = os.path.join(out_dir, f"run-{episode_id:04d}_speed{target_speed:.1f}.csv")
            print(f"[collect] map={map_name} episode={episode_id}/{num_episodes - 1} target_speed={target_speed}")
            stats = collect_one_run(env, centerline, target_speed, cfg, rng, num_ticks, substeps,
                                     out_path, map_name, episode_id=episode_id,
                                     randomize_start=randomize_start)
            map_stats.append(stats)

        env.close() if hasattr(env, "close") else None

        n_collisions = sum(1 for s in map_stats if s["collision"])
        total_steps = sum(s["num_steps"] for s in map_stats)
        summary[map_name] = {
            "episodes": len(map_stats), "collisions": n_collisions, "total_steps": total_steps,
        }
        print(f"[collect] map={map_name} done: {len(map_stats)} episodes, "
              f"{n_collisions} collisions, {total_steps} total steps.")

    print("[collect] summary:")
    for map_name, stats in summary.items():
        print(f"  {map_name}: {stats}")
    print("[collect] done.")


if __name__ == "__main__":
    main()
