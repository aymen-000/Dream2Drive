import argparse
import json
import os
import sys
from typing import List, Optional

from envs.f1tenth_env import F1TenthEnvConfig, F1TenthEnvironment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import discover_runs, build_run_arrays, train_test_run_split, OfflineSkillDataset, load_centerline_for_run
from models.skill_model import OPOSMSkillModel
from planning.cem_planner import CEMPlanner, CEMPlannerConfig, mpc_race
from planning.reward import CenterlineProgressReward
from planning.mpc_controller import KinematicMPCController, KinematicBicycleParams, MPCConfig,  classical_mpc_race
from utils.normalization import Normalizer
from scripts.config_utils import load_config, resolve_device


def load_model(cfg, checkpoint_path, device):
    model = OPOSMSkillModel(
        action_dim=len(cfg.data.action_features),
        z_dim=cfg.model.z_dim, hidden_dim=cfg.model.hidden_dim,
        gru_hidden_dim=cfg.model.gru_hidden_dim, gru_bidirectional=cfg.model.gru_bidirectional,
        min_std=cfg.model.min_std,
        tawm_predict_delta=cfg.model.tawm_predict_delta , state_dim=len(cfg.data.state_features) , lidar_dim=cfg.data.lidar_dim
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def tawm_prediction_mse(model, dataloader, device, num_batches=None):
    mses, elbos = [], []
    for i, batch in enumerate(dataloader):
        if num_batches is not None and i >= num_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        terms = model.elbo_terms(batch)
        s0_emb = model.encode(batch["s0"])
        sT_emb = model.encode(batch["sT"])
        mu, _ = model.tawm(s0_emb, terms["z"])
        mses.append(((mu - sT_emb) ** 2).mean().item()) 
        elbos.append(terms["elbo"].mean().item())
    return float(np.mean(mses)), float(np.mean(elbos))


def _summarize_runs(results_for_all_runs):

    track_coverages = [r["final_progress_s"] / r["total_track_s"] for r in results_for_all_runs]
    track_coverage = float(np.mean(track_coverages))
    lapc_completer = float(np.mean([r["laps_completed"] for r in results_for_all_runs]))
    collision_rate = float(np.mean([bool(r.get("collision", False)) for r in results_for_all_runs]))
    return track_coverage, lapc_completer, collision_rate


# --------------------------------------------------------------------------
# Map discovery: evaluate on maps under cfg.data.maps_root that were NOT
# used for training (cfg.collection.map_names), so the planning results
# reflect zero-shot generalization to unseen tracks rather than
# performance on tracks the offline dataset was collected on.
# --------------------------------------------------------------------------
def discover_available_maps(maps_root: str) -> List[str]:
    """Every subdirectory of `maps_root` that `load_centerline_for_run` can
    actually load (i.e. has the map yaml + centerline csv f1tenth_racetracks
    expects). Directories that don't parse as a valid map are skipped with
    a warning rather than failing the whole run."""
    if not os.path.isdir(maps_root):
        raise FileNotFoundError(f"maps_root '{maps_root}' does not exist.")
    candidates = sorted(
        d for d in os.listdir(maps_root) if os.path.isdir(os.path.join(maps_root, d))
    )
    valid = []
    for name in candidates:
        try:
            load_centerline_for_run(maps_root, name)
            valid.append(name)
        except Exception as e:
            print(f"[evaluate] skipping '{name}' under '{maps_root}' (not a usable map: {e})")
    return valid


def select_eval_maps(cfg, explicit: Optional[List[str]] = None,
                      include_train_maps: bool = False) -> List[str]:
    """Returns the map names to evaluate on.

    - `explicit`, if given, wins outright (e.g. `--eval_maps Melbourne Monza`).
    - Otherwise every valid map under `cfg.data.maps_root` is used, EXCLUDING
      the maps listed in `cfg.collection.map_names` (the maps the offline
      dataset -- and therefore the model -- was trained on), unless
      `include_train_maps=True`.
    """
    if explicit:
        return list(explicit)

    train_maps = set(cfg.collection.map_names)
    all_maps = discover_available_maps(cfg.data.maps_root)

    if include_train_maps:
        return all_maps

    held_out = [m for m in all_maps if m not in train_maps]
    if not held_out:
        raise RuntimeError(
            f"No held-out maps found under '{cfg.data.maps_root}' that aren't "
            f"also in cfg.collection.map_names ({sorted(train_maps)}). Add more "
            f"map directories under maps/, or pass --include_train_maps / "
            f"--eval_maps explicitly."
        )
    return held_out


def oposm_planning_eval(model, action_norm, state_norm, cfg, device, eval_runs, maps_root, num_runs=1):
    results_for_all_runs = []
    for run in eval_runs:
        map_yaml , map_ext , centerline = load_centerline_for_run(maps_root, run)
        waypoints = centerline.subsample_waypoints(cfg.planning.waypoint_spacing_m)
        reward_fn = CenterlineProgressReward(
            waypoints,
            waypoint_bonus=cfg.planning.waypoint_bonus_reward,
            progress_weight=cfg.planning.progress_reward_weight,
            off_track_penalty=cfg.planning.off_track_penalty,
            reward_mode=cfg.planning.reward, 
        )
        skill_horizon = cfg.data.subtraj_len - 1
        planner_cfg = CEMPlannerConfig(
            plan_length=cfg.planning.plan_length, population_size=cfg.planning.population_size,
            elite_frac=cfg.planning.elite_frac, cem_iters=cfg.planning.cem_iters,
            init_std=cfg.planning.init_std, device=device,
        )
        planner = CEMPlanner(model, state_norm, reward_fn, planner_cfg)
        env_cfg = F1TenthEnvConfig(
                    map_yaml=map_yaml,
                    map_ext=map_ext, 
                    lidar_dim=cfg.data.lidar_dim
                )
        heading0 = np.arctan2(*(centerline.points[1] - centerline.points[0])[::-1])
        initial_state_raw = np.array(
        [centerline.points[0, 0], centerline.points[0, 1], 0.0, 0.0, heading0, 0.0, 0.0],
        dtype=np.float32,
    )
        env = F1TenthEnvironment(model, state_norm, action_norm, cfg=env_cfg,
                                  initial_state_raw=initial_state_raw, device=device)
        print(f"[evaluate]   planning on '{run}' ...")
        result = mpc_race(
            planner, env, waypoints, skill_horizon=skill_horizon,
            max_skills=cfg.planning.max_skills_per_episode, max_laps=cfg.planning.max_laps,
            replan_every_skills=cfg.planning.replan_every_skills,
        )
        result["map_name"] = run
        results_for_all_runs.append(result)

    track_coverage, lapc_completer, collision_rate = _summarize_runs(results_for_all_runs)

    return results_for_all_runs, track_coverage, lapc_completer, collision_rate


def classical_mpc_eval(cfg, eval_runs, maps_root, num_runs=1):
    results_for_all_runs = []
    for run in eval_runs:
        map_yaml , map_ext , centerline = load_centerline_for_run(maps_root, run)
        waypoints = centerline.subsample_waypoints(cfg.planning.waypoint_spacing_m)
        reward_fn = CenterlineProgressReward(
            waypoints,
            waypoint_bonus=cfg.planning.waypoint_bonus_reward,
            progress_weight=cfg.planning.progress_reward_weight,
            off_track_penalty=cfg.planning.off_track_penalty,
        )
        mpc_cfg = MPCConfig(
            horizon=cfg.mpc.horizon, population_size=cfg.mpc.population_size,
            cem_iters=cfg.mpc.cem_iters, elite_frac=cfg.mpc.elite_frac,
            init_steer_std=cfg.mpc.init_steer_std, init_speed_std=cfg.mpc.init_speed_std,
            dt=cfg.mpc.dt,
        )
        params = KinematicBicycleParams(
            wheelbase_m=cfg.collection.wheelbase_m, max_steer_rad=cfg.collection.max_steer_rad,
        )
        controller = KinematicMPCController(reward_fn, params, mpc_cfg)
        env_cfg = F1TenthEnvConfig(
            map_yaml=map_yaml,
            map_ext=map_ext , 
            lidar_dim=cfg.data.lidar_dim
        ) 
        heading0 = np.arctan2(*(centerline.points[1] - centerline.points[0])[::-1])
        initial_state_raw = np.array(
        [centerline.points[0, 0], centerline.points[0, 1], 0.0, 0.0, heading0, 0.0, 0.0],
        dtype=np.float32,
    )

        env = F1TenthEnvironment(model=None, state_normalizer=None, action_normalizer=None,
                                  cfg=env_cfg, initial_state_raw=initial_state_raw)
        print(f"[evaluate]   kinematic MPC on '{run}' ...")
        result = classical_mpc_race(controller, env, waypoints, control_hz=int(1 / mpc_cfg.dt),
                                     max_seconds=cfg.mpc.max_seconds, max_laps=cfg.planning.max_laps)
        result["map_name"] = run
        results_for_all_runs.append(result)

    track_coverage, lapc_completer, collision_rate = _summarize_runs(results_for_all_runs)

    return results_for_all_runs, track_coverage, lapc_completer, collision_rate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--em_checkpoint", type=str, required=True)
    parser.add_argument("--naive_checkpoint", type=str , default=None)
    parser.add_argument("--normalizer_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_planning_runs", type=int, default=1)
    parser.add_argument("--skip_mpc_baseline", action="store_true", default=True,
                         help="Skip the classical kinematic-MPC baseline comparison.")
    parser.add_argument("--eval_maps", type=str, nargs="+", default=None,
                         help="Explicit list of map names to evaluate on. If omitted, every "
                              "map under cfg.data.maps_root NOT in cfg.collection.map_names "
                              "is used (i.e. held-out, zero-shot generalization maps).")
    parser.add_argument("--include_train_maps", action="store_true", default=False,
                         help="Also evaluate on the maps used for training (ignored if "
                              "--eval_maps is given).")
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)

    state_norm = Normalizer.load(os.path.join(args.normalizer_dir, "state_normalizer.json"))
    action_norm = Normalizer.load(os.path.join(args.normalizer_dir, "action_normalizer.json"))

    eval_runs = select_eval_maps(
        cfg, explicit=args.eval_maps, include_train_maps=args.include_train_maps,
    )
    print(f"[evaluate] training maps (from config): {list(cfg.collection.map_names)}")
    print(f"[evaluate] evaluating on {len(eval_runs)} map(s): {eval_runs}")

    results = {}
    eval_specs = [("EM (ours)", args.em_checkpoint)]
    if args.naive_checkpoint:
        eval_specs.append(("Naive VI", args.naive_checkpoint))

    for label, ckpt_path in eval_specs:
        print(f"\n=== Evaluating: {label} ===")

        model = load_model(cfg, ckpt_path, device)

        results_for_all_runs, track_coverage, lapc_completer, collision_rate = oposm_planning_eval(
            model,
            action_norm,
            state_norm,
            cfg,
            device,
            eval_runs,
            cfg.data.maps_root,
            num_runs=args.num_planning_runs
        )
        results[label] = {
            "eval_maps": eval_runs,
            "planning": {
                "track_coverage": float(track_coverage),
                "laps_completed": float(lapc_completer),
                "collision_rate": float(collision_rate),
                "num_runs": len(results_for_all_runs),

                "runs": [
                    {
                        "map_name": run.get("map_name"),
                        "laps_completed": int(run["laps_completed"]),
                        "final_progress_s": float(run["final_progress_s"]),
                        "total_track_s": float(run["total_track_s"]),
                        "collision": bool(run.get("collision", False)),
                    }
                    for run in results_for_all_runs
                ],
            }
        }


    # ============================================================
    # Evaluate classical MPC baseline
    # ============================================================

    if not args.skip_mpc_baseline:

        print("\n=== Evaluating: Kinematic MPC baseline (classical control) ===")

        results_for_all_runs, track_coverage, lapc_completer, collision_rate = classical_mpc_eval(
            cfg,
            eval_runs,
            cfg.data.maps_root,
            num_runs=args.num_planning_runs
        )

        results["Kinematic MPC"] = {
            "eval_maps": eval_runs,
            "prediction": None,

            "planning": {
                "track_coverage": float(track_coverage),
                "laps_completed": float(lapc_completer),
                "collision_rate": float(collision_rate),
                "num_runs": len(results_for_all_runs),

                "runs": [
                    {
                        "map_name": run.get("map_name"),
                        "laps_completed": int(run["laps_completed"]),
                        "final_progress_s": float(run["final_progress_s"]),
                        "total_track_s": float(run["total_track_s"]),
                        "collision": bool(run.get("collision", False)),
                    }
                    for run in results_for_all_runs
                ],
            }
        }


    # ============================================================
    # Save results
    # ============================================================

    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(
        args.output_dir,
        "evaluation_results.json"
    )

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {results_path}")
            




if __name__ == "__main__":
    main()