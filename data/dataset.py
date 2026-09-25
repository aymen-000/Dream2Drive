from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.normalization import Normalizer
from utils.track import Centerline


@dataclass
class RunArrays:
    name: str
    map_name: str
    states: np.ndarray
    actions: np.ndarray
    raw_states: np.ndarray
    lidar: np.ndarray
    dones: np.ndarray
    collisions: np.ndarray


def discover_runs(root: str, explicit: Optional[List[str]] = None) -> List[str]:
    if explicit:
        return explicit
    files = sorted(glob.glob(os.path.join(root, "*", "run-*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No 'run-*.csv' files found under '{root}'. Run "
            f"scripts/generate_offline_data.py first to collect the offline dataset."
        )
    return [os.path.relpath(f, root) for f in files]


def load_run_dataframe(root: str, run_rel_path: str, dropna: bool, stride: int) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(root, run_rel_path))
    if dropna:
        df = df.dropna().reset_index(drop=True)
    if stride > 1:
        df = df.iloc[::stride].reset_index(drop=True)
    return df


def lidar_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("lidar_")]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


class OfflineSkillDataset(Dataset):
    def __init__(self, runs: List[RunArrays], window_len: int, drop_collision_windows: bool = False):
        """
        Args:
            runs: per-episode arrays (see build_run_arrays).
            window_len: number of states per window (window_len - 1 actions).
            drop_collision_windows: if True, skip any window whose terminal
                transition (the action landing on sT) ended in a collision.
                Useful if you want a world model trained only on "clean"
                dynamics, with collision transitions handled separately.
        """
        assert window_len >= 2
        self.window_len = window_len
        self.runs = runs

        self._index: List[Tuple[int, int]] = []
        for ri, run in enumerate(runs):
            n = run.states.shape[0]
            last_start = n - window_len
            if last_start < 0:
                continue
            for t in range(last_start + 1):
                terminal_idx = t + window_len - 2  # row whose action produced sT
                if drop_collision_windows and bool(run.collisions[terminal_idx]):
                    continue
                self._index.append((ri, t))

        if not self._index:
            raise ValueError(
                "No run produced a single valid sub-trajectory window -- "
                "reduce data.subtraj_len/subsample_stride, disable "
                "drop_collision_windows, or collect longer/more runs with "
                "scripts/generate_offline_data.py."
            )

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ri, t = self._index[idx]
        run = self.runs[ri]
        T = self.window_len - 1
        states_window = run.states[t: t + self.window_len]
        actions_window = run.actions[t: t + T]
        lidar_window = run.lidar[t: t + T]
        terminal_idx = t + T - 1
        return {
            "s0": torch.as_tensor(states_window[0], dtype=torch.float32),
            "states_seq": torch.as_tensor(states_window[:-1], dtype=torch.float32),
            "actions_seq": torch.as_tensor(actions_window, dtype=torch.float32),
            "lidar_seq": torch.as_tensor(lidar_window, dtype=torch.float32),
            "sT": torch.as_tensor(states_window[-1], dtype=torch.float32),
            "done": torch.tensor(bool(run.dones[terminal_idx]), dtype=torch.bool),
            "collision": torch.tensor(bool(run.collisions[terminal_idx]), dtype=torch.bool),
        }


def build_run_arrays(
    root: str, run_names: List[str], state_features: List[str], action_features: List[str],
    dropna: bool, stride: int,
    state_normalizer: Optional[Normalizer] = None, action_normalizer: Optional[Normalizer] = None,
) -> Tuple[List[RunArrays], Normalizer, Normalizer]:

    raw_states_per_run, raw_actions_per_run, lidar_per_run, map_names, names = [], [], [], [], []
    dones_per_run, collisions_per_run = [], []
    for name in run_names:
        df = load_run_dataframe(root, name, dropna, stride)
        missing_state = [c for c in state_features if c not in df.columns]
        missing_action = [c for c in action_features if c not in df.columns]
        if missing_state or missing_action:
            raise ValueError(
                f"Run '{name}' is missing expected columns. "
                f"missing state cols={missing_state}, missing action cols={missing_action}"
            )
        lidar_cols = lidar_columns(df)
        if not lidar_cols:
            raise ValueError(
                f"Run '{name}' has no 'lidar_*' columns. "
                f"Re-run scripts/generate_offline_data.py with the updated collector."
            )
        raw_states_per_run.append(df[state_features].to_numpy(dtype=np.float32))
        raw_actions_per_run.append(df[action_features].to_numpy(dtype=np.float32))
        lidar_per_run.append(df[lidar_cols].to_numpy(dtype=np.float32))
        map_names.append(df["map_name"].iloc[0] if "map_name" in df.columns else "unknown")
        names.append(name)

        n = len(df)
        if "done" in df.columns:
            dones = df["done"].to_numpy(dtype=bool)
        else:
            dones = np.zeros(n, dtype=bool)
            if n > 0:
                dones[-1] = True
        if "collision" in df.columns:
            collisions = df["collision"].to_numpy(dtype=bool)
        else:
            collisions = np.zeros(n, dtype=bool)
        dones_per_run.append(dones)
        collisions_per_run.append(collisions)

    all_raw_states = np.concatenate(raw_states_per_run, axis=0)
    all_raw_actions = np.concatenate(raw_actions_per_run, axis=0)

    if state_normalizer is None:
        state_normalizer = Normalizer.fit(all_raw_states, feature_names=state_features)
    if action_normalizer is None:
        action_normalizer = Normalizer.fit(all_raw_actions, feature_names=action_features)

    runs = []
    for name, map_name, raw_s, raw_a, lidar, dones, collisions in zip(
        names, map_names, raw_states_per_run, raw_actions_per_run, lidar_per_run,
        dones_per_run, collisions_per_run,
    ):
        runs.append(RunArrays(
            name=name, map_name=map_name,
            states=state_normalizer.transform(raw_s).astype(np.float32),
            actions=action_normalizer.transform(raw_a).astype(np.float32),
            raw_states=raw_s,
            lidar=lidar,
            dones=dones,
            collisions=collisions,
        ))
    return runs, state_normalizer, action_normalizer


def train_test_run_split(run_names: List[str], train_frac: float, seed: int = 0):
    rng = np.random.RandomState(seed)
    names = list(run_names)
    rng.shuffle(names)
    n_train = max(1, int(round(train_frac * len(names))))
    return names[:n_train], names[n_train:]


def load_centerline_for_run(maps_root: str, map_name:str) -> Centerline:
    from utils.track import discover_map
    map_yaml,map_ext, centerline_csv = discover_map(maps_root, map_name)
    return map_yaml , map_ext ,Centerline.from_csv(centerline_csv)