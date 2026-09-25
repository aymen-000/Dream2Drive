from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def _read_centerline_csv(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        first_line = f.readline()

    if ";" in first_line:
        header = [c.strip().lstrip("#").strip() for c in first_line.split(";")]
        df = pd.read_csv(path, sep=";", comment=None, names=header, skiprows=1)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("#").strip() for c in df.columns]
    return df


@dataclass
class Centerline:

    points: np.ndarray          # [N, 2] (x, y) centerline points, in track order
    track_half_width: Optional[np.ndarray] = None  # [N] mean of left/right widths, if available

    def __post_init__(self):
        seg = np.diff(self.points, axis=0, append=self.points[:1])
        seg_len = np.linalg.norm(seg, axis=1)
        self.cum_dist = np.concatenate([[0.0], np.cumsum(seg_len)])[:-1]  # [N]
        self.total_length = float(np.sum(seg_len))
        self.num_points = self.points.shape[0]

    @classmethod
    def from_csv(cls, path: str, x_col: str = "x_m", y_col: str = "y_m",
                 width_left_col: str = "w_tr_left_m", width_right_col: str = "w_tr_right_m"
                 ) -> "Centerline":
        df = _read_centerline_csv(path)
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(
                f"Centerline file '{path}' is missing '{x_col}'/'{y_col}' columns; "
                f"found columns: {list(df.columns)}. If your map uses different "
                f"column names, update utils/track.py's x_col/y_col arguments."
            )
        points = df[[x_col, y_col]].to_numpy(dtype=np.float32)

        half_width = None
        if width_left_col in df.columns and width_right_col in df.columns:
            half_width = (
                (df[width_left_col].to_numpy(dtype=np.float32)
                 + df[width_right_col].to_numpy(dtype=np.float32)) / 2.0
            )
        return cls(points=points, track_half_width=half_width)

    def subsample_waypoints(self, spacing_m: float) -> np.ndarray:
        """Returns a [M, 2] subset of centerline points spaced roughly
        `spacing_m` apart along the track, used as planning-time
        'checkpoints' (the F1TENTH analogue of the drone track's gates)."""
        if spacing_m <= 0:
            return self.points.copy()
        targets = np.arange(0.0, self.total_length, spacing_m)
        idx = np.searchsorted(self.cum_dist, targets, side="left")
        idx = np.clip(idx, 0, self.num_points - 1)
        return self.points[idx]

    def nearest_index(self, xy: np.ndarray) -> int:
        d2 = np.sum((self.points - xy[None, :]) ** 2, axis=1)
        return int(np.argmin(d2))

    def progress_m(self, xy: np.ndarray) -> float:
        """Approximate arc-length progress (in meters, [0, total_length))
        of the closest centerline point to `xy`."""
        return float(self.cum_dist[self.nearest_index(xy)])

    def lateral_error_m(self, xy: np.ndarray) -> float:
        """Distance from `xy` to the nearest centerline point -- used as a
        cheap off-track proxy when per-point track widths aren't available."""
        idx = self.nearest_index(xy)
        return float(np.linalg.norm(xy - self.points[idx]))


def discover_map(maps_root: str, map_name: str):
    """Resolves the (image, yaml, centerline_csv) triple for a map, following
    the f1tenth_racetracks layout:

        maps_root/<map_name>/<map_name>_map.png
        maps_root/<map_name>/<map_name>_map.yaml
        maps_root/<map_name>/<map_name>_centerline.csv

    Returns (map_yaml_path, map_ext, centerline_csv_path). Raises with a
    helpful message (pointing at f1tenth_racetracks) if the map isn't found.
    """
    map_dir = os.path.join(maps_root, map_name)
    map_yaml = os.path.join(map_dir, f"{map_name}_map.yaml")
    centerline_csv = os.path.join(map_dir, f"{map_name}_centerline.csv")

    if not os.path.exists(map_yaml):
        raise FileNotFoundError(
            f"Could not find '{map_yaml}'. Download F1TENTH tracks (e.g. from "
            f"https://github.com/f1tenth/f1tenth_racetracks) and place them under "
            f"'{maps_root}/<map_name>/' following the '<map_name>_map.yaml' / "
            f"'<map_name>_map.png' / '<map_name>_centerline.csv' naming convention, "
            f"or point `data.maps_root` / `data.map_names` in your config at wherever "
            f"you've put them."
        )
    for ext in (".png", ".pgm"):
        candidate = os.path.join(map_dir, f"{map_name}_map{ext}")
        if os.path.exists(candidate):
            map_ext = ext
            break
    else:
        raise FileNotFoundError(f"No '{map_name}_map.png' or '.pgm' found in {map_dir}")

    if not os.path.exists(centerline_csv):
        raise FileNotFoundError(
            f"Could not find '{centerline_csv}'. This is only used at "
            f"planning/reward time (never for training) -- f1tenth_racetracks "
            f"ships a '<map>_centerline.csv' for every track."
        )
    return map_yaml, map_ext, centerline_csv


