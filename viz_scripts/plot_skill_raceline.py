from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CEM LiDAR rollout NPZ")
    parser.add_argument("--out", default=None, help="Output PNG (default: beside input)")
    args = parser.parse_args()

    with np.load(args.input) as data:
        trajectory = data["trajectory"][:, :2]
        skills = data["executed_zs"]
        centerline = data["centerline"] if "centerline" in data.files else None

    num_skills = len(skills)
    if num_skills == 0:
        raise ValueError("The rollout contains no executed skills")
    if len(trajectory) != num_skills + 1:
        raise ValueError(
            f"Expected one more raceline pose than skills, got "
            f"{len(trajectory)} poses and {num_skills} skills"
        )

    segments = np.stack((trajectory[:-1], trajectory[1:]), axis=1)
    cmap = plt.get_cmap("turbo", num_skills)
    norm = mcolors.BoundaryNorm(np.arange(0.5, num_skills + 1.5), cmap.N)
    skill_ids = np.arange(1, num_skills + 1)
    collection = LineCollection(segments, cmap=cmap, norm=norm,
                                linewidths=2.5, capstyle="round")
    collection.set_array(skill_ids)

    fig, ax = plt.subplots(figsize=(8, 7))
    if centerline is not None:
        closed_centerline = np.vstack((centerline, centerline[:1]))
        ax.plot(closed_centerline[:, 0], closed_centerline[:, 1],
                color="#a0a0a0", lw=1.0, ls="--", label="Track centerline", zorder=1)
    ax.add_collection(collection)
    ax.scatter(*trajectory[0], marker="o", s=45, color="#1b9e77",
               edgecolor="white", linewidth=0.8, label="Start", zorder=4)
    ax.scatter(*trajectory[-1], marker="X", s=55, color="#d73027",
               edgecolor="white", linewidth=0.7, label="Finish", zorder=4)
    ax.autoscale()
    ax.margins(0.04)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Raceline colored by executed skill")
    ax.legend(loc="best")

    colorbar = fig.colorbar(collection, ax=ax, pad=0.025, fraction=0.045)
    colorbar.set_label("Executed skill index")
    if num_skills <= 20:
        ticks = skill_ids
    else:
        stride = int(np.ceil(num_skills / 15))
        ticks = np.unique(np.r_[skill_ids[::stride], skill_ids[-1]])
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([str(int(i)) for i in ticks])
    fig.tight_layout()

    output = args.out or os.path.splitext(args.input)[0] + "_skills.png"
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)

    csv_path = os.path.splitext(output)[0] + ".csv"
    headers = (["skill_index"] + [f"z_{i}" for i in range(skills.shape[1])]
               + ["start_x", "start_y", "end_x", "end_y"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, (z, segment) in enumerate(zip(skills, segments), start=1):
            writer.writerow([i, *z.tolist(), *segment[0].tolist(), *segment[1].tolist()])

    print(f"[skill-raceline] wrote {output}")
    print(f"[skill-raceline] wrote {csv_path}")


if __name__ == "__main__":
    main()
