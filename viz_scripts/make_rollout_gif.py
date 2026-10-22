from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

TRACK_COLOR = "#a0a0a0"
RACELINE_COLOR = "#1f77b4"
IMAGINED_COLOR = "#d62728"
POP_COLOR = "#7f7f7f"


def load_steps(save_dir):
    files = sorted(glob.glob(os.path.join(save_dir, "plan_step_*.npz")))
    if not files:
        raise FileNotFoundError(f"No plan_step_*.npz files found in {save_dir}")
    steps = []
    for f in files:
        d = np.load(f)
        steps.append({k: d[k] for k in d.files})
    return steps


def racing_line_indices(init_positions, trajectory):
    """For each plan step, the index in `trajectory` closest to the car's position then
    (never going backwards), so the race line can be sliced trajectory[:idx+1]."""
    idxs, prev = [], 0
    for p in init_positions:
        d = np.linalg.norm(trajectory[prev:, :2] - p[None, :], axis=1)
        prev = prev + int(np.argmin(d))
        idxs.append(prev)
    return idxs


def compute_limits(track, steps, trajectory, pad=0.06):
    pts = [np.stack([s["init_position"] for s in steps])]
    pts += [s["best_predicted_positions"] for s in steps]
    if track is not None:
        pts.append(track[:, :2])
    if trajectory is not None:
        pts.append(trajectory[:, :2])
    allp = np.concatenate(pts, axis=0)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return lo - pad * span, hi + pad * span


def render_frame(t, steps, track, trajectory, traj_idx, lims, show_population, init_positions):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    lo, hi = lims
    step = steps[t]
    cur = step["init_position"]

    if track is not None:
        closed = np.vstack([track[:, :2], track[:1, :2]])
        ax.plot(closed[:, 0], closed[:, 1], color=TRACK_COLOR, lw=1.0, ls="--", zorder=1, label="track")

    # race line = what the car has actually driven so far
    if trajectory is not None:
        driven = trajectory[: traj_idx[t] + 1, :2]
    else:
        driven = init_positions[: t + 1]
    ax.plot(driven[:, 0], driven[:, 1], color=RACELINE_COLOR, lw=2.2, zorder=3, label="race line")

    if show_population and "population_positions" in step:
        pop = step["population_positions"].reshape(-1, 2)
        ax.scatter(pop[:, 0], pop[:, 1], s=5, color=POP_COLOR, alpha=0.18, linewidths=0, zorder=2)

    # imagined points of the current plan
    imag = step["best_predicted_positions"]
    ax.scatter(imag[:, 0], imag[:, 1], color=IMAGINED_COLOR, s=np.linspace(20, 55, len(imag)),
               edgecolors="black", linewidths=0.4, zorder=4, label="imagined points")

    # the car right now
    ax.scatter(*cur, color="black", marker="s", s=45, zorder=5, label="car")

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"step {t}/{len(steps) - 1}   R={float(step['best_reward']):.2f}")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(frame)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", required=True, help="dir with track.npz and plan_step_*.npz")
    p.add_argument("--out", default=None, help="output .gif (default: save_dir/rollouts.gif)")
    p.add_argument("--traj", default=None, help="optional *_plan.npz from run_plan.py (key 'trajectory')")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--loop", type=int, default=0, help="0 = loop forever")
    p.add_argument("--stride", type=int, default=1, help="use every Nth planning step")
    p.add_argument("--show_population", action="store_true", help="also draw the whole CEM population as faint dots")
    args = p.parse_args()

    steps = load_steps(args.save_dir)[:: max(1, args.stride)]
    track_path = os.path.join(args.save_dir, "track.npz")
    track = np.load(track_path)["waypoints"] if os.path.exists(track_path) else None

    init_positions = np.stack([s["init_position"] for s in steps])
    trajectory, traj_idx = None, None
    if args.traj:
        trajectory = np.load(args.traj)["trajectory"][:, :2]
        traj_idx = racing_line_indices(init_positions, trajectory)

    lims = compute_limits(track, steps, trajectory)

    frames = [render_frame(t, steps, track, trajectory, traj_idx, lims, args.show_population, init_positions)
              for t in range(len(steps))]

    out_path = args.out or os.path.join(args.save_dir, "rollouts.gif")
    pal = [f.convert("P", palette=Image.ADAPTIVE) for f in frames]
    pal[0].save(out_path, save_all=True, append_images=pal[1:],
                duration=int(1000 / args.fps), loop=args.loop, optimize=False)
    print(f"wrote {out_path} ({len(frames)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
