from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "savefig.bbox": "tight",
})

TRACK_COLOR = "#9a9a9a"
BEST_COLOR = "#d62728"
ELITE_COLOR = "#1f77b4"
START_COLOR = "black"
POP_CMAP = "viridis"


def load_track(save_dir: str) -> np.ndarray | None:
    path = os.path.join(save_dir, "track.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)["waypoints"]


def load_step(save_dir: str, step_idx: int) -> dict:
    path = os.path.join(save_dir, f"plan_step_{step_idx:05d}.npz")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def list_steps(save_dir: str) -> list[int]:
    files = sorted(glob.glob(os.path.join(save_dir, "plan_step_*.npz")))
    return [int(os.path.basename(f)[len("plan_step_"):-len(".npz")]) for f in files]


def plot_step(track: np.ndarray | None, step: dict, step_idx: int, out_path: str,
              show_population: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    if track is not None:
        closed = np.vstack([track, track[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=TRACK_COLOR, lw=1.0,
                ls="--", zorder=1, label="track")

    init_pos = step["init_position"]

    if show_population and "population_positions" in step:
        pop = step["population_positions"]          # (K, L, 2)
        pop_rewards = step["population_rewards"]     # (K,)
        elite_idx = set(step["elite_idx"].tolist())
        L = pop.shape[1]

        # non-elite candidates: every imagined terminal state as a faded dot,
        # colored by that candidate's reward (no line connecting them --
        # the model only predicts terminal states, so a line would imply a
        # rollout shape it never actually computed)
        non_elite = [k for k in range(pop.shape[0]) if k not in elite_idx]
        if non_elite:
            pts = pop[non_elite].reshape(-1, 2)                          # (num_non_elite * L, 2)
            colors = np.repeat(pop_rewards[non_elite], L)
            sca = ax.scatter(pts[:, 0], pts[:, 1], c=colors, cmap=POP_CMAP,
                              s=8, alpha=0.35, linewidths=0, zorder=2)
            cbar = fig.colorbar(sca, ax=ax, shrink=0.75, pad=0.02)
            cbar.set_label("imagined reward", fontsize=9)

        # elite candidates: bigger dots, distinct color
        if elite_idx:
            pts = pop[sorted(elite_idx)].reshape(-1, 2)
            ax.scatter(pts[:, 0], pts[:, 1], color=ELITE_COLOR, s=16,
                       alpha=0.7, linewidths=0, zorder=3, label="elite")

    # best trajectory: bold dots, size growing with step order so you can read direction
    best_pts = step["best_predicted_positions"]                          # (L, 2)
    sizes = np.linspace(20, 55, len(best_pts))
    ax.scatter(best_pts[:, 0], best_pts[:, 1], color=BEST_COLOR, s=sizes,
               edgecolors="black", linewidths=0.4, zorder=4,
               label=f"best (R={float(step['best_reward']):.2f})")
    ax.scatter(*init_pos, color=START_COLOR, s=25, zorder=5, marker="s",
               label="current state")

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Imagined CEM rollouts -- step {step_idx}")
    ax.legend(loc="best", fontsize=8)

    fig.savefig(out_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", required=True, help="dir passed as cfg.save_dir / mpc_race(save_dir=...)")
    p.add_argument("--step", default="0", help="a step index, or 'all' to render every saved step")
    p.add_argument("--out_dir", default=None, help="where to write PNGs (default: save_dir/figs)")
    p.add_argument("--no_population", action="store_true", help="only draw the best trajectory, no CEM candidates")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.save_dir, "figs")
    os.makedirs(out_dir, exist_ok=True)

    track = load_track(args.save_dir)

    if args.step == "all":
        steps = list_steps(args.save_dir)
        if not steps:
            raise FileNotFoundError(f"No plan_step_*.npz files found in {args.save_dir}")
        for s in steps:
            data = load_step(args.save_dir, s)
            out_path = os.path.join(out_dir, f"plan_step_{s:05d}.png")
            plot_step(track, data, s, out_path, show_population=not args.no_population)
            print(f"wrote {out_path}")
    else:
        s = int(args.step)
        data = load_step(args.save_dir, s)
        out_path = os.path.join(out_dir, f"plan_step_{s:05d}.png")
        plot_step(track, data, s, out_path, show_population=not args.no_population)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
