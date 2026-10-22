from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def ray_cells(start, end):
    """Integer Bresenham cells from start through end, inclusive."""
    x0, y0 = map(int, start)
    x1, y1 = map(int, end)
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx - dy
    cells = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        twice_error = 2 * error
        if twice_error > -dy:
            error -= dy
            x0 += sx
        if twice_error < dx:
            error += dx
            y0 += sy
    return cells


def scan_endpoints(scans, poses, angles, max_range):
    all_points = []
    for scan, pose in zip(scans, poses):
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(scan)):
            continue
        ranges = np.clip(scan, 0.0, max_range)
        theta = pose[2] + angles
        points = np.column_stack((pose[0] + ranges * np.cos(theta),
                                  pose[1] + ranges * np.sin(theta)))
        all_points.append(points)
    return np.concatenate(all_points, axis=0) if all_points else np.empty((0, 2))


def make_occupancy(scans, poses, angles, max_range, resolution, bounds):
    xmin, xmax, ymin, ymax = bounds
    width = int(np.ceil((xmax - xmin) / resolution)) + 1
    height = int(np.ceil((ymax - ymin) / resolution)) + 1
    free_count = np.zeros((height, width), dtype=np.uint16)
    hit_count = np.zeros((height, width), dtype=np.uint16)

    for scan, pose in zip(scans, poses):
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(scan)):
            continue
        origin = (int((pose[0] - xmin) / resolution),
                  int((pose[1] - ymin) / resolution))
        theta = pose[2] + angles
        for distance, angle in zip(np.clip(scan, 0.0, max_range), theta):
            endpoint_x = pose[0] + distance * np.cos(angle)
            endpoint_y = pose[1] + distance * np.sin(angle)
            endpoint = (int((endpoint_x - xmin) / resolution),
                        int((endpoint_y - ymin) / resolution))
            cells = ray_cells(origin, endpoint)
            is_hit = distance < max_range - 0.05
            free_cells = cells[:-1] if is_hit else cells
            for x, y in free_cells:
                if 0 <= x < width and 0 <= y < height:
                    free_count[y, x] += 1
            if is_hit:
                x, y = cells[-1]
                if 0 <= x < width and 0 <= y < height:
                    hit_count[y, x] += 1

    observed = (free_count + hit_count) > 0
    occupancy = np.full((height, width), np.nan, dtype=np.float32)
    occupancy[observed] = hit_count[observed] / (
        free_count[observed] + hit_count[observed]
    )
    return occupancy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="NPZ saved by compare_lidar_imagination.py")
    parser.add_argument("--out", default=None, help="Output image path (default: alongside input)")
    parser.add_argument("--resolution", type=float, default=0.1, help="Grid cell size in meters")
    parser.add_argument("--fov", type=float, default=4.7, help="LiDAR field of view in radians")
    parser.add_argument("--max_range", type=float, default=10.0, help="LiDAR maximum range in meters")
    args = parser.parse_args()
    if args.resolution <= 0 or args.fov <= 0 or args.max_range <= 0:
        parser.error("resolution, fov, and max_range must be positive")

    with np.load(args.input) as data:
        real_scans = data["real_scans"]
        imagined_scans = data["imagined_scans"]
        poses = data["poses"]
        centerline = data["centerline"] if "centerline" in data.files else None

    valid = np.all(np.isfinite(poses), axis=1)
    valid &= np.all(np.isfinite(real_scans), axis=1)
    valid &= np.all(np.isfinite(imagined_scans), axis=1)
    real_scans, imagined_scans, poses = real_scans[valid], imagined_scans[valid], poses[valid]
    # Skip the shared initial observation; map only scans after CEM skills ran.
    if len(poses) > 1:
        real_scans, imagined_scans, poses = real_scans[1:], imagined_scans[1:], poses[1:]
    if len(poses) == 0:
        raise ValueError("Input file contains no valid scan/pose pairs")

    beam_angles = np.linspace(-args.fov / 2.0, args.fov / 2.0, real_scans.shape[1])
    endpoints = np.vstack((scan_endpoints(real_scans, poses, beam_angles, args.max_range),
                           scan_endpoints(imagined_scans, poses, beam_angles, args.max_range),
                           poses[:, :2],
                           centerline if centerline is not None else poses[:, :2]))
    margin = args.max_range * 0.05
    xmin, ymin = endpoints.min(axis=0) - margin
    xmax, ymax = endpoints.max(axis=0) + margin
    bounds = (float(xmin), float(xmax), float(ymin), float(ymax))

    real_map = make_occupancy(real_scans, poses, beam_angles, args.max_range,
                              args.resolution, bounds)
    imagined_map = make_occupancy(imagined_scans, poses, beam_angles, args.max_range,
                                  args.resolution, bounds)
    extent = (bounds[0], bounds[1], bounds[2], bounds[3])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, occupancy, title in zip(axes, (real_map, imagined_map),
                                    ("Real LiDAR map", "Imagined LiDAR map")):
        cmap = plt.get_cmap("gray_r").copy()
        cmap.set_bad("#bcbcbc")
        ax.imshow(occupancy, origin="lower", extent=extent, cmap=cmap,
                  vmin=0.0, vmax=1.0, interpolation="nearest")
        if centerline is not None:
            ax.plot(centerline[:, 0], centerline[:, 1], color="#d62728", lw=0.8,
                    alpha=0.8, label="centerline")
        ax.plot(poses[:, 0], poses[:, 1], color="#1f77b4", lw=1.0,
                label="executed path")
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("y (m)")
    fig.tight_layout()

    output = args.out or os.path.splitext(args.input)[0] + "_maps.png"
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    raw_out = os.path.splitext(output)[0] + ".npz"
    np.savez_compressed(raw_out, real_map=real_map, imagined_map=imagined_map,
                        bounds=np.asarray(bounds), resolution=np.float32(args.resolution))
    print(f"[lidar-map] wrote {output}")
    print(f"[lidar-map] wrote {raw_out}")


if __name__ == "__main__":
    main()
