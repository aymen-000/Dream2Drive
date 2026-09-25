""" 
From https://github.com/BDEvan5/TrajectoryAidedLearning/blob/master/TrajectoryAidedLearning/Utils/RewardSignals.py wtih some modifications 
"""
from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import math, cmath




def get_distance(x1=(0, 0), x2=(0, 0)):
    d = [0.0, 0.0]
    for i in range(2):
        d[i] = x1[i] - x2[i]
    return np.linalg.norm(d)


def find_closest_pt(pt, wpts):
    """Returns the two closest points in order along wpts"""
    dists = [get_distance(pt, wpt) for wpt in wpts]
    min_i = np.argmin(dists)
    d_i = dists[min_i]
    if min_i == len(dists) - 1:
        min_i -= 1
    if dists[max(min_i - 1, 0)] > dists[min_i + 1]:
        p_i = wpts[min_i]
        p_ii = wpts[min_i + 1]
        d_i = dists[min_i]
        d_ii = dists[min_i + 1]
    else:
        p_i = wpts[min_i - 1]
        p_ii = wpts[min_i]
        d_i = dists[min_i - 1]
        d_ii = dists[min_i]
    return p_i, p_ii, d_i, d_ii


def get_tiangle_h(a, b, c):
    s = (a + b + c) / 2
    A = np.sqrt(s * (s - a) * (s - b) * (s - c))
    h = 2 * A / c
    return h


def get_gradient(x1=(0, 0), x2=(0, 0)):
    t = (x1[1] - x2[1])
    b = (x1[0] - x2[0])
    if b != 0:
        return t / b
    return 1000000  # near infinite gradient


def get_bearing(x1=(0, 0), x2=(0, 0)):
    grad = get_gradient(x1, x2)
    dx = x2[0] - x1[0]
    th_start_end = np.arctan(grad)
    if dx == 0:
        if x2[1] - x1[1] > 0:
            th_start_end = 0
        else:
            th_start_end = np.pi
    elif th_start_end > 0:
        if dx > 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = -np.pi / 2 - th_start_end
    else:
        if dx > 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = -np.pi / 2 - th_start_end
    return th_start_end


def robust_angle_difference_rad(x, y):
    """r = x - y, wrapped to [-pi, pi]"""
    return np.arctan2(np.sin(x - y), np.cos(x - y))

def get_cross_track_heading(point, wpts):
    p_i, p_ii, d_i, d_ii = find_closest_pt(point, wpts)
    d_ii_i = get_distance(p_i, p_ii)
    if d_ii_i < 1e-8:
        h = 0.0
    else:
        try:
            h = get_tiangle_h(d_i, d_ii, d_ii_i)
        except (ValueError, FloatingPointError):
            h = min(d_i, d_ii)
    heading = get_bearing(p_i, p_ii)
    return heading, h


def calculate_progress(point, wpts, cum_dist):
    """Closest-point arclength progress"""
    dists = [get_distance(point, wpt) for wpt in wpts]
    min_i = int(np.argmin(dists))
    if min_i == len(wpts) - 1:
        min_i -= 1
    p_i, p_ii = wpts[min_i], wpts[min_i + 1]
    seg_vec = np.array(p_ii) - np.array(p_i)
    seg_len = np.linalg.norm(seg_vec) + 1e-8
    t = np.clip(np.dot(np.array(point) - np.array(p_i), seg_vec) / (seg_len ** 2), 0.0, 1.0)
    return cum_dist[min_i] + t * seg_len


class CenterlineProgressReward:
    def __init__(self, waypoints_xy: np.ndarray,
                 waypoint_bonus: float = 0.0, progress_weight: float = 1.0,
                 track_half_width: float = None, off_track_penalty: float = 0.0,
                 reward_mode: str = "progress", max_v: float = 1.0,
                 backward_penalty: float = 2.0,
                 max_progress_per_step: float = 12.0,
                 collision_penalty: float = 10.0):
        self.waypoints = waypoints_xy
        self.num_waypoints = waypoints_xy.shape[0]
        self.waypoint_bonus = waypoint_bonus
        self.progress_weight = progress_weight
        self.track_half_width = track_half_width
        self.off_track_penalty = off_track_penalty
        self.reward_mode = reward_mode
        self.max_v = max_v
        self.backward_penalty = backward_penalty
        self.max_progress_per_step = max_progress_per_step
        self.collision_penalty = collision_penalty

        if self.num_waypoints < 3:
            raise ValueError("CenterlineProgressReward needs at least three waypoints")

        # Treat the track as closed.  The final waypoint normally does not
        # duplicate the first, so the closing segment must be included.
        wp = self.waypoints
        self.segment_start = wp
        self.segment_vec = np.roll(wp, -1, axis=0) - wp
        self.segment_len = np.linalg.norm(self.segment_vec, axis=1)
        if np.any(self.segment_len < 1e-8):
            raise ValueError("Centerline contains duplicate adjacent waypoints")
        self.segment_len_sq = self.segment_len ** 2
        self.cum_dist = np.concatenate([[0.0], np.cumsum(self.segment_len)])[:-1]
        self.total_s = float(np.sum(self.segment_len))

        # TALearningReward constants, taken directly from your snippet
        self.beta_c = 0.4
        self.beta_steer_weight = 0.4
        self.beta_velocity_weight = 0.4
        self.max_steer_diff = 0.8
        self.max_velocity_diff = 2.0

        # CrossTrackHeadReward constants, taken directly from your snippet
        self.r_velocity = 1
        self.r_distance = 1

    def _project_positions(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project ``(..., 2)`` positions onto the closed centerline.

        This is deliberately vectorized: CEM scores thousands of trajectories
        every iteration, and the former per-point Python nearest-waypoint loop
        dominated planning time on CPU.
        """
        points = np.asarray(positions, dtype=np.float32)
        offset = points[..., None, :] - self.segment_start
        t = np.sum(offset * self.segment_vec, axis=-1) / self.segment_len_sq
        t = np.clip(t, 0.0, 1.0)
        projected = self.segment_start + t[..., None] * self.segment_vec
        d2 = np.sum((points[..., None, :] - projected) ** 2, axis=-1)
        segment_idx = np.argmin(d2, axis=-1)
        chosen_t = np.take_along_axis(t, segment_idx[..., None], axis=-1)[..., 0]
        s = self.cum_dist[segment_idx] + chosen_t * self.segment_len[segment_idx]
        cross_track_error = np.sqrt(np.take_along_axis(d2, segment_idx[..., None], axis=-1)[..., 0])
        return s, cross_track_error

    def __call__(self, positions: np.ndarray,
                  headings: Optional[np.ndarray] = None,
                  velocities: Optional[np.ndarray] = None,
                  agent_actions: Optional[np.ndarray] = None,
                  classic_actions: Optional[np.ndarray] = None,
                  collisions: Optional[np.ndarray] = None,
                  lap_done: Optional[np.ndarray] = None,
                  init_position: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        positions:       (K, L, 2)
        headings:        (K, L)      required for "cth"
        velocities:      (K, L)      required for "cth"
        agent_actions:   (K, L, 2)   required for "tal"  [steer, speed]
        classic_actions: (K, L, 2)   required for "tal"  [steer, speed] from planner
        collisions:      (K, L) bool optional, terminal -1 (all modes)
        lap_done:        (K, L) bool optional, terminal +1 (all modes)
        init_position:   (K, 2) optional, position just before l=0, needed
                          for "progress" mode's first-step delta
        """
        K, L, _ = positions.shape
        rewards = np.zeros(K, dtype=np.float32)
        wpts_list = self.waypoints.tolist()

        if self.reward_mode == "progress":
            s, cross_track_error = self._project_positions(positions)
            if init_position is None:
                prev_s = s[:, :1]
            else:
                prev_s, _ = self._project_positions(init_position)
                prev_s = prev_s[:, None]
            delta_s = np.diff(np.concatenate([prev_s, s], axis=1), axis=1)
            delta_s = np.where(delta_s < -0.5 * self.total_s, delta_s + self.total_s, delta_s)
            delta_s = np.where(delta_s > 0.5 * self.total_s, delta_s - self.total_s, delta_s)
            forward = np.clip(delta_s, 0.0, self.max_progress_per_step)
            backward = np.clip(-delta_s, 0.0, self.max_progress_per_step)
            valid = np.ones((K, L), dtype=bool)
            if collisions is not None:
                collisions = np.asarray(collisions, dtype=bool)
                if collisions.shape != (K, L):
                    raise ValueError(f"collisions must have shape {(K, L)}, got {collisions.shape}")
                valid = np.cumsum(collisions, axis=1) == 0

            rewards = self.progress_weight * (forward * valid).sum(axis=1)
            rewards -= self.backward_penalty * (backward * valid).sum(axis=1)
            if self.track_half_width is not None and self.off_track_penalty:
                excess = np.clip(cross_track_error - self.track_half_width, 0.0, None)
                rewards -= self.off_track_penalty * (np.square(excess) * valid).sum(axis=1)
            if collisions is not None:
                rewards -= self.collision_penalty * collisions.any(axis=1)
            return rewards.astype(np.float32)

        for k in range(K):
            r = 0.0
            prev_pos = init_position[k] if init_position is not None else positions[k, 0]

            for l in range(L):
                pos = positions[k, l]

                if collisions is not None and collisions[k, l]:
                    r -= self.collision_penalty
                    break
                if lap_done is not None and lap_done[k, l]:
                    r += 1
                    break

                if self.reward_mode == "cth":
                    if headings is None or velocities is None:
                        raise ValueError("headings and velocities are required for reward_mode='cth'")
                    theta = float(headings[k, l])
                    v = float(velocities[k, l])
                    track_heading, dc = get_cross_track_heading(pos, wpts_list)
                    d_heading = abs(robust_angle_difference_rad(track_heading, theta))
                    r_heading = np.cos(d_heading) * self.r_velocity
                    r_heading *= (v / self.max_v)
                    r_dist = dc * self.r_distance
                    step_r = r_heading - r_dist
                    step_r = max(step_r, 0)  # from source

                elif self.reward_mode == "tal":
                    if agent_actions is None or classic_actions is None:
                        raise ValueError("agent_actions and classic_actions required for reward_mode='tal'")
                    u_agent = agent_actions[k, l]
                    u_classic = classic_actions[k, l]
                    steer_r = (abs(u_classic[0] - u_agent[0]) / self.max_steer_diff) * self.beta_steer_weight
                    throttle_r = (abs(u_classic[1] - u_agent[1]) / self.max_velocity_diff) * self.beta_velocity_weight
                    step_r = self.beta_c - steer_r - throttle_r
                    step_r = max(step_r, 0)  
                    step_r *= 0.5  

                else:
                    raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

                r += self.progress_weight * step_r
                prev_pos = pos

            rewards[k] = r


        return rewards
