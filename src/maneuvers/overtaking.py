import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Optional

from data.smoothing import smooth
from features.base import OvertakingFeatures
from features.vehicle_dynamics import speed, acceleration, longitudinal_acceleration, lateral_acceleration, \
  longitudinal_velocity, lateral_velocity
from maneuvers.base import OvertakingManeuver
from maneuvers.utils import get_lateral_longitudinal, extract_overtake_windows


"""
This module implements overtaking maneuver detection based on lateral distance thresholds. 
The main idea is to identify the center of the overtake (where the lateral distance is maximal) 
and then search backward and forward in time for points where the lateral distance falls below certain thresholds, 
indicating the start and end of the maneuver. 
The thresholds are generated logarithmically based on the maximum lateral distance at the overtaking point.
"""



def compute_thresholds(lat_z: float, num_thresholds: int=5, min_thresh: float=0.05, global_max: float=1.2) -> List[float]:
  """Generate a set of logarithmic thresholds based on the lateral distance at the overtake center."""
  upper_bound = np.clip(abs(lat_z), min_thresh, global_max)
  if num_thresholds <= 1:
    return [upper_bound]

  thresholds = np.logspace(
    np.log10(min_thresh),
    np.log10(upper_bound),
    num=num_thresholds
  )

  if abs(lat_z) < global_max:
    thresholds = thresholds[:-1]

  return thresholds.tolist()


def detect_overtake_edges_threshold(
    lateral_series: np.ndarray,
    center_idx: int,
    min_frames: int=10,
    max_frames: int=130,
    num_thresholds: int=5,
    max_threshold: float=1.5
) -> Optional[Tuple[int, int]]:
  """Detect the start and end indices of an overtake event based on thresholded lateral distances."""
  n = len(lateral_series)
  abs_lat = np.abs(lateral_series)

  lat_z = abs(lateral_series[center_idx])
  thresholds = compute_thresholds(lat_z, num_thresholds, global_max=max_threshold)

  # ----------------- Define window around t_zero -----------------
  start_search = max(center_idx - max_frames, 0)
  end_search = min(center_idx + max_frames + 1, n)
  lat_window = abs_lat[start_search:end_search]

  # ----------------- Backward search (start) -----------------
  start_idx = None
  backward_limit = center_idx - min_frames
  backward_slice = lat_window[:backward_limit - start_search + 1]
  for th in thresholds:
    candidates = np.where(backward_slice <= th)[0]
    if len(candidates) > 0:
      start_idx = start_search + candidates[-1]  # furthest backward satisfying threshold
      break

  if start_idx is None:
    start_idx = None

  # ----------------- Forward search (end) -----------------
  end_idx = None
  forward_limit = center_idx + min_frames
  forward_slice = lat_window[forward_limit - start_search:]
  for th in thresholds:
    candidates = np.where(forward_slice <= th)[0]
    if len(candidates) > 0:
      end_idx = forward_limit + candidates[0]
      break

  if end_idx is None:
    end_idx = None

  return start_idx, end_idx


def detect_overtaking(
    trajectories: pd.DataFrame,
    interaction: pd.Series,
    num_thresholds: int=5,
    max_lateral_distance: float=3,
    min_lateral_distance_cross: float=0.3,
    min_frames: int=10,
    max_frames: int=125,
    max_threshold: float=1.2
) -> Optional[OvertakingManeuver]:
  """Detect a single overtaking maneuver between two interacting agents."""
  a_idx, b_idx = interaction["track_id"], interaction["other_id"]
  a = trajectories[trajectories["track_id"] == a_idx]
  b = trajectories[trajectories["track_id"] == b_idx]

  # compute lateral and longitudinal distances in follower frame
  ts, a_lateral, a_longitudinal = get_lateral_longitudinal(a, b)
  ts, b_lateral, b_longitudinal = get_lateral_longitudinal(b, a)

  a_long_smooth = smooth(a_longitudinal, 0.5)
  b_long_smooth = smooth(b_longitudinal, 0.5)
  a_lat_smooth = smooth(a_lateral, 0.5)
  b_lat_smooth = smooth(b_lateral, 0.5)

  v_a = speed(a)
  v_b = speed(b)

  acc_a, acc_b = acceleration(a), acceleration(b)


  if ts is None:
    return None

  windows = extract_overtake_windows(ts, a_long_smooth, b_long_smooth)
  candidates = []

  for w in windows:
    start, z, end = w["start"], w["center"], w["end"]

    if a_longitudinal[start] > 0:
      f, l = a_idx, b_idx
      long, lat = a_long_smooth, a_lat_smooth
      v_f, v_l = v_a, v_b
      acc_l, acc_f = acc_a, acc_b
      long_acc, lat_acc = longitudinal_acceleration(a), lateral_acceleration(a)
      v_long, v_lat = longitudinal_velocity(a), lateral_velocity(a)
    else:
      f, l = b_idx, a_idx
      long, lat = b_long_smooth, b_lat_smooth
      v_f, v_l = v_b, v_a
      acc_l, acc_f = acc_b, acc_a
      long_acc, lat_acc = longitudinal_acceleration(b), lateral_acceleration(b)
      v_long, v_lat = longitudinal_velocity(b), lateral_velocity(b)

    decreasing = long[start] > 0 > long[end]
    too_far = abs(lat[z]) < min_lateral_distance_cross or abs(lat[z]) > max_lateral_distance

    if not decreasing or too_far:
      continue

    start_idx, end_idx = detect_overtake_edges_threshold(
      lat, z,
      min_frames=min_frames,
      max_frames=max_frames,
      num_thresholds=num_thresholds,
      max_threshold=max_threshold
    )

    if start_idx is not None:
      distance_check = np.all(np.abs(lat[start_idx:z-1]) < max_lateral_distance)
      length_check = z - start_idx > min_frames
      monotony = np.all(long[start_idx:z-1] > 0)
      distance_travelled_start = np.hypot(long[z] - long[start_idx], lat[z] - lat[start_idx])
      sanity_check = distance_check and length_check and monotony
      if not sanity_check:
        start_idx = None
      else:
        start_idx = max(start_idx, 0)
    else:
      distance_travelled_start = 0

    if end_idx is not None:
      distance_check = np.all(np.abs(lat[z+1:end_idx]) < max_lateral_distance)
      monotony = np.all(long[z+1:end_idx] < 0)
      length_check = end_idx - z > min_frames
      distance_travelled_end = np.hypot(long[end_idx] - long[z], lat[end_idx] - lat[z])
      sanity_check = distance_check and length_check and monotony
      if not sanity_check:
        end_idx = None
      else:
        end_idx = min(end_idx, len(ts) - 1)
    else:
      distance_travelled_end = 0

    if start_idx is None and end_idx is None:
      continue

    t_start = float(ts[start_idx]) if start_idx is not None else None
    t_end = float(ts[end_idx]) if end_idx is not None else None
    t_z = float(ts[z])
    duration = t_end - t_start if t_start is not None and t_end is not None else None
    distance = np.sqrt(long**2 + lat**2)

    f_min, f_max = start_idx or z, end_idx or z

    rel_v = v_f[f_min:f_max+1] - v_l[f_min:f_max+1]
    rel_acc = acc_f[f_min:f_max+1] - acc_l[f_min:f_max+1]
    candidates.append(
      OvertakingManeuver(
        ego_id=int(f), other_id=int(l),
        t_start=t_start,
        t_cross=t_z,
        t_end=t_end,
        duration=duration,
        features=OvertakingFeatures(
          left_side=bool(lat[z] > 0),
          duration_pull_out=float(t_z - t_start) if t_start is not None else None,
          duration_merge_back=float(t_end - t_z) if t_end is not None else None,

          overtake_length=float(distance_travelled_start + distance_travelled_end),

          distance_start=float(distance[f_min]),
          distance_cross=float(distance[z]),
          distance_end=float(distance[f_max]),

          lateral_offset_start=float(lat[f_min]),
          lateral_offset_cross=float(lat[z]),
          lateral_offset_end=float(lat[f_max]),
          lateral_offset_max=float(np.max(np.abs(lat[f_min:f_max+1]))),

          speed_mean=float(np.mean(v_f[f_min:f_max+1])),
          speed_max=float(np.max(v_f[f_min:f_max+1])),
          speed_std=float(np.std(v_f[f_min:f_max+1])),
          speed_gain=float(v_f[f_max] - v_f[f_min]),

          leader_speed_mean=float(np.mean(v_l[f_min:f_max+1])),
          leader_speed_std=float(np.std(v_l[f_min:f_max+1])),
          leader_acc_mean=float(np.mean(acc_l[f_min:f_max+1])),
          leader_acc_std=float(np.std(acc_l[f_min:f_max+1])),

          lat_speed_max=float(np.max(np.abs(v_lat[f_min:f_max + 1]))),
          lat_acc_max=float(np.max(lat_acc[f_min:f_max + 1])),
          lat_acc_min=float(np.min(lat_acc[f_min:f_max + 1])),

          long_acc_max=float(np.max(long_acc[f_min:f_max+1])),
          long_acc_std=float(np.std(long_acc[f_min:f_max+1])),

          rel_acc_max=float(np.max(rel_acc)),
          rel_acc_std=float(np.std(rel_acc)),

          rel_speed_min=float(np.min(rel_v)),
          rel_speed_max=float(np.max(rel_v)),
          rel_speed_std=float(np.std(rel_v)),

          rel_speed_mean=float(np.mean(np.abs(rel_v))),
          rel_speed_start=float(np.abs(rel_v[0])),
          rel_speed_cross=float(np.abs(rel_v[z-f_min])),
          rel_speed_end=float(np.abs(rel_v[-1]))
        )
      )
    )

  if not candidates:
    return None

  # prefer fully valid ones
  for c in candidates:
    if c.t_start and c.t_end:
      return c

  return candidates[0]


def get_overtaking_maneuvers(traj_df: pd.DataFrame, interactions: pd.Series, config: dict) -> List[OvertakingManeuver]:
  """
  Extract all overtaking maneuvers from trajectory data and interaction metadata.
  """
  maneuvers = []
  next_id = 0
  for _, interaction in tqdm(interactions.iterrows(), total=interactions.shape[0]):
    a, b = interaction["track_id"], interaction["other_id"]

    traj_pair = traj_df[traj_df["track_id"].isin([a, b])]
    window = traj_pair[
      (traj_pair["timestamp"] >= interaction["t_start"]-10) &
      (traj_pair["timestamp"] <= interaction["t_end"]+10)
    ]
    result = detect_overtaking(window, interaction, **config)
    if result:
      result.id = next_id
      next_id += 1
      maneuvers.append(result)

  return list(set(maneuvers))
