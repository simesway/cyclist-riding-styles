import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
from dataclasses import dataclass

from data.smoothing import smooth
from data.utils import clean_heading
from features.safety_metrics import time_headway
from features.vehicle_dynamics import longitudinal_velocity
from maneuvers.utils import get_lateral_longitudinal, detect_sign_flips


# TODO: integrate FollowingManeuver into module
@dataclass
class FollowingManeuver:
  follower_id: int
  leader_id: int
  t_start: float
  t_end: float
  duration: float

  distance_min: float
  distance_mean: float
  lateral_offset_mean: float

  thw_avg: float
  thw_min: float

  follower_speed_mean: float
  leader_speed_mean: float
  speed_diff_mean: float

  rel_heading_std: float



def get_true_intervals(bool_array):
  """Return list of (start_idx, end_idx) for contiguous True regions."""
  intervals = []
  in_interval = False
  start = 0
  for i, val in enumerate(bool_array):
    if val and not in_interval:
      in_interval = True
      start = i
    elif not val and in_interval:
      in_interval = False
      intervals.append((start, i - 1))
  if in_interval:
    intervals.append((start, len(bool_array) - 1))
  return intervals


def detect_following(
    trajectories: pd.DataFrame,
    interaction: pd.Series,
    min_length: float=1.,
    max_lateral_distance: float = 1.,
    min_long_distance: float = 1.,
    max_long_distance: float = 30.,
    max_time_headway: float = 6.,
    max_rel_heading: float = 35,
) -> List[Tuple]:
  trajectories = trajectories.sort_values(by=["timestamp"])
  a_idx, b_idx = interaction["track_id"], interaction["other_id"]
  a = trajectories[trajectories["track_id"] == a_idx]
  b = trajectories[trajectories["track_id"] == b_idx]

  ts, a_lateral, a_longitudinal = get_lateral_longitudinal(a, b)
  ts, b_lateral, b_longitudinal = get_lateral_longitudinal(b, a)

  a_lat_smooth = smooth(a_lateral, 0.5)
  b_lat_smooth = smooth(b_lateral, 0.5)
  a_long_smooth = smooth(a_longitudinal, 0.5)
  b_long_smooth = smooth(b_longitudinal, 0.5)

  ta = a[a["timestamp"].isin(ts)]
  tb = b[b["timestamp"].isin(ts)]

  ha = clean_heading(ta["rotation_z"].to_numpy())
  hb = clean_heading(tb["rotation_z"].to_numpy())
  v_long_a = longitudinal_velocity(ta).to_numpy()
  v_long_b = longitudinal_velocity(tb).to_numpy()
  v_long_a_smooth = smooth(v_long_a, 0.2)
  v_long_b_smooth = smooth(v_long_b, 0.2)


  thw_a = time_headway(a_long_smooth, v_long_a_smooth)
  thw_b = time_headway(b_long_smooth, v_long_b_smooth)


  h_diff = np.degrees(ha - hb)
  rel_heading = np.abs(h_diff)

  L = len(ts)

  zero_crossings = detect_sign_flips(a_long_smooth)

  intervals = []

  if zero_crossings is None:
    intervals.append((0, L-1))
  else:
    start_idx = 0
    for z in zero_crossings:
      intervals.append((start_idx, z-1))
      start_idx = z+1
    if start_idx < L:
      intervals.append((start_idx, L-1))

  result = []
  for s, e in intervals:
    if ts[e] - ts[s] < min_length:
      continue

    if a_long_smooth[s] > 0:
      l, f = b_idx, a_idx
      long, lat = a_long_smooth[s:e], a_lat_smooth[s:e]
      thw = thw_a[s:e]
    else:
      l, f = a_idx, b_idx
      long, lat = b_long_smooth[s:e], b_lat_smooth[s:e]
      thw = thw_b[s:e]

    lat_offset_ok = np.abs(lat) < max_lateral_distance
    spatial_headway_ok = (min_long_distance < np.abs(long)) & (np.abs(long) < max_long_distance)
    time_headway_ok = np.abs(thw) < max_time_headway
    rel_heading_ok = rel_heading[s:e] < max_rel_heading
    is_following = lat_offset_ok & rel_heading_ok & spatial_headway_ok & time_headway_ok

    segment_intervals = get_true_intervals(is_following)
    absolute_intervals = [
      (s + start, s + end)
      for start, end in segment_intervals
      if ts[s+end] - ts[s+start] >= min_length
    ]
    result.extend([
      (int(f), int(l), float(ts[start]), float(ts[end]))
      for start, end in absolute_intervals
    ])
  return result


def get_following_maneuvers(traj_df: pd.DataFrame, interactions: pd.Series, config: dict) -> List[Tuple]:
  """
  Extract all following maneuvers from trajectory data and interaction metadata.

  Parameters
  ----------
  traj_df : DataFrame
     Full trajectory dataset.
  interactions : DataFrame
     Interaction metadata with (track_id, other_id, t_start, t_end).

  Returns
  -------
  List[Tuple]
     List of overtaking event tuples.
  """
  maneuvers = []
  for _, interaction in tqdm(interactions.iterrows(), total=interactions.shape[0]):
    a, b = interaction["track_id"], interaction["other_id"]

    traj_pair = traj_df[traj_df["track_id"].isin([a, b])]
    window = traj_pair[
      (traj_pair["timestamp"] >= interaction["t_start"]) &
      (traj_pair["timestamp"] <= interaction["t_end"])
    ]
    result = detect_following(window, interaction, **config)
    if result is not None:
      maneuvers.extend(result)

  return maneuvers