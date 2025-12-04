import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Optional
from maneuvers.utils import get_lateral_longitudinal, extract_overtake_windows



def compute_thresholds(lat_z: float, num_thresholds: int=5, min_thresh: float=0.05, global_max: float=1.5) -> List[float]:
  """Generate a set of logarithmic thresholds based on the lateral distance at the overtake center."""
  upper_bound = min(abs(lat_z) + 1e-6, global_max)
  thresholds = np.logspace(
    np.log10(min_thresh),
    np.log10(upper_bound),
    num=num_thresholds
  ).tolist()

  if abs(lat_z) >= global_max:
    return thresholds

  return thresholds[:-1]


def detect_overtake_edges_threshold(
    lateral_series: np.ndarray,
    center_idx: int,
    min_frames: int=10,
    max_frames: int=130
) -> Optional[Tuple[int, int]]:
  """Detect the start and end indices of an overtake event based on thresholded lateral distances."""
  n = len(lateral_series)
  abs_lat = np.abs(lateral_series)

  lat_z = abs(lateral_series[center_idx])
  thresholds = compute_thresholds(lat_z)

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
    min_lateral_distance: float=0.3,
    max_lateral_distance: float=3
) -> Optional[Tuple[int, int, Optional[float], float, Optional[float]]]:
  """Detect a single overtaking maneuver between two interacting agents."""
  a_idx, b_idx = interaction["track_id"], interaction["other_id"]
  a = trajectories[trajectories["track_id"] == a_idx]
  b = trajectories[trajectories["track_id"] == b_idx]

  # compute lateral and longitudinal distances in follower frame
  ts, a_lateral, a_longitudinal = get_lateral_longitudinal(a, b)
  ts, b_lateral, b_longitudinal = get_lateral_longitudinal(b, a)

  if ts is None:
    return None

  windows = extract_overtake_windows(ts, a_longitudinal, b_longitudinal)
  candidates = []

  for w in windows:
    start, z, end = w["start"], w["center"], w["end"]

    # Determine follower and leader based on longitudinal sign
    if a_longitudinal[start] > 0:
      f, l = a_idx, b_idx
      long, lat = a_longitudinal, a_lateral
    else:
      f, l = b_idx, a_idx
      long, lat = b_longitudinal, b_lateral

    decreasing = long[start] > 0 > long[end]
    too_far = abs(lat[z]) < min_lateral_distance or abs(lat[z]) > max_lateral_distance

    if not decreasing or too_far:
      continue

    lat_smooth = gaussian_filter1d(lat, sigma=3)
    start_idx, end_idx = detect_overtake_edges_threshold(lat_smooth, z)

    min_frames = 5
    if start_idx is not None:
      distance_check = abs(lat[start_idx]) < max_lateral_distance or abs(lat[start_idx]) < min_lateral_distance
      length_check = z - start_idx > min_frames
      sanity_check = distance_check and length_check
      if not sanity_check:
        start_idx = None
      else:
        start_idx = max(start_idx, 0)

    if end_idx is not None:
      distance_check =  abs(lat[end_idx]) < max_lateral_distance or abs(lat[end_idx]) < min_lateral_distance
      length_check = end_idx - z > min_frames
      sanity_check = distance_check and length_check
      if not sanity_check:
        end_idx = None
      else:
        end_idx = min(end_idx, len(ts) - 1)

    if start_idx is None and end_idx is None:
      continue

    t_start = float(ts[start_idx]) if start_idx is not None else None
    t_end = float(ts[end_idx]) if end_idx is not None else None
    t_z = float(ts[z])

    candidates.append((int(l), int(f), t_start, t_z, t_end))

  if not candidates:
    return None

  # prefer fully valid ones
  for c in candidates:
    if c[2] and c[4]:
      return c

  return candidates[0]


def get_overtaking_maneuvers(traj_df: pd.DataFrame, interactions: pd.Series) -> List[Tuple]:
  """
  Extract all overtaking maneuvers from trajectory data and interaction metadata.

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
      (traj_pair["timestamp"] >= interaction["t_start"]-10) &
      (traj_pair["timestamp"] <= interaction["t_end"]+10)
    ]
    result = detect_overtaking(window, interaction)
    if result is not None:
      maneuvers.append(result)

  return maneuvers
