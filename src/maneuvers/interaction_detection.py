import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict

from shapely.strtree import STRtree
from shapely.geometry import LineString

from data.utils import get_time_window
from data.transform import traj_list_to_linestrings

INTERACTION_FEATURES = (
  "track_id",   # id of traj being analyzed
  "other_id",   # id of interaction traj
  "t_start",    # start of interaction (first moment within distance threshold)
  "t_end",      # end of interaction (last moment within distance threshold)
  "duration",   # total time
  "dist_min",   # minimum euclidean distance between trajs
  "dist_mean",  # mean euclidean distance between trajs
  "rel_speed",  # average relative speed between the two trajs
  "rel_heading",# mean absolute difference in heading
  "is_crossing" # boolean whether the other traj is crossing from one to the other side
)

def simplify_and_segment(traj: LineString, simplify_tol=0.5) -> List[LineString]:
  """Method to simplify and segment a Trajectory. (Zheng 2015)"""
  # 1. simplify using Douglas-Peucker Algorithm
  simplified = traj.simplify(simplify_tol, preserve_topology=False)
  # 2. split into segments
  pts = list(simplified.coords)
  segments = [LineString([pts[i], pts[i+1]]) for i in range(len(pts)-1)]
  return segments

def build_str_tree(traj_dict: Dict[int, LineString]) -> Tuple[STRtree, List[LineString], Dict[LineString, int]]:
  """Build the STR-Tree for the given Trajectories."""
  segments = []
  segment_lookup = {}
  for traj_id, traj in traj_dict.items():
    segs = simplify_and_segment(traj, .1)
    segments.extend(segs)
    for seg in segs:
      segment_lookup[seg] = traj_id

  tree = STRtree(segments)
  return tree, segments, segment_lookup

def filter_interactions(
  df,
  candidates: Dict[int, List[int]],
  distance: float,
  min_duration: float=0.0
) -> List[Tuple[int, int, float, float, float, float, float, int]]:
  """Verify the interactions are within the given distance, and compute interaction features."""
  out = []

  df = df.sort_values(by=["timestamp"])

  ts = df["timestamp"].to_numpy()
  ids = df["track_id"].to_numpy()

  x = df["translation_x"].to_numpy()
  y = df["translation_y"].to_numpy()

  vx = df["velocity_x"].to_numpy()
  vy = df["velocity_y"].to_numpy()

  hd = df["rotation_z"].to_numpy() # heading [-pi, pi]

  dist2 = distance * distance  # avoid sqrt

  for t1, cand_list in candidates.items():
    mask1 = ids == t1
    ts1 = ts[mask1]
    x1 = x[mask1]
    y1 = y[mask1]
    vx1 = vx[mask1]
    vy1 = vy[mask1]
    h1 = hd[mask1]

    for t2 in cand_list:
      if t1 == t2:
        continue

      mask2 = ids == t2
      ts2 = ts[mask2]
      x2 = x[mask2]
      y2 = y[mask2]
      vx2 = vx[mask2]
      vy2 = vy[mask2]
      h2 = hd[mask2]

      # timestamp alignment
      common, i1, i2 = np.intersect1d(ts1, ts2, return_indices=True)
      if len(common) == 0:
        continue

      dx = x1[i1] - x2[i2]
      dy = y1[i1] - y2[i2]
      dist_sq = dx*dx + dy*dy
      close_mask = dist_sq <= dist2

      if not np.any(close_mask):
        continue

      # indices inside the window
      idx = np.where(close_mask)[0]
      # split where gaps are more than 1
      splits = np.where(np.diff(idx) > 1)[0] + 1
      groups = np.split(idx, splits)

      rel_speed = np.sqrt((vx1[i1] - vx2[i2]) ** 2 + (vy1[i1] - vy2[i2]) ** 2)

      hdiff = np.abs(h1[i1] - h2[i2]) # absolute heading difference
      hdiff = np.where(hdiff > np.pi, 2 * np.pi - hdiff, hdiff)

      side = np.sign(dx * vy2[i2] - dy * vx2[i2])

      for g in groups:
        start_t = float(common[g[0]])
        end_t   = float(common[g[-1]])
        duration = end_t - start_t

        if duration < min_duration or start_t == end_t:
          continue

        seg_dist = np.sqrt(dist_sq[g])
        min_d = float(seg_dist.min())
        mean_d = float(seg_dist.mean())

        rel_speed_mean = float(rel_speed[g].mean())
        rel_head_mean = float(np.degrees(hdiff[g].mean()))

        seg_side = side[g]
        is_crossing = np.any(seg_side != seg_side[0])

        out.append((
          t1, t2,
          start_t, end_t,
          round(duration, 2),
          round(min_d, 3),
          round(mean_d, 3),
          round(rel_speed_mean, 3),
          round(rel_head_mean, 2),
          int(is_crossing)
        ))

  return out

def detect_interactions(df: pd.DataFrame, track_ids: List[int], distance: float) -> Dict[int, List[int]]:
  """Queries the STR-Tree to gain intersecting trajectory segments and candidates for interactions."""
  target_dict = traj_list_to_linestrings(df[df["track_id"].isin(track_ids)])
  traj_dict = traj_list_to_linestrings(df)
  tree, segments, segment_lookup = build_str_tree(traj_dict)

  result = {}
  for traj_id, traj in target_dict.items():
    tube = traj.buffer(distance)
    seg_ids = tree.query(tube)
    candidates = set(segment_lookup[segments[seg_id]] for seg_id in seg_ids)
    result[traj_id] = [c for c in candidates if traj_id != c]
  return result

def get_interactions(
    df: pd.DataFrame,
    track_ids: List[int],
    distance: float,
    min_duration: float=None,
    batch_size: int=5
) -> List[Tuple[int, int, float, float, float, float, float, int]]:
  """Get interactions between the specified and other participants. Limits the time window to the batch of trajectories."""
  result = []
  for batch in tqdm(range(0, len(track_ids), batch_size)):
    batch_ids = track_ids[batch:batch+batch_size]
    # limit df to the time window of target tracks
    t0, t1 = get_time_window(df, batch_ids)
    df_window = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)]

    candidates = detect_interactions(df_window, batch_ids, distance)
    interactions = filter_interactions(df_window, candidates, distance, min_duration=min_duration)
    result.extend(interactions)
  return result

def save_interactions_to_csv(interactions, path):
  """Save interactions to a csv file."""
  with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(INTERACTION_FEATURES)
    writer.writerows(interactions)