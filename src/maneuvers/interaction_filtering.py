import numpy as np
import pandas as pd
from data.transform import traj_list_to_linestrings
from data.utils import get_time_window
from tqdm import tqdm


def filter_raw_interactions(
    raw_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    min_distance=10.0,  # meters
    min_overlap=5.0,  # seconds
    max_heading_diff=45  # degrees
):
  filtered = []

  # data cleaning
  no_traj_duplicates = ~((interactions_df["dist_mean"] < 0.5) | (interactions_df["rel_speed"] == 0))
  min_distance_filter = (interactions_df["dist_min"] < 20) & (interactions_df["dist_mean"] < 40) & (interactions_df["rel_heading"] <= max_heading_diff)
  mask = no_traj_duplicates & min_distance_filter
  interactions_df = interactions_df[mask]
  raw_df.sort_values(by="timestamp", inplace=True)

  for idx, row in tqdm(interactions_df.iterrows(), total=interactions_df.shape[0]):
    track_a = row['track_id']
    track_b = row['other_id']
    t_start, t_end = row['t_start'], row['t_end']

    subset = raw_df[raw_df['track_id'].isin([track_a, track_b])]
    time_w = subset[(subset['timestamp'] >= t_start) & (subset['timestamp'] <= t_end)]

    # Check if heading difference is small (Kutsch et al. 2025) (Feng & Munnamgi)
    #if row["rel_heading"] > max_heading_diff:
    #  continue

    # Co-existing for min overlap (Kutsch et al. 2025)
    t_start_a, t_end_a = get_time_window(subset, [track_a])
    t_start_b, t_end_b = get_time_window(subset, [track_b])
    t_overlap = min(t_end_a, t_end_b) - max(t_start_a, t_start_b)
    if t_overlap < min_overlap:
      continue

    # Both moved for at least min_distance meters (Kutsch et al. 2025)
    lines = traj_list_to_linestrings(time_w)
    if lines[track_a].length < min_distance or lines[track_b].length < min_distance:
      continue

    filtered.append(row)

  filtered_df = pd.DataFrame(filtered).reset_index(drop=True)
  int_cols = ['track_id', 'other_id', 'is_crossing']
  filtered_df[int_cols] = filtered_df[int_cols].astype(int)
  return filtered_df