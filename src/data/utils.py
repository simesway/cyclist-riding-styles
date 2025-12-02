import pandas as pd
import numpy as np
from typing import List, Tuple

def get_time_window(df: pd.DataFrame, traj_ids: List[int]) -> Tuple[int, int]:
  """Extract the time window from a dataframe for the given trajectories."""
  subset = df[df["track_id"].isin(traj_ids)]
  return subset["timestamp"].min(), subset["timestamp"].max()


def apply_time_window(df: pd.DataFrame, t_start: int=None, t_end: int=None, t_col="timestamp") -> pd.DataFrame:
  mask = df[t_col].between(t_start, t_end, inclusive="both")
  return df[mask]


def clean_heading(h_raw: np.ndarray) -> np.ndarray:
  h = h_raw.copy()

  #h = np.unwrap(h)

  ref = np.median(h[:max(25, len(h) // 2)])

  diff = h[0] - ref
  h[0] -= np.round(diff / np.pi) * np.pi

  # propagate corrections relative to previous frame
  for i in range(1, len(h)):
    delta = h[i] - h[i - 1]
    h[i] -= np.round(delta / np.pi) * np.pi

  return h