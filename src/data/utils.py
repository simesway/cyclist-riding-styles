import pandas as pd
from typing import List, Tuple

def get_time_window(df: pd.DataFrame, traj_ids: List[int]) -> Tuple[int, int]:
  """Extract the time window from a dataframe for the given trajectories."""
  subset = df[df["track_id"].isin(traj_ids)]
  return subset["timestamp"].min(), subset["timestamp"].max()