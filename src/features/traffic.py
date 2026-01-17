import numpy as np
import pandas as pd


def counts_within_radius(df: pd.DataFrame, category_id: int, radius, min_velocity=0.0):
  """
  Compute counts per timestamp for a given category, within a radius,
  optionally filtering by minimum velocity.
  """
  req_cols = ["timestamp", "distance", "category", "velocity"]
  missing = [col for col in req_cols if col not in df.columns]
  assert not missing, f"Missing column {', '.join(missing)}"

  df = df[req_cols].copy()
  mask = (
      (df["distance"] <= radius) &
      (df["category"] == category_id) &
      (df["velocity"] >= min_velocity)
  )
  df = df[mask]

  counts = df.groupby("timestamp").size().reset_index(name="count")

  return counts[["timestamp", "count"]]


def density(df: pd.DataFrame, category_id: int, radius, min_velocity=0.0):
  """
  Compute densities per timestamp for a given category within a radius.
  Internally uses counts_within_radius().
  """
  counts = counts_within_radius(df, category_id, radius, min_velocity)

  area = np.pi * radius ** 2
  counts["density"] = counts["count"] / area

  return counts[["timestamp", "density"]]


def merge_column(ego_df, other_df, col_name: str, new_name=None, fillna=None):
  """
  Left-join a density DataFrame to ego_df on 'timestamp'.
  """
  if new_name is None:
    new_name = col_name

  merged = ego_df.merge(
    other_df[["timestamp", col_name]].rename(columns={col_name: new_name}),
    on="timestamp",
    how="left"
  )
  if fillna is not None:
    merged[new_name] = merged[new_name].fillna(fillna)
  return merged