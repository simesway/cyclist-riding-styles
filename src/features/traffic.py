from typing import Optional

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


def ttc_aggregates(
    df: pd.DataFrame,
    r_safe: float,
    t_thresh: float,
    category_id: Optional[int] = None,
    prefix: str = "",
    max_horizon=np.inf,
):
  """
  Per-timestamp TTC / DCA aggregates:
  - min_ttc
  - any_ttc_below_t
  - num_ttc_below_t
  - min_dca
  """
  req_cols = [
    "timestamp", "distance", "dx", "dy",
    "vx_rel", "vy_rel", "category"
  ]
  missing = [c for c in req_cols if c not in df.columns]
  assert not missing, f"Missing column {', '.join(missing)}"

  df = df[req_cols].copy()
  finite = np.isfinite(df[req_cols]).all(axis=1)
  df = df[finite].copy()
  if category_id is not None:
    df = df[df["category"] == category_id]

  if prefix:
    prefix = f"{prefix}_"

  eps = 1e-6

  # relative velocity
  dv2 = df["vx_rel"] ** 2 + df["vy_rel"] ** 2
  dot = df["dx"] * df["vx_rel"] + df["dy"] * df["vy_rel"]

  # distance squared
  d2 = df["distance"] ** 2

  # TTC to safety radius
  disc = dot ** 2 - dv2 * (d2 - r_safe ** 2)
  disc = np.maximum(disc, 0)
  valid = dv2 > eps

  ttc = np.full(len(df), np.inf)
  ttc[valid] = (-dot[valid] - np.sqrt(disc[valid])) / dv2[valid]
  ttc = np.clip(ttc, 0, max_horizon)
  ttc[np.isnan(ttc)] = np.inf

  # DCA (distance at closest approach)
  t_star = np.zeros(len(df))
  t_star[dv2 > eps] = -dot[dv2 > eps] / dv2[dv2 > eps]
  t_star = np.clip(t_star, 0, max_horizon)

  dca = np.sqrt(
    (df["dx"] + df["vx_rel"] * t_star) ** 2 +
    (df["dy"] + df["vy_rel"] * t_star) ** 2
  )
  dca[np.isnan(dca)] = np.inf

  df["ttc"] = ttc
  df["dca"] = dca

  # replace inf by nan to avoid subtraction by inf in aggregate
  df["ttc"] = df["ttc"].replace(np.inf, np.nan)
  df["dca"] = df["dca"].replace(np.inf, np.nan)

  agg = (
    df.groupby("timestamp")
    .agg(
      **{
        f"{prefix}min_ttc": ("ttc", "min"),
        f"{prefix}any_ttc_below_th":
          ("ttc", lambda x: np.any(x < t_thresh)),
        f"{prefix}num_ttc_below_th":
          ("ttc", lambda x: np.sum(x < t_thresh)),
        f"{prefix}min_dca": ("dca", "min"),
      }
    )
    .reset_index()
  )

  return agg


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