import numpy as np
import pandas as pd
from typing import List, Optional

from features.base import RegimeAggregation
from maneuvers.base import Maneuver, WindowRecord
from clustering.semantics import RegimeClusterMapper


class ManeuverAggregator:
  def __init__(self, maneuvers: List[Maneuver], windows: List[WindowRecord]):
    """
    Aggregator to compute features at the maneuver and cluster level from window-level features.
     - build_windows_df: converts list of WindowRecords to a flat DataFrame for easier aggregation
     - aggregate_per_maneuver: computes mean (and optionally std) of window features per maneuver, and attaches cluster_id
     - aggregate_per_cluster: averages maneuver-level features across maneuvers in the same cluster to get cluster-level features
     Note: this class assumes that maneuvers have already been assigned to clusters (i.e. cluster_id is set).
    """
    self.maneuvers = {m.id: m for m in maneuvers}  # map id -> Maneuver
    self.windows = windows
    self.windows_df: Optional[pd.DataFrame] = None
    self.maneuver_df: Optional[pd.DataFrame] = None
    self.cluster_df: Optional[pd.DataFrame] = None

  def build_windows_df(self) -> pd.DataFrame:
    """Convert windows to a DataFrame, flattening the WindowRecord structure."""
    self.windows_df = pd.DataFrame([w.flatten() for w in self.windows])
    return self.windows_df

  def aggregate_per_maneuver(self, add_std: bool = True, drop=None) -> pd.DataFrame:
    """Aggregate window features per maneuver, optionally adding std deviation."""
    if self.windows_df is None:
      self.build_windows_df()

    if drop is None:
      drop = ["ego_id", "t_start", "t_end", ]

    numeric_cols = self.windows_df.drop(columns=drop).select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "maneuver_id"]  # keep maneuver_id

    agg_map = {col: ["mean", "std"] if add_std else "mean" for col in numeric_cols}

    self.maneuver_df = self.windows_df.groupby("maneuver_id", as_index=False).agg(agg_map)

    # flatten MultiIndex columns if std added
    if add_std:
      self.maneuver_df.columns = [
        f"{i}_{j}" if j else f"{i}" for i, j in self.maneuver_df.columns
      ]

    self.maneuver_df["cluster_id"] = self.maneuver_df["maneuver_id"].map(
      lambda mid: self.maneuvers[mid].cluster_id
    )

    self.maneuver_df.set_index('maneuver_id', inplace=True)

    return self.maneuver_df

  def aggregate_per_cluster(self) -> pd.DataFrame:
    """Aggregate maneuver features per cluster by averaging across maneuvers in the same cluster."""
    if self.maneuver_df is None:
      raise ValueError("Call aggregate_per_maneuver first.")

    numeric_cols = self.maneuver_df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "cluster_id"]

    cluster_agg_map = {col: "mean" for col in numeric_cols}

    self.cluster_df = self.maneuver_df.groupby("cluster_id", as_index=False).agg(cluster_agg_map)

    self.cluster_df.set_index("cluster_id", inplace=True)
    return self.cluster_df


class LocalRegimeAggregator:
  """
  Old aggregator that assumed a 2 cluster solution (stable and volatile) for each scenario.
  Kept for reference but not used in main pipeline.
  new version in src/maneuvers/regime_aggregation.py
  """
  @staticmethod
  def aggregate(
      maneuver: Maneuver,
      windows: List[WindowRecord],
      regime_mapper: RegimeClusterMapper,
      use_volatile_transition_rate: bool = False,
      attach: bool=True
  ) -> RegimeAggregation:
    """Aggregate local regime assignments for a given maneuver."""
    windows = [w for w in windows if w.meta.maneuver_id == maneuver.id]
    w_sorted = sorted(windows, key=lambda w: w.t_start)
    regimes = [w.local_regime for w in w_sorted if w.local_regime is not None]

    is_stable = regime_mapper.is_stable(regimes, as_numpy=True)

    N = len(regimes)
    if N == 0:
      raise ValueError("No regimes assigned to maneuver windows")

    volatile = ~is_stable # e.g. volatile = 0 1 1 0 1

    p_volatile = volatile.mean()

    if use_volatile_transition_rate:
      # volatile-based transition rate (stable -> volatile)
      volatile_onsets = (is_stable[:-1] == True) & (is_stable[1:] == False)
      transition_rate = volatile_onsets.sum() / max(N - 1, 1)
    else:
      # general transition rate (stable -> volatile & volatile -> stable)
      transitions = is_stable[:-1] != is_stable[1:]
      transition_rate = transitions.sum() / max(N-1, 1)

    padded = np.concatenate(([0], volatile.view(np.int8), [0])) # padded = 0 0 1 1 0 1 0
    diff = np.diff(padded) # diff:   0 1 0 -1 1 -1
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    run_lengths = ends - starts

    if len(run_lengths) == 0:
      mean_run_volatile = 0.0
      std_run_volatile = 0.0
      mean_distance_between_volatile = N
    else:
      mean_run_volatile = run_lengths.mean()
      std_run_volatile = run_lengths.std()

      if len(starts) < 2:
        mean_distance_between_volatile = N
      else:
        distances = starts[1:] - ends[:-1]
        mean_distance_between_volatile = distances.mean()


    regime_aggregation = RegimeAggregation(
        maneuver_id=int(maneuver.id),
        n_windows=int(N),
        is_active=bool(0.0 < p_volatile < 1.0),
        p_volatile=float(p_volatile),
        transition_rate=float(transition_rate),
        mean_run_volatile=float(mean_run_volatile),
        std_run_volatile=float(std_run_volatile),
        mean_volatile_gap=float(mean_distance_between_volatile/N)
      )

    if attach:
      maneuver.regime_aggregation = regime_aggregation

    return regime_aggregation
