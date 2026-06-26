import numpy as np
import pandas as pd

from features.adapters import FeatureAdapter


"""
This class maps clusters to "stable" or "volatile" regimes based on the average volatility of the cluster centers.
The cluster with the lowest average volatility is labeled "stable", while the one with the highest is labeled "volatile".
This was an early class used to interpret the clusters, before the integration of the regime aggregation.

The class is still used in some notebooks, but solely for cluster labels in plots a.
"""

class RegimeClusterMapper:
  def __init__(self):
    self.cluster_map = {}

  def fit(self, df_centers: pd.DataFrame, adapter: FeatureAdapter):
    volatility_features = [
      n for n in adapter.feature_names
      if any(k in n for k in ("std", "mad", "qcv", "cv", "var"))
    ]
    df_vol = df_centers[volatility_features]
    volatility_mean = df_vol.mean(axis=1)

    volatile_cluster = volatility_mean.idxmax()
    stable_cluster = volatility_mean.idxmin()

    self.cluster_map = {
      stable_cluster: "stable",
      volatile_cluster: "volatile"
    }
    return self.cluster_map

  def is_stable(self, sequence, as_numpy=False):
    if not self.cluster_map:
      raise ValueError("Cluster map not initialized")

    bool_sequence = [
      self.cluster_map[c] == "stable"
      for c in sequence
    ]

    if as_numpy:
      return np.array(bool_sequence, dtype=bool)

    return bool_sequence
