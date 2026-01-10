import numpy as np
import pandas as pd
from typing import List

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from clustering.clusterer import Clusterer
from clustering.pca import PCAReducer, SplitPCAReducer
from clustering.stability import RegimeStabilityTester
from clustering.utils import build_feature_matrix
from features.adapters import FeatureAdapter
from maneuvers.base import WindowRecord


class RidingRegimePipeline:
  def __init__(
    self,
    feature_adapter: FeatureAdapter,
    clusterer: Clusterer,
    pca: PCAReducer | SplitPCAReducer,
  ):
    self.feature_adapter = feature_adapter
    self.clusterer = clusterer
    self.pca = pca
    self.scaler = StandardScaler()

  def run(self, windows: List[WindowRecord]) -> None:
    """
    Modifies WindowRecords in-place by assigning
    w.meta.local_regime
    """
    X, ws = build_feature_matrix(windows, self.feature_adapter)

    if len(ws) == 0:
      raise ValueError("No windows with riding features")

    X = self.scaler.fit_transform(X)

    if self.pca is not None:
      X = self.pca.fit_transform(X)

    labels = self.clusterer.fit_predict(X)

    if len(labels) != len(ws):
      raise RuntimeError("Label/window mismatch")

    for w, label in zip(ws, labels):
      w.local_regime = int(label)

  def get_cluster_centers(self, in_original_units=True) -> pd.DataFrame:
    centers_pc = self.clusterer.model_.cluster_centers_

    centers = self.pca.inverse_transform(centers_pc)
    if in_original_units:
      centers = self.scaler.inverse_transform(centers)

    df = pd.DataFrame(centers, columns=self.feature_adapter.feature_names)
    df.index.name = "cluster_id"
    return df

  def stability_test(
      self,
      windows: List[WindowRecord],
      subsample_frac: float=0.8,
      noise_scale: float=0.1,
      n_runs: int=30
  ) -> pd.DataFrame:
    X, ws = build_feature_matrix(windows, self.feature_adapter)

    if len(ws) == 0:
      raise ValueError("No windows with riding features")

    X_scaled = self.scaler.fit_transform(X)
    if self.pca is not None:
      X_scaled = self.pca.fit_transform(X_scaled)

    def factory(random_state=None):
      # return a fresh instance of the clusterer with given random state
      new_clusterer = self.clusterer.__class__(**self.clusterer.get_params())
      if random_state is not None:
        new_clusterer.random_state=random_state
      return new_clusterer

    tester = RegimeStabilityTester(
      clusterer_factory=factory,
      noise_scale=noise_scale,
      subsample_frac=subsample_frac,
      random_state=0
    )

    return tester.run_repeated(X_scaled, n_runs=n_runs)

  def compute_cluster_metrics(self, windows: List[WindowRecord]) -> pd.DataFrame:
    """
    Compute overall clustering metrics for the current local_regime labels.
    """
    labels = [w.local_regime for w in windows if hasattr(w, "local_regime")]
    if len(labels) == 0:
      raise ValueError("No windows have been clustered yet")

    X, ws = build_feature_matrix(windows, self.feature_adapter)
    if len(ws) == 0:
      raise ValueError("No windows with features")

    X_scaled = self.scaler.transform(X)
    if self.pca is not None:
      X_scaled = self.pca.transform(X_scaled)

    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))

    metrics = {}
    if len(np.unique(labels)) > 1:
      metrics['silhouette'] = silhouette_score(X_scaled, labels)
      metrics['calinski_harabasz'] = calinski_harabasz_score(X_scaled, labels)
      metrics['davies_bouldin'] = davies_bouldin_score(X_scaled, labels)
    else:
      metrics['silhouette'] = np.nan
      metrics['calinski_harabasz'] = np.nan
      metrics['davies_bouldin'] = np.nan

    metrics.update({f'cluster_{k}_size': v for k, v in cluster_sizes.items()})
    return pd.DataFrame([metrics])

