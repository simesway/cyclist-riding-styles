import numpy as np
import pandas as pd
from typing import List, Callable, Iterable, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from clustering.clusterer import Clusterer
from clustering.pca import PCAReducer, SplitPCAReducer
from clustering.stability import RegimeStabilityTester
from features.adapters import FeatureAdapter, FeatureMatrixBuilder, ManeuverAdapter


class ClusteringPipeline:
  def __init__(
    self,
    adapter: FeatureAdapter | ManeuverAdapter,
    getter: Callable,
    clusterer: Clusterer,
    pca: PCAReducer | SplitPCAReducer,
  ):
    self.adapter = adapter
    self.builder = FeatureMatrixBuilder(adapter, getter, drop_none=False)
    self.clusterer = clusterer
    self.pca = pca
    self.scaler = StandardScaler()

    self._X_scaled = None
    self._labels = None

  def run(self, items: Iterable) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Modifies WindowRecords in-place by assigning
    w.meta.local_regime
    """
    X, valid_items = self.builder.to_numpy(items)

    if len(valid_items) == 0:
      raise ValueError("No windows with riding features")

    X = self.scaler.fit_transform(X)

    if self.pca is not None:
      X = self.pca.fit_transform(X)

    labels = self.clusterer.fit_predict(X)

    if len(labels) != len(valid_items):
      raise RuntimeError("Label/valid_items mismatch")

    self._X_scaled = X
    self._labels = labels

    return X, labels, valid_items

  def get_cluster_centers(self, in_original_units=True) -> pd.DataFrame:
    centers_pc = self.clusterer.model_.cluster_centers_

    centers = self.pca.inverse_transform(centers_pc)
    if in_original_units:
      centers = self.scaler.inverse_transform(centers)

    df = pd.DataFrame(centers, columns=self.adapter.feature_names)
    df.index.name = "cluster_id"
    return df

  def stability_test(
      self,
      items: Iterable,
      subsample_frac: float=0.8,
      noise_scale: float=0.1,
      n_runs: int=30
  ) -> pd.DataFrame:
    X, ws = self.builder.to_numpy(items)

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

  @staticmethod
  def compute_cluster_metrics_from_X(
      X: np.ndarray,
      labels: np.ndarray,
  ) -> pd.DataFrame:
    metrics = {}

    unique, counts = np.unique(labels, return_counts=True)
    metrics.update({f"cluster_{k}_size": v for k, v in zip(unique, counts)})

    if len(unique) > 1:
      metrics["silhouette"] = silhouette_score(X, labels)
      metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
      metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
    else:
      metrics["silhouette"] = np.nan
      metrics["calinski_harabasz"] = np.nan
      metrics["davies_bouldin"] = np.nan

    return pd.DataFrame([metrics])

  def compute_cluster_metrics(self) -> pd.DataFrame:
    if self._labels is None or self._X_scaled is None:
        raise RuntimeError("Pipeline has not been run")
    return self.compute_cluster_metrics_from_X(
      self._X_scaled,
      self._labels
    )
