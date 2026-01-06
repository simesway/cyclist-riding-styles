import pandas as pd
from typing import List

from sklearn.preprocessing import StandardScaler

from clustering.clusterer import Clusterer
from clustering.pca import PCAReducer, SplitPCAReducer
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
    X, ws = build_feature_matrix(windows)

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
