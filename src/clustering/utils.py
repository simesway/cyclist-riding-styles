import numpy as np

from features.base import RidingFeatures
from maneuvers.base import WindowRecord
from features.adapters import FeatureAdapter


def build_feature_matrix(windows: list[WindowRecord]):

  adapter = FeatureAdapter(RidingFeatures)
  X = []
  valid_windows = []

  for w in windows:
    if w.riding is None:
        continue
    X.append(adapter.to_vector(w.riding))
    valid_windows.append(w)

  return np.asarray(X), valid_windows