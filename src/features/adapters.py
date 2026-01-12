from dataclasses import fields as dataclass_fields
from typing import List, Callable, Any, Iterable, Union

import numpy as np
import pandas as pd

from features.base import RegimeAggregation, OvertakingFeatures, FollowingFeatures


class FeatureAdapter:
  def __init__(self, feature_class, exclude: List[str] = None):
    self.feature_class = feature_class
    self._names = [f.name for f in dataclass_fields(feature_class)]
    self.exclude = set(exclude or [])
    self._active_names = [n for n in self._names if n not in self.exclude]

  def to_vector(self, f) -> List[float]:
    return [getattr(f, name) for name in self._active_names]

  def from_vector(self, v: List[float]):
    if len(v) != len(self._active_names):
      raise ValueError("Vector length mismatch")
    return self.feature_class(**dict(zip(self._active_names, v)))

  @property
  def magnitude_idx(self) -> List[int]:
    return [
      i for i, n in enumerate(self._names)
      if any(k in n for k in ("min", "max", "mean"))
         and not any(k in n for k in ("std", "mad", "qcv", "cv", "var"))
         and n in self._active_names
    ]

  @property
  def volatility_idx(self) -> List[int]:
    return [
      i for i, n in enumerate(self._names)
      if any(k in n for k in ("std", "mad", "qcv", "cv", "var")) and n in self._active_names
    ]

  @property
  def feature_names(self) -> List[str]:
    return self._active_names


class ManeuverAdapter:
  def __init__(
      self,
      feature_cls: Union[OvertakingFeatures, FollowingFeatures],
      exclude: List[str] = None,
  ):
    self.exclude = set(exclude or [])

    self.feature_adapter = FeatureAdapter(
      feature_cls,
      exclude=list(self.exclude)
    )

    self.regime_adapter = FeatureAdapter(
      RegimeAggregation,
      exclude=list(self.exclude)
    )

  def to_vector(self, m) -> List[float]:
    v = []
    v.extend(self.feature_adapter.to_vector(m.features))
    v.extend(self.regime_adapter.to_vector(m.regime_aggregation))
    return v

  @property
  def feature_names(self) -> List[str]:
    return self.feature_adapter.feature_names + self.regime_adapter.feature_names


class FeatureMatrixBuilder:
    def __init__(self, adapter, getter: Callable[[Any], Any], drop_none: bool = True):
        self.adapter = adapter
        self.getter = getter
        self.drop_none = drop_none

    def _collect(self, items: Iterable[Any]):
        X = []
        valid_items = []

        for item in items:
            obj = self.getter(item)
            if obj is None and self.drop_none:
                continue
            X.append(self.adapter.to_vector(obj))
            valid_items.append(item)

        return X, valid_items

    def to_numpy(self, items: Iterable[Any]):
        X, valid_items = self._collect(items)
        return np.asarray(X), valid_items

    def to_dataframe(self, items: Iterable[Any]):
        X, valid_items = self._collect(items)
        df = pd.DataFrame(X, columns=self.adapter.feature_names)
        return df, valid_items
