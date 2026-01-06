from dataclasses import fields as dataclass_fields
from typing import List


class FeatureAdapter:
  def __init__(self, feature_class):
    self.feature_class = feature_class
    self._names = [f.name for f in dataclass_fields(feature_class)]

  def to_vector(self, f) -> list[float]:
    return [getattr(f, name) for name in self._names]

  def from_vector(self, v: list[float]):
    if len(v) != len(self._names):
      raise ValueError("Vector length mismatch")
    return self.feature_class(**dict(zip(self._names, v)))

  @property
  def magnitude_idx(self) -> List[int]:
    return [
      i for i, n in enumerate(self._names)
      if any(k in n for k in ("min", "max", "mean"))
         and not any(k in n for k in ("std", "mad", "qcv", "cv", "var"))
    ]

  @property
  def volatility_idx(self) -> List[int]:
    return [
      i for i, n in enumerate(self._names)
      if any(k in n for k in ("std", "mad", "qcv", "cv", "var"))
    ]

  @property
  def feature_names(self) -> List[str]:
    return self._names