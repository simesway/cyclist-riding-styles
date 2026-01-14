import pandas as pd

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from features.volatility import stats_acc, stats_basic
from maneuvers.base import Maneuver
from features.base import RidingFeatures, TrafficFeatures, InfrastructureFeatures

T = TypeVar("T")


class FeatureExtractor(ABC, Generic[T]):
  """Base class for window-level feature extractors"""

  @abstractmethod
  def extract(self, window_df, **kwargs) -> T:
    pass


class NullExtractor(FeatureExtractor[None]):
  def extract(self, *_, **__):
    return None


class RidingFeatureExtractor(FeatureExtractor[RidingFeatures]):
  def extract(self, df: pd.DataFrame, **_) -> RidingFeatures:
    speed = stats_basic(df, "speed")
    acc = stats_basic(df, "long_acc") # added in WindowBuilder.infer_features
    return RidingFeatures(
      speed_max=speed["max"],
      speed_min=speed["min"],
      speed_mean=speed["mean"],
      speed_std=speed["std"],
      speed_mad=speed["mad"],
      speed_qcv=speed["qcv"],
      acc_max=acc["max"],
      acc_min=acc["min"],
      acc_mean=acc["mean"],
      acc_std=acc["std"],
      acc_mad=acc["mad"],
      acc_qcv=acc["qcv"],
      speed_max=float(speed["max"]),
      speed_min=float(speed["min"]),
      speed_mean=float(speed["mean"]),
      speed_std=float(speed["std"]),
      speed_mad=float(speed["mad"]),
      speed_qcv=float(speed["qcv"]),
      acc_max=float(acc["max"]),
      acc_min=float(acc["min"]),
      acc_mean=float(acc["mean"]),
      acc_std=float(acc["std"]),
      acc_mad=float(acc["mad"]),
      acc_qcv=float(acc["qcv"]),
    )

class TrafficFeatureExtractor(FeatureExtractor[TrafficFeatures]):
  """Extracts leader-follower interaction and traffic features"""
  def extract(self, window_df, **kwargs) -> TrafficFeatures:
    maneuver: Maneuver = kwargs["maneuver"]
    ...

class InfrastructureFeatureExtractor(FeatureExtractor[InfrastructureFeatures]):
  """Infers slow-varying infrastructure features per window"""
  def extract(self, window_df, **_) -> InfrastructureFeatures:
    ...
