from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from maneuvers.base import RidingFeatures, TrafficFeatures, Maneuver, InfrastructureFeatures

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
  def extract(self, window_df, **_) -> RidingFeatures:
    ...

class TrafficFeatureExtractor(FeatureExtractor[TrafficFeatures]):
  """Extracts leader-follower interaction and traffic features"""
  def extract(self, window_df, **kwargs) -> TrafficFeatures:
    maneuver: Maneuver = kwargs["maneuver"]
    ...

class InfrastructureFeatureExtractor(FeatureExtractor[InfrastructureFeatures]):
  """Infers slow-varying infrastructure features per window"""
  def extract(self, window_df, **_) -> InfrastructureFeatures:
    ...
