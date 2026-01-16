import pandas as pd

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import trafficfeatures.instant as inst

from features.vehicle_dynamics import longitudinal_acceleration, rotation_fluctuation_signal
from features.volatility import stats_basic, max_abs
from maneuvers.base import Maneuver
from features.base import RidingFeatures, TrafficFeatures, InfrastructureFeatures

T = TypeVar("T")


class FeatureExtractor(ABC, Generic[T]):
  """Base class for window-level feature extractors"""

  @abstractmethod
  def prepare(self, df) -> pd.DataFrame:
    """Prepares the Maneuver DataFrame for feature extraction (e.g., computes derived signals)"""
    pass

  @abstractmethod
  def extract(self, window_df, **kwargs) -> T:
    """Extracts features from the given window DataFrame"""
    pass


class NullExtractor(FeatureExtractor[None]):
  def prepare(self, df) -> pd.DataFrame:
    return df

  def extract(self, *_, **__):
    return None


class RidingFeatureExtractor(FeatureExtractor[RidingFeatures]):
  def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
    df["velocity"] = inst.magnitude(df, ["velocity_x", "velocity_y"])
    df["acceleration"] = inst.magnitude(df, ["acceleration_x", "acceleration_y"])
    df["long_acc"] = pd.Series(longitudinal_acceleration(df), index=df.index)
    df["rot_fluc"] = pd.Series(rotation_fluctuation_signal(df), index=df.index)
    return df

  def extract(self, df: pd.DataFrame, **_) -> RidingFeatures:
    speed = stats_basic(df, "velocity")
    acc = stats_basic(df, "long_acc") # added in WindowBuilder.infer_features
    rot_fluc = stats_basic(df, "rot_fluc")
    return RidingFeatures(
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
  def prepare(self, df) -> pd.DataFrame:
    # No special preparation needed
    return df

  def extract(self, df, **_) -> InfrastructureFeatures:
    return InfrastructureFeatures(
      on_motorway=bool(df["on_motorway"].mean() > 0.5),
      on_bikelane=bool(df["on_bikelane"].mean() > 0.5),
      on_sidewalk=bool(df["on_sidewalk"].mean() > 0.5),

      offset_lane_center=float(df["offset_lane_center"].mean()),
      rel_offset_lane_center=float(df["rel_offset_lane_center"].mean()),
      min_lateral_clearance=float(df["min_lateral_clearance"].mean()),
      max_lateral_clearance=float(df["max_lateral_clearance"].mean()),
      lane_width=float(df["lane_width"].mean()),

      distance_traffic_light=float(df["distance_traffic_light"].mean()),
      distance_regulatory_sign=float(df["distance_regulatory_sign"].mean()),
      distance_warning_sign=float(df["distance_warning_sign"].mean()),
    )
