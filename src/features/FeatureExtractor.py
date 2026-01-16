import pandas as pd

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import trafficfeatures.instant as inst
import trafficfeatures.opendrive as odr

from features.infrastructure import on_lane, offset_lane_center, rel_offset_norm, min_lateral_clearance, \
  max_lateral_clearance, distance_to_signal
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
      rot_fluc_mad=float(rot_fluc["mad"]),
      rot_fluc_std=float(rot_fluc["std"]),
      rot_fluc_max_abs=float(max_abs(df["rot_fluc"].to_numpy()))
    )

class TrafficFeatureExtractor(FeatureExtractor[TrafficFeatures]):
  """Extracts leader-follower interaction and traffic features"""
  def extract(self, window_df, **kwargs) -> TrafficFeatures:
    maneuver: Maneuver = kwargs["maneuver"]
    ...

class InfrastructureFeatureExtractor(FeatureExtractor[InfrastructureFeatures]):
  def __init__(self):
    self.signals = odr.get_signals()

  def prepare(self, df) -> pd.DataFrame:
    df["on_bikelane"] = on_lane(df, ["biking"])
    df["on_sidewalk"] = on_lane(df, ["sidewalk"])
    df["on_motorway"] = on_lane(df, ["driving", "shoulder", "parking", "tram"])

    df["offset_lane_center"] = offset_lane_center(df)
    df["rel_offset_lane_center"] = rel_offset_norm(df)
    df["min_lateral_clearance"] = min_lateral_clearance(df)
    df["max_lateral_clearance"] = max_lateral_clearance(df)

    # distance based on (Rupi & Krizek 2019)
    df["distance_traffic_light"] = distance_to_signal(df, self.signals, "TrafficLight", max_radius=60)
    df["distance_regulatory_sign"] = distance_to_signal(df, self.signals, "RegulatorySign", max_radius=30)
    df["distance_warning_sign"] = distance_to_signal(df, self.signals, "WarningSign", max_radius=20)

    return df

  def extract(self, df, **_) -> InfrastructureFeatures:
    return InfrastructureFeatures(
      on_motorway=bool(df["on_motorway"].any()),
      on_bikelane=bool(df["on_bikelane"].mean() > 0.5),
      on_sidewalk=bool(df["on_sidewalk"].any()),

      offset_lane_center=float(df["offset_lane_center"].mean()),
      rel_offset_lane_center=float(df["rel_offset_lane_center"].mean()),

      left_clearance=float(df["inner_dist"].mean()),
      min_lateral_clearance=float(df["min_lateral_clearance"].mean()),
      max_lateral_clearance=float(df["max_lateral_clearance"].mean()),
      lane_width=float(df["lane_width"].mean()),

      distance_traffic_light=float(df["distance_traffic_light"].mean()),
      distance_regulatory_sign=float(df["distance_regulatory_sign"].mean()),
      distance_warning_sign=float(df["distance_warning_sign"].mean()),
    )
