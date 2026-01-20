import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import trafficfeatures.instant as inst
import trafficfeatures.opendrive as odr

from features.infrastructure import on_lane, offset_lane_center, rel_offset_norm, min_lateral_clearance, \
  max_lateral_clearance, distance_to_signal
from features.traffic import merge_column, counts_within_radius, ttc_aggregates
from features.vehicle_dynamics import longitudinal_acceleration, rotation_fluctuation_signal
from features.volatility import stats_basic, max_abs
from maneuvers.base import Maneuver
from features.base import RidingFeatures, TrafficFeatures, InfrastructureFeatures
from maneuvers.encounter_extractor import EncounterExtractor

T = TypeVar("T")


class FeatureExtractor(ABC, Generic[T]):
  """Base class for window-level feature extractors"""

  @abstractmethod
  def prepare(self, df, maneuver: Maneuver) -> pd.DataFrame:
    """Prepares the Maneuver DataFrame for feature extraction (e.g., computes derived signals)"""
    pass

  @abstractmethod
  def extract(self, window_df, **kwargs) -> T:
    """Extracts features from the given window DataFrame"""
    pass


class NullExtractor(FeatureExtractor[None]):
  def prepare(self, df, maneuver: Maneuver) -> pd.DataFrame:
    return df

  def extract(self, *_, **__):
    return None


class RidingFeatureExtractor(FeatureExtractor[RidingFeatures]):
  def prepare(self, df, maneuver: Maneuver) -> pd.DataFrame:
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
  def __init__(
      self,
      raw_trajectories: pd.DataFrame,
      horizon: float = 6.0,
      risk_ttc_thresh: float=2.0,
      dca_threshold: float=5.0,
  ):
    self.extractor = EncounterExtractor(raw_trajectories)
    self.horizon = horizon
    self.risk_ttc_thresh = risk_ttc_thresh
    self.dca_threshold = dca_threshold

  @staticmethod
  def p10(x):
    return float(x.quantile(0.1)) if len(x) else 0.0

  @staticmethod
  def frac(x):
    return float(x.mean()) if len(x) else 0.0

  @staticmethod
  def exposure(min_ttc, thresh):
    return float((thresh - min_ttc).clip(lower=0).sum())

  def prepare(self, df, maneuver: Maneuver) -> pd.DataFrame:
    sub_df = self.extractor.get_encounters(maneuver.ego_id, 20, maneuver.t_start, maneuver.t_end)

    sub_df = sub_df[(sub_df["distance"] <= 30) & (sub_df["distance"] >= 0.4)]
    sub_df["velocity"] = inst.magnitude(sub_df, ["velocity_x", "velocity_y"])
    sub_df = sub_df[sub_df["velocity"] >= 0.5]

    df = merge_column(
      df, counts_within_radius(sub_df, 1, 20, 5),
      col_name="count", new_name="car_count", fillna=0
    )

    df = merge_column(
      df, counts_within_radius(sub_df, 2, 5, 0.5),
      col_name="count", new_name="pedestrian_count", fillna=0
    )

    df = merge_column(
      df, counts_within_radius(sub_df, 3, 10, 2),
      col_name="count", new_name="bicycle_count", fillna=0
    )

    horizon = self.horizon

    ttc_df = ttc_aggregates(sub_df, 1.2, category_id=1, prefix="car", max_horizon=horizon, t_thresh=self.risk_ttc_thresh)
    df = df.merge(ttc_df, on="timestamp", how="left")
    ttc_df = ttc_aggregates(sub_df, 0.5, category_id=2, prefix="pedestrian", max_horizon=horizon, t_thresh=self.risk_ttc_thresh)
    df = df.merge(ttc_df, on="timestamp", how="left")
    ttc_df = ttc_aggregates(sub_df, .75, category_id=3, prefix="bicycle", max_horizon=horizon, t_thresh=self.risk_ttc_thresh)
    df = df.merge(ttc_df, on="timestamp", how="left")


    fillna_dict = {
      "car_min_ttc": horizon,
      "car_min_dca": np.inf,
      "car_any_ttc_below_th": False,
      "car_num_ttc_below_th": 0,
      "pedestrian_min_ttc": horizon,
      "pedestrian_min_dca": np.inf,
      "pedestrian_any_ttc_below_th": False,
      "pedestrian_num_ttc_below_th": 0,
      "bicycle_min_ttc": horizon,
      "bicycle_min_dca": np.inf,
      "bicycle_any_ttc_below_th": False,
      "bicycle_num_ttc_below_th": 0,
    }

    for col, val in fillna_dict.items():
      if isinstance(val, bool):
        df[col] = df[col].astype(bool)
      else:
        df[col] = df[col].astype(float)
      df[col] = df[col].fillna(val)

    return df

  def extract(self, df, **_) -> TrafficFeatures:
    dca_th = self.dca_threshold
    return TrafficFeatures(
      # Counts
      car_count_mean=float(df["car_count"].mean()),
      car_count_max=float(df["car_count"].max()),
      pedestrian_count_mean=float(df["pedestrian_count"].mean()),
      pedestrian_count_max=float(df["pedestrian_count"].max()),
      bicycle_count_mean=float(df["bicycle_count"].mean()),
      bicycle_count_max=float(df["bicycle_count"].max()),

      # Car
      car_min_ttc_min=float(df["car_min_ttc"].min()),
      car_min_ttc_p10=self.p10(df["car_min_ttc"]),
      car_fraction_ttc_below_th=self.frac(df["car_any_ttc_below_th"]),
      car_max_num_ttc_below_th=float(df["car_num_ttc_below_th"].max()),
      car_min_dca_min=float(df["car_min_dca"].min()),
      car_min_dca_p10=self.p10(df["car_min_dca"]),
      car_fraction_dca_below_th=self.frac(df["car_min_dca"] < dca_th),
      car_ttc_exposure_th=self.exposure(df["car_min_ttc"], self.risk_ttc_thresh),

      # Pedestrian
      pedestrian_min_ttc_min=float(df["pedestrian_min_ttc"].min()),
      pedestrian_min_ttc_p10=self.p10(df["pedestrian_min_ttc"]),
      pedestrian_fraction_ttc_below_th=self.frac(df["pedestrian_any_ttc_below_th"]),
      pedestrian_max_num_ttc_below_th=float(df["pedestrian_num_ttc_below_th"].max()),
      pedestrian_min_dca_min=float(df["pedestrian_min_dca"].min()),
      pedestrian_min_dca_p10=self.p10(df["pedestrian_min_dca"]),
      pedestrian_fraction_dca_below_th=self.frac(df["pedestrian_min_dca"] < dca_th),
      pedestrian_ttc_exposure_th=self.exposure(df["pedestrian_min_ttc"], self.risk_ttc_thresh),

      # Bicycle
      bicycle_min_ttc_min=float(df["bicycle_min_ttc"].min()),
      bicycle_min_ttc_p10=self.p10(df["bicycle_min_ttc"]),
      bicycle_fraction_ttc_below_th=self.frac(df["bicycle_any_ttc_below_th"]),
      bicycle_max_num_ttc_below_th=float(df["bicycle_num_ttc_below_th"].max()),
      bicycle_min_dca_min=float(df["bicycle_min_dca"].min()),
      bicycle_min_dca_p10=self.p10(df["bicycle_min_dca"]),
      bicycle_fraction_dca_below_th=self.frac(df["bicycle_min_dca"] < dca_th),
      bicycle_ttc_exposure_th=self.exposure(df["bicycle_min_ttc"], self.risk_ttc_thresh),
    )

class InfrastructureFeatureExtractor(FeatureExtractor[InfrastructureFeatures]):
  def __init__(self):
    self.signals = odr.get_signals()

  def prepare(self, df, maneuver: Maneuver) -> pd.DataFrame:
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
