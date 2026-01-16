from dataclasses import dataclass

@dataclass
class RidingFeatures:
  # Magnitude:
  speed_max: float
  speed_min: float
  acc_max: float
  acc_min: float
  # Volatility
  speed_mean: float
  speed_std: float
  speed_mad: float
  speed_qcv: float
  acc_mean: float
  acc_std: float
  acc_mad: float
  acc_qcv: float

  rot_fluc_mad: float
  rot_fluc_std: float
  rot_fluc_max_abs: float


@dataclass
class TrafficFeatures:
  thw_mean: float


@dataclass
class InfrastructureFeatures:
  # Road type features
  on_motorway: bool
  on_bikelane: bool
  on_sidewalk: bool

  offset_lane_center: float
  rel_offset_lane_center: float

  left_clearance: float
  min_lateral_clearance: float
  max_lateral_clearance: float
  lane_width: float

  # other features
  distance_traffic_light: float
  distance_regulatory_sign: float
  distance_warning_sign: float


@dataclass
class OvertakingFeatures:
  left_side: bool
  long_distance_min: float
  long_distance_max: float
  lateral_offset_start: float
  lateral_offset_end: float
  lateral_offset_max: float
  lateral_offset_cross: float
  follower_speed_mean: float
  leader_speed_mean: float
  speed_diff_mean: float
  follower_acc_max: float


@dataclass
class FollowingFeatures:
  long_distance_min: float
  long_distance_mean: float
  lateral_offset_mean: float
  thw_mean: float
  thw_min: float
  follower_speed_mean: float
  leader_speed_mean: float
  speed_diff_mean: float
  rel_heading_std: float


@dataclass
class RegimeAggregation:
  maneuver_id: int
  n_windows: int
  is_active: bool
  p_volatile: float
  transition_rate: float
  mean_run_volatile: float
  std_run_volatile: float
  mean_volatile_gap: float
