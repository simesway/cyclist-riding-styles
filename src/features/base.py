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
  # Counts
  car_count_mean: float
  car_count_max: float
  pedestrian_count_mean: float
  pedestrian_count_max: float
  bicycle_count_mean: float
  bicycle_count_max: float

  # Car
  car_min_ttc_min: float # worst-case severity
  car_min_ttc_p10: float # robust tail severity

  car_fraction_ttc_below_1_5s: float # frequency of risky states
  car_max_num_ttc_below_1_5s: float  # peak interaction pressure
  car_fraction_ttc_below_2s: float # frequency of risky states
  car_max_num_ttc_below_2s: float  # peak interaction pressure
  car_fraction_ttc_below_3s: float # frequency of risky states
  car_max_num_ttc_below_3s: float  # peak interaction pressure
  car_ttc_exposure_1_5s: float
  car_ttc_exposure_2s: float
  car_ttc_exposure_3s: float

  car_max_drac: float  # maximum deceleration rate to avoid crash
  car_max_drac_p90: float  # robust deceleration tail

  car_min_dca_min: float  # closest spatial interaction
  car_min_dca_p10: float  # robust spatial tail
  car_fraction_dca_below_th: float # sustained proximity

  # Pedestrian
  pedestrian_min_ttc_min: float
  pedestrian_min_ttc_p10: float

  pedestrian_fraction_ttc_below_1_5s: float
  pedestrian_max_num_ttc_below_1_5s: float
  pedestrian_fraction_ttc_below_2s: float
  pedestrian_max_num_ttc_below_2s: float
  pedestrian_fraction_ttc_below_3s: float
  pedestrian_max_num_ttc_below_3s: float
  pedestrian_ttc_exposure_1_5s: float
  pedestrian_ttc_exposure_2s: float
  pedestrian_ttc_exposure_3s: float

  pedestrian_max_drac: float
  pedestrian_max_drac_p90: float

  pedestrian_min_dca_min: float
  pedestrian_min_dca_p10: float
  pedestrian_fraction_dca_below_th: float

  # Bicycle
  bicycle_min_ttc_min: float
  bicycle_min_ttc_p10: float

  bicycle_fraction_ttc_below_1_5s: float
  bicycle_max_num_ttc_below_1_5s: float
  bicycle_fraction_ttc_below_2s: float
  bicycle_max_num_ttc_below_2s: float
  bicycle_fraction_ttc_below_3s: float
  bicycle_max_num_ttc_below_3s: float
  bicycle_ttc_exposure_1_5s: float
  bicycle_ttc_exposure_2s: float
  bicycle_ttc_exposure_3s: float

  bicycle_max_drac: float
  bicycle_max_drac_p90: float

  bicycle_min_dca_min: float
  bicycle_min_dca_p10: float
  bicycle_fraction_dca_below_th: float

  def __post_init__(self):
    # ensure all fields are float
    for field in self.__dataclass_fields__:
      value = getattr(self, field)
      setattr(self, field, float(value))



@dataclass
class InfrastructureFeatures:
  # Road type features
  p_on_motorway: float
  p_on_bikelane: float
  p_on_sidewalk: float

  offset_lane_center: float
  rel_offset_lane_center: float
  rel_position_lane: float

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
