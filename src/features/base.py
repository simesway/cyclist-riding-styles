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


@dataclass
class TrafficFeatures:
  thw_mean: float


@dataclass
class InfrastructureFeatures:
  lane_width: float


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
