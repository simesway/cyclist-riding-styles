from dataclasses import dataclass, asdict
from typing import Optional, Literal

from maneuvers.utils import flatten_optional


@dataclass
class Maneuver:
  id: int | None
  ego_id: int
  t_start: float
  t_end: float
  duration: float

  def flatten(self):
    return asdict(self)


@dataclass
class Interaction(Maneuver):
  other_id: int


@dataclass
class FollowingManeuver(Interaction):
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
class OvertakingManeuver(Interaction):
  t_cross: float

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


@dataclass(frozen=True)
class ManeuverMeta:
  maneuver_id: int
  maneuver_type: Literal["following", "overtaking"]
  ego_id: int


@dataclass
class RidingFeatures:
  speed_mean: float


@dataclass
class TrafficFeatures:
  thw_mean: float


@dataclass
class InfrastructureFeatures:
  lane_width: float


@dataclass
class WindowRecord:
  meta: ManeuverMeta
  t_start: float
  t_end: float
  features: Optional[RidingFeatures] = None
  environment: Optional[TrafficFeatures] = None
  infrastructure: Optional[InfrastructureFeatures] = None

  def flatten(self):
    return {
      **asdict(self.meta),
      "t_start": self.t_start,
      "t_end": self.t_end,
      **flatten_optional(self.features, "ride", RidingFeatures),
      **flatten_optional(self.environment, "env", TrafficFeatures),
      **flatten_optional(self.infrastructure, "infra", InfrastructureFeatures),
    }
