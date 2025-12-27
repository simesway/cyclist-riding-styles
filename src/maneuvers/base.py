from dataclasses import dataclass, asdict
from typing import Optional

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
class FollowingManeuver(Maneuver):
  other_id: int

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
class OvertakingManeuver(Maneuver):
  other_id: int
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


@dataclass
class RidingFeatures:
  speed_mean: float


@dataclass
class EnvironmentFeatures:
  thw_mean: float


@dataclass
class InfrastructureFeatures:
  lane_width: float


@dataclass
class WindowRecord:
  maneuver_id: int
  t_start: float
  t_end: float
  features: Optional[RidingFeatures] = None
  environment: Optional[EnvironmentFeatures] = None
  infrastructure: Optional[InfrastructureFeatures] = None

  def flatten(self):
    out = {
      "maneuver_id": self.maneuver_id,
      "t_start": self.t_start,
      "t_end": self.t_end,
    }
    for attr_name in ["features", "environment", "infrastructure"]:
      attr = getattr(self, attr_name)
      if attr is not None:
        out.update(asdict(attr))
      else:
        # fill missing keys with None
        klass = self.__annotations__[attr_name].__args__[0]  # get dataclass type
        for f in klass.__dataclass_fields__:
          out[f] = None
    return out

