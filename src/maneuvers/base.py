from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional, Literal, List

from data.utils import apply_time_window
from features.base import RidingFeatures, TrafficFeatures, InfrastructureFeatures
from maneuvers.utils import flatten_optional


ManeuverType = Literal["interaction", "following", "overtaking"]

@dataclass(eq=False)
class Maneuver(ABC):
  id: int | None
  ego_id: int
  t_start: float
  t_end: float
  duration: float

  @property
  @abstractmethod
  def maneuver_type(self) -> ManeuverType:
    ...

  def flatten(self):
    return asdict(self)

  def identity(self):
    # base identity (subclasses can extend if needed)
    return self.ego_id, self.t_start, self.t_end

  def __eq__(self, other):
    if not isinstance(other, Maneuver):
      return NotImplemented
    return self.identity() == other.identity()

  def __hash__(self):
    return hash(self.identity())


@dataclass(eq=False)
class Interaction(Maneuver):
  other_id: int

  @property
  def maneuver_type(self) -> Literal["interaction"]:
    return "interaction"

  def identity(self):
    return self.ego_id, self.other_id, self.t_start, self.t_end


@dataclass(eq=False)
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

  @property
  def maneuver_type(self) -> Literal["following"]:
    return "following"


@dataclass(eq=False)
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

  @property
  def maneuver_type(self) -> Literal["overtaking"]:
    return "overtaking"


@dataclass(frozen=True)
class ManeuverMeta:
  maneuver_id: int
  maneuver_type: ManeuverType
  ego_id: int


@dataclass
class WindowRecord:
  meta: ManeuverMeta
  t_start: float
  t_end: float
  riding: Optional[RidingFeatures] = None
  traffic: Optional[TrafficFeatures] = None
  infrastructure: Optional[InfrastructureFeatures] = None
  local_regime: Optional[int] = None

  def flatten(self):
    return {
      **asdict(self.meta),
      "t_start": self.t_start,
      "t_end": self.t_end,
      **flatten_optional(self.riding, "ride", RidingFeatures),
      **flatten_optional(self.traffic, "env", TrafficFeatures),
      **flatten_optional(self.infrastructure, "infra", InfrastructureFeatures),
    }


def group_windows_by_maneuver(
    windows: List[WindowRecord]
) -> dict[int, List[WindowRecord]]:
    """Groups windows by their maneuver ID."""
    grouped = defaultdict(list)
    for w in windows:
        mid = w.meta.maneuver_id
        grouped[mid].append(w)
    return dict(grouped)


class ManeuverSlicer:
  """Extracts ego (and other) trajectory for a given maneuver."""
  @staticmethod
  def slice(traj_df, maneuver: Maneuver):
    """Returns ego (and other) trajectory df between t_start and t_end."""
    ids = [maneuver.ego_id]
    if isinstance(maneuver, Interaction):
      ids.append(maneuver.other_id)
    ego_traj = traj_df[traj_df["track_id"].isin(ids)]
    ego_traj = ego_traj.sort_values("timestamp")
    return apply_time_window(ego_traj, maneuver.t_start, maneuver.t_end)