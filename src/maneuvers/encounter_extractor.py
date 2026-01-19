import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Optional

from shapely.strtree import STRtree
from shapely.geometry import LineString

from data.utils import apply_time_window


class EncounterExtractor:
  """
  Class to extract encounters between trajectories using spatial indexing.
  Builds an STR-Tree for efficient spatial queries on simplified and segmented trajectories.
  Builds global and temporal filtered Index for one target track at a time.
  """
  def __init__(self, df: pd.DataFrame):
    self.df = df.copy()

    self.t_start = None
    self.t_end = None

    self.df_subset: Optional[pd.DataFrame] = None
    self.tree: Optional[STRtree] = None
    self.segments: List[LineString] = []
    self.segment_lookup: Dict[LineString, int] = {}
    self.traj_dict: Dict[int, LineString] = {}
    
    self.x_col = "translation_x"
    self.y_col = "translation_y"


  @staticmethod
  def simplify_and_segment(traj: LineString, simplify_tol=0.5) -> List[LineString]:
    """Method to simplify and segment a Trajectory. (Zheng 2015)"""
    # 1. simplify using Douglas-Peucker Algorithm
    simplified = traj.simplify(simplify_tol, preserve_topology=False)
    # 2. split into segments
    pts = list(simplified.coords)
    segments = [LineString([pts[i], pts[i + 1]]) for i in range(len(pts) - 1)]
    return segments

  @staticmethod
  def traj_list_to_linestrings(traj_df: pd.DataFrame) -> Dict[int, LineString]:
    """Convert multiple trajectory to a dict: {track_id: LineString}."""
    result = {}
    for track_id, traj in traj_df.groupby("track_id"):
      points = traj[["translation_x", "translation_y"]].to_numpy()
      if len(points) > 1:
        result[track_id] = LineString(points)
    return result

  def build_str_tree(self, traj_dict: Dict[int, LineString], exclude_ids: List[int]) -> Tuple[STRtree, List[LineString], Dict[LineString, int]]:
    """Build the STR-Tree for the given Trajectories."""
    segments = []
    segment_lookup = {}
    for traj_id, traj in traj_dict.items():
      if traj_id in exclude_ids:
        continue
      segs = self.simplify_and_segment(traj, .1)
      segments.extend(segs)
      for seg in segs:
        segment_lookup[seg] = traj_id

    tree = STRtree(segments)
    return tree, segments, segment_lookup


  def _build_temporal_index(self, t_start, t_end):
    """Build an Index using an STR-Tree for a given time window."""
    if self.t_start is not None and self.t_start <= t_start and self.t_end >= t_end:
      return

    self.t_start, self.t_end = t_start, t_end

    self.df_subset = apply_time_window(self.df, t_start, t_end)
    self.traj_dict = self.traj_list_to_linestrings(self.df_subset)

    self.tree, self.segments, self.segment_lookup = self.build_str_tree(self.traj_dict, [])

  def query_candidates(self, target_id: int, distance: float) -> List[int]:
    """Queries the STR-Tree to gain intersecting trajectory segments and candidates for interactions."""
    target_traj = self.traj_dict[target_id]
    if target_traj is None:
      return []

    tube = target_traj.buffer(distance)
    seg_ids = self.tree.query(tube)
    candidates = set(self.segment_lookup[self.segments[seg_id]] for seg_id in seg_ids)
    return [c for c in candidates if c != target_id]

  def get_distances(self, target_id: int, candidates: List[int]) -> pd.DataFrame:
    """
    Compute distances between target and candidate trajectories.
    Returns a DataFrame with relative positions, velocities, and distances.
    """
    target_df = self.df_subset[self.df_subset["track_id"] == target_id]
    cand_df = self.df_subset[self.df_subset["track_id"].isin(candidates)]

    merged = cand_df.merge(
      target_df[["timestamp", self.x_col, self.y_col, "velocity_x", "velocity_y"]],
      on="timestamp",
      suffixes=("", "_target")
    )

    merged["dx"] = merged[self.x_col] - merged[self.x_col+"_target"]
    merged["dy"] = merged[self.y_col] - merged[self.y_col+"_target"]

    merged["vx_rel"] = merged["velocity_x"] - merged["velocity_x_target"]
    merged["vy_rel"] = merged["velocity_y"] - merged["velocity_y_target"]

    merged["distance"] = np.sqrt(merged["dx"] ** 2 + merged["dy"] ** 2)

    merged.drop(columns=[self.x_col+"_target", self.y_col+"_target"], inplace=True)
    
    return merged


  def get_encounters(self, target_id: int, distance: float, t_start: float, t_end: float) -> pd.DataFrame:
    """Get encounter candidates for a given track within a time window."""
    self._build_temporal_index(t_start, t_end)
    candidates = self.query_candidates(target_id, distance)
    encounters = self.get_distances(target_id, candidates)
    return encounters