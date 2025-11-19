from shapely.geometry import LineString
from pandas import DataFrame
from typing import Dict

def traj_to_linestring(traj_df: DataFrame, x_col="translation_x", y_col="translation_y") -> LineString:
  """Convert a single trajectory dataframe to a LineString."""
  return LineString(traj_df[[x_col, y_col]].to_numpy())

def traj_list_to_linestrings(traj_df: DataFrame, id_col="track_id", x_col="translation_x", y_col="translation_y") -> Dict[int, LineString]:
  """Convert multiple trajectory to a dict: {track_id: LineString}."""
  result = {}
  for track_id, traj in traj_df.groupby(id_col):
    points = traj[[x_col, y_col]].to_numpy()
    if len(points) > 1:
      result[track_id] = LineString(points)
  return result