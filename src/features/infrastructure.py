from typing import List

import numpy as np
import pandas as pd
from config import cfg

import trafficfeatures

trafficfeatures.init(cfg.paths.xodr)

import trafficfeatures.filter as flt
import trafficfeatures.instant as inst
import trafficfeatures.opendrive as odr


def infer_basic_infrastructure_features(df: pd.DataFrame) -> pd.DataFrame:
  """Infers basic infrastructure features and adds them to the DataFrame"""
  df['abs_x'] = inst.abs_pos(df, 'translation_x', cfg.map.offset_x)
  df['abs_y'] = inst.abs_pos(df, 'translation_y', cfg.map.offset_y)

  df['velocity'] = inst.magnitude(df, ['velocity_x', 'velocity_y'])
  df['acceleration'] = inst.magnitude(df, ['acceleration_x', 'acceleration_y'])

  df['rotation_z'] = inst.normalize_angle(df, 'rotation_z')

  # road data
  road_data = odr.get_road(df)
  # remove duplicate road assignments
  road_data = road_data.reset_index().drop_duplicates(subset="index").set_index("index")
  df = df.join(road_data)
  df = odr.fill_single_nan_roads(df)
  df = odr.fill_double_nan_roads(df)
  df['lane_angle'] = odr.lane_angle(df)

  df['heading_deviation'] = inst.heading_deviation(df)

  lane_boundary_data = odr.lane_boundaries_and_width(df)
  df = df.join(lane_boundary_data)

  df = flt.clip_tracks_to_map_bounds(df)
  return df



def on_lane(df, lane_types: List[str]):
  return df["lane_type"].isin(lane_types)

def offset_lane_center(df):
  return df["inner_dist"] - df["lane_width"] / 2


def rel_offset_norm(df):
  return np.abs(df["offset_lane_center"] / df["lane_width"])


def min_lateral_clearance(df):
  return df[['inner_dist', 'outer_dist']].min(axis=1)


def max_lateral_clearance(df):
  return df[['inner_dist', 'outer_dist']].max(axis=1)


def distance_to_signal(df, signal_df, signal_type='TrafficLight', max_radius=np.inf):
  """
  Return Euclidean distance (meters) from each row in df to nearest signal of signal_type.
  - df must have abs_x, abs_y
  - signal_df must have x, y, type
  - max_radius: if provided, distances > max_radius are set to NaN (default: no limit)
  """
  sigs = signal_df[signal_df['name'] == signal_type]
  if sigs.empty:
    return pd.Series(np.nan, index=df.index, name=f'dist_to_{signal_type}')

  sig_x = sigs['x'].to_numpy()
  sig_y = sigs['y'].to_numpy()

  cx = df['abs_x'].to_numpy()
  cy = df['abs_y'].to_numpy()

  out = np.full(len(df), np.nan, dtype=float)

  for i in range(len(df)):
    vx = sig_x - cx[i]
    vy = sig_y - cy[i]
    dists = np.hypot(vx, vy)
    if dists.size == 0:
      out[i] = np.nan
      continue
    min_d = float(dists.min())
    out[i] = min_d if min_d <= max_radius else np.nan

  return pd.Series(out, index=df.index, name=f'dist_to_{signal_type}')