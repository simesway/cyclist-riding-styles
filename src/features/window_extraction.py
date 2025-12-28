from typing import List, Tuple
import pandas as pd

from data.utils import apply_time_window
from features.FeatureExtractor import TrafficFeatureExtractor, InfrastructureFeatureExtractor, RidingFeatureExtractor


class SlidingWindows:
  """Sliding windows over a trajectory segment."""
  def __init__(self, win_s: float, overlap: float):
    assert 0 <= overlap < 1, "overlap must be in [0, 1)"
    self.win_s = win_s
    self.overlap = overlap
    self.step = win_s * (1 - overlap)

  def compute_window_bounds(self, t_start: float, t_end: float) -> List[Tuple[float, float]]:
    """Compute sliding window bounds for a given time range."""
    bounds = []
    t = t_start
    while t + self.win_s <= t_end + 1e-8:
      bounds.append((t, t + self.win_s))
      t += self.step

    if bounds and bounds[-1][1] < t_end:
      bounds.append((t_end - self.win_s, t_end))

    return bounds

  def extract(self, df) -> List[Tuple[float, float, pd.DataFrame]]:
    """Extract windows from the DataFrame."""
    t_start = df['timestamp'].min()
    t_end = df['timestamp'].max()

    window_bounds = self.compute_window_bounds(t_start, t_end)

    windows = []
    for t0, t1 in window_bounds:
      window_df = apply_time_window(df, t0, t1)
      windows.append((t0, t1, window_df))

    return windows


class WindowBuilder:
  """Builds WindowRecord objects for a maneuver"""
  def __init__(
    self,
    window_extractor: SlidingWindows,
    riding_extractor: RidingFeatureExtractor,
    traffic_extractor: TrafficFeatureExtractor,
    infra_extractor: InfrastructureFeatureExtractor
  ):
    self.window_extractor = window_extractor
    self.riding_extractor = riding_extractor
    self.traffic_extractor = traffic_extractor
    self.infra_extractor = infra_extractor