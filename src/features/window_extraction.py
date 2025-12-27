from typing import List, Tuple

from maneuvers.base import RidingFeatures, InfrastructureFeatures, TrafficFeatures, Maneuver


def compute_window_bounds(
  t_start: float,
  t_end: float,
  win_length_s: float,
  overlap: float
) -> List[Tuple[float, float]]:
  assert 0 <= overlap < 1, "overlap must be in [0, 1)"
  step = win_length_s * (1 - overlap)
  bounds = []

  t = t_start
  while t + win_length_s <= t_end + 1e-8:  # small epsilon for floating point
    bounds.append((t, t + win_length_s))
    t += step

  # Optionally include a last partial window if t_end not exactly reached
  if bounds and bounds[-1][1] < t_end:
    bounds.append((t_end - win_length_s, t_end))

  return bounds

def extract_riding_features(window_df) -> RidingFeatures:
  """Extracts local riding behavior features"""
  pass

def infer_infrastructure_features(window_df) -> InfrastructureFeatures:
  """Derives infrastructure features for the window"""
  pass


