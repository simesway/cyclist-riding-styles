import numpy as np

def time_headway(d_long: np.ndarray, v_long: np.ndarray) -> np.ndarray:
  return d_long / (v_long + 1e-6)