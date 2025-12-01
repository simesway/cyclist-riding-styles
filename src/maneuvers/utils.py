import numpy as np
from data.utils import clean_heading


def get_lateral_longitudinal(a, b):
  """
  Compute lateral and longitudinal vectors in a fixed POV (vehicle a).

  a, b: DataFrames with columns ["timestamp", "translation_x", "translation_y", "rotation_z"]

  Returns:
      ts: aligned timestamps
      follower_ids: array of 1 (a) or 2 (b)
      leader_ids: array of 1 (a) or 2 (b)
      lateral: array in a’s frame
      longitudinal: array in a’s frame
  """
  # sync timestamps
  ts, i_a, i_b = np.intersect1d(
    a["timestamp"].to_numpy(),
    b["timestamp"].to_numpy(),
    return_indices=True
  )
  if len(ts) == 0:
    return None, None, None, None, None

  # aligned positions
  ax = a["translation_x"].to_numpy()[i_a]
  ay = a["translation_y"].to_numpy()[i_a]
  bx = b["translation_x"].to_numpy()[i_b]
  by = b["translation_y"].to_numpy()[i_b]

  # headings
  ha = clean_heading(a["rotation_z"].to_numpy()[i_a])
  hb = clean_heading(b["rotation_z"].to_numpy()[i_b])

  # vector from a → b
  dx = bx - ax
  dy = by - ay

  # a’s heading
  fx = np.cos(ha)
  fy = np.sin(ha)
  lx = -fy
  ly = fx

  # projections in a’s frame
  longitudinal = dx * fx + dy * fy
  lateral = dx * lx + dy * ly

  # follower/leader assignment based on longitudinal distance
  follower_ids = np.where(longitudinal < 0, 1, 2)
  leader_ids = np.where(longitudinal < 0, 2, 1)

  return ts, lateral, longitudinal



def detect_sign_flips(arr: np.ndarray) -> np.ndarray:
  """
  Returns indices where arr changes sign (+→- or -→+)
  """
  signs = np.sign(arr)
  signs[signs == 0] = 1  # treat exact zero as positive to avoid false flips
  flips = np.where(np.diff(signs) != 0)[0]
  return flips


def find_positive_to_negative_crossings(longitudinal: np.ndarray) -> np.ndarray:
    """
    Returns indices where longitudinal flips from positive to negative.
    """
    signs = np.sign(longitudinal)
    # consider only +1 → -1 transitions
    crossings = np.where((signs[:-1] > 0) & (signs[1:] < 0))[0] + 1
    return crossings