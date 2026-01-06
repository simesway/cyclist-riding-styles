import numpy as np
import pandas as pd
from typing import List, Dict, Any, Type
from data.utils import clean_heading
from dataclasses import asdict, fields
from typing import List




def save_maneuvers_to_csv(maneuvers, filepath):
  df = pd.DataFrame([m.__dict__ for m in maneuvers])
  float_cols = df.select_dtypes(include="float").columns
  df[float_cols] = df[float_cols].round(3)
  df = df.drop_duplicates()
  df.to_csv(filepath, index=False)


def flatten_optional(obj, prefix: str, cls=None) -> dict:
  if obj is not None:
    return {f"{prefix}_{k}": v for k, v in asdict(obj).items()}
  if cls is None:
    return {}
  return {f"{prefix}_{k}": None for k in cls.__dataclass_fields__}


def unflatten_optional(row: Dict[str, Any], prefix: str, cls: Type):
    data = {f.name: row.get(f"{prefix}_{f.name}") for f in fields(cls)}
    if all(v is None for v in data.values()):
        return None
    return cls(**data)


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
    return None, None, None

  # aligned positions
  ax = a["translation_x"].to_numpy()[i_a]
  ay = a["translation_y"].to_numpy()[i_a]
  bx = b["translation_x"].to_numpy()[i_b]
  by = b["translation_y"].to_numpy()[i_b]

  # headings
  ha = clean_heading(a["rotation_z"].to_numpy()[i_a])
  #hb = clean_heading(b["rotation_z"].to_numpy()[i_b])

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
  #follower_ids = np.where(longitudinal < 0, 1, 2)
  #leader_ids = np.where(longitudinal < 0, 2, 1)

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


def filter_spikes(longitudinal, crossings, k=3):
    """Remove spikes: require sign to persist for k frames."""
    cleaned = []
    L = len(longitudinal)
    for c in crossings:
        if c + 1 >= L:
            continue
        new_sign = np.sign(longitudinal[c + 1])
        end = min(c + 1 + k, L)
        future = np.sign(longitudinal[c + 1:end])
        if np.all(future == new_sign):
            cleaned.append(c)
    return np.array(cleaned)


def merge_crossings(crossings, delta=12):
    """Merge nearby crossings into one group."""
    if len(crossings) == 0:
        return []
    groups = [[crossings[0]]]
    for c in crossings[1:]:
        if c - groups[-1][-1] <= delta:
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def pick_switch_idx(group, longitudinal, k=5):
  best, best_len = group[0], -1
  L = len(longitudinal)

  for c in group:
    end = min(c + k, L)
    run = np.sign(longitudinal[c + 1:end])
    score = np.sum(run < 0)
    if score > best_len:
      best, best_len = c, score

  return best


def extract_overtake_windows(
    ts, long_a, long_b,
    delta=15, k=25, max_frames=63
) -> List[dict]:
  L = len(long_a)
  if L < 2:
    return []

  raw = np.unique(np.concatenate([
    filter_spikes(long_a, find_positive_to_negative_crossings(long_a), k),
    filter_spikes(long_b, find_positive_to_negative_crossings(long_b), k),
  ]))
  if len(raw) == 0:
    return []

  groups = merge_crossings(raw, delta)
  windows = []

  for gi, g in enumerate(groups):
    switch_idx = g[0]
    switch_idx = int(switch_idx)

    prev_cross = groups[gi - 1][-1] if gi > 0 else 0
    next_cross = groups[gi + 1][0] if gi < len(groups) - 1 else L - 1

    prev_cross = int(prev_cross)
    next_cross = int(next_cross)

    # decide POV at switch
    if long_a[switch_idx - 1] > 0:
      long_ref = long_a
      follower, leader = "A", "B"
    else:
      long_ref = long_b
      follower, leader = "B", "A"

    # extrema-based bounds
    start_idx = prev_cross + int(np.argmax(np.abs(long_ref[prev_cross:switch_idx + 1])))
    end_idx = switch_idx + int(np.argmax(np.abs(long_ref[switch_idx:next_cross + 1])))

    # clamp to max_frames
    half = max_frames // 2
    start_idx = max(start_idx, switch_idx - half, 0)
    end_idx = min(end_idx, switch_idx + half, L - 1)

    windows.append({
      "center": switch_idx,
      "start": int(start_idx),
      "end": int(end_idx),
      "ts": ts[start_idx:end_idx + 1],
      "longitudinal": long_ref[start_idx:end_idx + 1],
      "follower": follower,
      "leader": leader,
    })

  return windows

