import numpy as np
from typing import List, Dict
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


def pick_switch_idx(group, longitudinal):
    """Pick the max |longitudinal| frame inside the group as switch."""
    start = group[0]
    end = group[-1] + 1
    local = longitudinal[start:end]
    return start + int(np.argmin(np.abs(local)))


def extract_overtake_windows(ts, long_a, long_b, delta=15, k=25, max_frames=63) -> List[Dict]:
    """
    Extract overtaking windows from trajectories in the POV of the initial follower.
    Each window spans from max |longitudinal| after previous crossing
    to max |longitudinal| before next crossing.
    """
    L = len(long_a)
    if L < 2:
        return []

    # 1. raw zero crossings from both POVs
    raw_a = find_positive_to_negative_crossings(long_a)
    raw_b = find_positive_to_negative_crossings(long_b)
    if len(raw_a) + len(raw_b) == 0:
        return []

    # 2. remove spikes
    crossings_a = filter_spikes(long_a, raw_a, k)
    crossings_b = filter_spikes(long_b, raw_b, k)
    crossings = np.unique(np.concatenate([raw_a, raw_b]))
    if len(crossings) == 0:
        return []

    # 3. merge nearby crossings into groups for switch index
    groups = merge_crossings(crossings, delta)

    windows = []

    last_switch = 0

    for g in groups:
      # pick switch index using long_a (just as a reference)
      switch_idx = pick_switch_idx(g, long_a)

      # determine interval using neighboring crossings
      idx_in_crossings = np.searchsorted(crossings, switch_idx)
      prev_cross = last_switch
      next_cross = L - 1

      # 1. determine initial follower for this window
      # check A's POV first frame of window candidate
      if long_a[switch_idx-1] > 0:
        long_ref = long_a
        follower, leader = "A", "B"
      else:
        long_ref = long_b
        follower, leader = "B", "A"

      # 2. compute start and end based on max |longitudinal| in this POV
      local_start = long_ref[prev_cross:switch_idx + 1]
      start_idx = prev_cross + int(np.argmax(np.abs(local_start)))

      local_end = long_ref[switch_idx:next_cross + 1]
      end_idx = switch_idx + int(np.argmax(np.abs(local_end)))

      # 3. limit window to max_frames
      half_max = max_frames // 2
      start_idx = max(0, switch_idx - half_max, start_idx)
      end_idx = min(L - 1, switch_idx + half_max, end_idx)

      # extract longitudinal for this POV
      long_win = long_ref[start_idx:end_idx + 1]
      ts_win = ts[start_idx:end_idx + 1]

      windows.append({
        "center": switch_idx,
        "start": start_idx,
        "end": end_idx,
        "ts": ts_win,
        "longitudinal": long_win,
        "follower": follower,
        "leader": leader
      })

    return windows
