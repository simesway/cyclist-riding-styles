import numpy as np

from typing import List
from collections import Counter
from dataclasses import dataclass

from maneuvers.base import WindowRecord


@dataclass
class RegimeAggregation:
  maneuver_id: int
  n_windows: int
  regime_proportions: np.ndarray
  transition_rate: float


class LocalRegimeAggregator:
  def __init__(self, n_regimes: int):
    self.n_regimes = n_regimes

  def aggregate(self, maneuver_id: int, windows: List[WindowRecord]) -> RegimeAggregation:
    regimes = np.array(
      [w.local_regime for w in windows if w.local_regime is not None]
    )

    if len(regimes) == 0:
      raise ValueError("No regimes assigned to maneuver windows")

    counts = Counter(regimes)
    proportions = np.zeros(self.n_regimes)
    for k,v in counts.items():
      proportions[k] = v / len(regimes)

    transitions = np.sum(regimes[1:] != regimes[:-1])
    transition_rate = transitions / max(len(regimes) - 1, 1)

    return RegimeAggregation(maneuver_id, len(windows), proportions, transition_rate)