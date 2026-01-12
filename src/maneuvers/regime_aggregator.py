import numpy as np

from typing import List

from clustering.semantics import RegimeClusterMapper
from features.base import RegimeAggregation
from maneuvers.base import WindowRecord, Maneuver


class LocalRegimeAggregator:
  def __init__(self, n_regimes: int):
    self.n_regimes = n_regimes

  def aggregate(
      self,
      maneuver: Maneuver,
      windows: List[WindowRecord],
      regime_mapper: RegimeClusterMapper,
      attach=True
  ) -> RegimeAggregation:
    windows = [w for w in windows if w.meta.maneuver_id == maneuver.id]
    w_sorted = sorted(windows, key=lambda w: w.t_start)
    regimes = [w.local_regime for w in w_sorted if w.local_regime is not None]

    is_stable = regime_mapper.is_stable(regimes, as_numpy=True)

    N = len(regimes)
    if N == 0:
      raise ValueError("No regimes assigned to maneuver windows")

    volatile = ~is_stable # e.g. volatile = 0 1 1 0 1

    p_volatile = volatile.mean()

    # general transition rate (stable -> volatile & volatile -> stable)
    transitions = is_stable[:-1] != is_stable[1:]
    transition_rate = transitions.sum() / max(N-1, 1)

    # volatile-based transition rate (stable -> volatile)
    #volatile_onsets = (is_stable[:-1] == True) & (is_stable[1:] == False)
    #transition_rate = volatile_onsets.sum() / max(N - 1, 1)

    padded = np.concatenate(([0], volatile.view(np.int8), [0])) # padded = 0 0 1 1 0 1 0
    diff = np.diff(padded) # diff:   0 1 0 -1 1 -1
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    run_lengths = ends - starts

    if len(run_lengths) == 0:
      mean_run_volatile = 0.0
      std_run_volatile = 0.0
      mean_distance_between_volatile = N
    else:
      mean_run_volatile = run_lengths.mean()
      std_run_volatile = run_lengths.std()

      if len(starts) < 2:
        mean_distance_between_volatile = N
      else:
        distances = starts[1:] - ends[:-1]
        mean_distance_between_volatile = distances.mean()


    regime_aggregation = RegimeAggregation(
        maneuver_id=maneuver.id,
        n_windows=N,
        is_active=0.0 < p_volatile < 1.0,
        p_volatile=p_volatile,
        transition_rate=transition_rate,
        mean_run_volatile=mean_run_volatile,
        std_run_volatile=std_run_volatile,
        mean_volatile_gap=mean_distance_between_volatile/N
      )

    if attach:
      maneuver.regime_aggregation = regime_aggregation

    return regime_aggregation

