import numpy as np

from math import log2
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter

from maneuvers.base import Maneuver, WindowRecord

ScenarioType = str
RegimeType = Tuple[ScenarioType, int]


@dataclass
class ScenarioStats:
  scenario: ScenarioType
  exposure: float
  entropy: float

@dataclass
class RegimeStats:
  regime_type: RegimeType
  proportion: float
  mean_run_length: float
  std_run_length: float
  com_global: Optional[float]
  com_scenario: Optional[float]


@dataclass
class RegimeAggregation:
  maneuver_id: int
  n_windows: int

  global_entropy: float

  scenario_stats: Dict[ScenarioType, ScenarioStats]
  regime_stats: Dict[RegimeType, RegimeStats]



class RegimeAggregator:
  def aggregate(
      self,
      maneuver: Maneuver,
      windows: List[WindowRecord],
      attach: bool=True
  ) -> RegimeAggregation:
    maneuver_windows = [w for w in windows if w.meta.maneuver_id == maneuver.id and w.local_regime is not None]

    sequence: List[RegimeType] = self._build_sequence(maneuver_windows)

    if len(sequence) == 0:
      raise ValueError("No regimes assigned to maneuver windows")

    scenario_stats = self._compute_scenario_stats(sequence)
    regime_stats = self._compute_regime_stats(sequence)
    global_entropy = self._compute_global_entropy(sequence)

    aggregation = RegimeAggregation(
      maneuver_id=int(maneuver.id),
      n_windows=len(sequence),
      global_entropy=global_entropy,
      scenario_stats=scenario_stats,
      regime_stats=regime_stats
    )

    if attach:
      maneuver.regime_aggregation = aggregation

    return aggregation

  @staticmethod
  def _build_sequence(windows: List[WindowRecord]) -> List[RegimeType]:
    """Build a time-ordered sequence of (scenario, regime) pairs from the maneuver windows."""
    windows.sort(key=lambda w: w.t_start)
    sequence: List[RegimeType] = [
      (w.scenario, w.local_regime)
      for w in windows
    ]
    return sequence

  @staticmethod
  def _compute_scenario_stats(sequence: List[RegimeType]) -> Dict[ScenarioType, ScenarioStats]:
    n_total = len(sequence)

    scenario_counts = defaultdict(int)
    regime_counts_by_scenario = defaultdict(lambda: defaultdict(int))

    for s, r in sequence:
      scenario_counts[s] += 1
      regime_counts_by_scenario[s][r] += 1

    scenario_stats: Dict[ScenarioType, ScenarioStats] = {}

    for s, count_s in scenario_counts.items():
      exposure = count_s / n_total

      regime_counts = regime_counts_by_scenario[s]
      n_regimes = len(regime_counts)

      if n_regimes <= 1:
        entropy = 0.0
      else:
        entropy = -sum(
          (c / count_s) * log2(c / count_s)
          for c in regime_counts.values()
        ) / log2(n_regimes)

      scenario_stats[s] = ScenarioStats(
        scenario=s,
        exposure=exposure,
        entropy=entropy
      )
    return scenario_stats

  @staticmethod
  def _compute_regime_stats(sequence: List[RegimeType]) -> Dict[RegimeType, RegimeStats]:
    n_total = len(sequence)

    counts = defaultdict(int)
    positions_global = defaultdict(list)
    positions_by_scenario = defaultdict(lambda: defaultdict(list)) # for COM scenario
    run_lengths = defaultdict(list)

    # --- Compute runs and positions ---
    if not sequence:
      return {}

    current = sequence[0]
    run_length = 1
    counts[current] += 1
    positions_global[current].append(0)
    positions_by_scenario[current[0]][current].append(0)

    for t, r in enumerate(sequence[1:], start=1):
      counts[r] += 1
      positions_global[r].append(t)
      positions_by_scenario[r[0]][r].append(
        sum(1 for i, x in enumerate(sequence[:t + 1]) if x[0] == r[0]) - 1
      )

      if r == current:
        run_length += 1
      else:
        run_lengths[current].append(run_length)
        current = r
        run_length = 1
    run_lengths[current].append(run_length)  # last run

    # --- Compute stats ---
    regime_stats: Dict[RegimeType, RegimeStats] = {}

    for r, count_r in counts.items():
      proportion = count_r / n_total

      # run length stats
      lengths = np.array(run_lengths[r])
      mean_run = lengths.mean() if len(lengths) > 0 else 0.0
      std_run = lengths.std() if len(lengths) > 1 else 0.0

      # COM global
      com_global = float(np.mean([pos/n_total for pos in positions_global[r]])) if positions_global[r] else None

      # COM scenario
      s = r[0]
      pos_scenario = positions_by_scenario[s][r]
      T_s = len([x for x in sequence if x[0] == s])
      com_scenario = float(np.mean([p / T_s for p in pos_scenario])) if pos_scenario else None

      regime_stats[r] = RegimeStats(
        regime_type=r,
        proportion=proportion,
        mean_run_length=mean_run,
        std_run_length=std_run,
        com_global=com_global,
        com_scenario=com_scenario
      )

    return regime_stats

  @staticmethod
  def _compute_global_entropy(sequence: List[RegimeType]) -> float:
    n_total = len(sequence)
    if n_total == 0:
      return 0.0

    counts = Counter(sequence)
    n_regimes = len(counts)
    if n_regimes <= 1:
      return 0.0  # single regime → entropy 0

    probs = [c / n_total for c in counts.values()]
    entropy = -sum(p * log2(p) for p in probs) / log2(n_regimes)
    return entropy