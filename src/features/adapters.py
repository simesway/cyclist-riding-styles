from dataclasses import fields as dataclass_fields
from typing import List, Callable, Any, Iterable, Union, Optional, Dict

import numpy as np
import pandas as pd

from features.base import RegimeAggregation, OvertakingFeatures, FollowingFeatures
from maneuvers.base import ScenarioType, RegimeStats


class FeatureAdapter:
  def __init__(self, feature_class, exclude: List[str] = None):
    self.feature_class = feature_class
    self._names = [f.name for f in dataclass_fields(feature_class)]
    self.exclude = set(exclude or [])
    self._active_names = [n for n in self._names if n not in self.exclude]

  def to_vector(self, f) -> List[float]:
    return [getattr(f, name) for name in self._active_names]

  def from_vector(self, v: List[float]):
    if len(v) != len(self._active_names):
      raise ValueError("Vector length mismatch")
    return self.feature_class(**dict(zip(self._active_names, v)))

  @property
  def magnitude_idx(self) -> List[int]:
    return [
      i for i, n in enumerate(self._names)
      if any(k in n for k in ("min", "max", "mean"))
         and not any(k in n for k in ("std", "mad", "qcv", "cv", "var"))
         and n in self._active_names
    ]

  @property
  def volatility_idx(self) -> List[int]:
    return [
      i for i, n in enumerate(self._names)
      if any(k in n for k in ("std", "mad", "qcv", "cv", "var")) and n in self._active_names
    ]

  @property
  def feature_names(self) -> List[str]:
    return self._active_names



class RegimeAggregationAdapter:
  def __init__(
      self,
      scenario_config: Dict[ScenarioType, int],
      exclude: List[str] = None,
  ):
    self.scenario_config = scenario_config
    self.exclude = set(exclude or [])

    self._feature_names = self._build_feature_names()

  def to_vector(self, agg: RegimeAggregation) -> List[float]:
    vec = []

    # --- global entropy ---
    if "global_entropy" not in self.exclude:
      vec.append(agg.global_entropy)

    for scenario, n_regimes in self.scenario_config.items():

      s_stats = agg.scenario_stats.get(scenario)

      # --- scenario level ---
      if "scenario_entropy" not in self.exclude:
        vec.append(s_stats.entropy if s_stats else 0.0)

      if "scenario_exposure" not in self.exclude:
        vec.append(s_stats.exposure if s_stats else 0.0)

      # --- regime level ---
      for k in range(n_regimes):
        r_stats = agg.regime_stats.get((scenario, k))

        vec.extend(self._regime_values(r_stats))

    return vec

  @property
  def feature_names(self) -> List[str]:
    return self._feature_names

  def _regime_values(self, r: Optional[RegimeStats]) -> List[float]:
    values = []

    def v(name, val):
      if name not in self.exclude:
        values.append(val)

    if r:
      v("proportion", r.proportion)
      v("mean_run_length", r.mean_run_length)
      v("std_run_length", r.std_run_length)
      v("com_global", r.com_global or 0.0)
      v("com_scenario", r.com_scenario or 0.0)
    else:
      # fill with zeros
      for name in (
          "proportion",
          "mean_run_length",
          "std_run_length",
          "com_global",
          "com_scenario",
      ):
        if name not in self.exclude:
          values.append(0.0)

    return values

  def _build_feature_names(self) -> List[str]:
    names = []

    if "global_entropy" not in self.exclude:
      names.append("global_entropy")

    for scenario, n_regimes in self.scenario_config.items():

      if "scenario_entropy" not in self.exclude:
        names.append(f"{scenario}_entropy")

      if "scenario_exposure" not in self.exclude:
        names.append(f"{scenario}_exposure")

      for k in range(n_regimes):
        for suffix in (
            "proportion",
            "mean_run_length",
            "std_run_length",
            "com_global",
            "com_scenario",
        ):
          if suffix not in self.exclude:
            names.append(f"{scenario}_r{k}_{suffix}")

    return names



class ManeuverAdapter:
  def __init__(
      self,
      feature_cls: Union[OvertakingFeatures, FollowingFeatures],
      scenario_config: Dict[ScenarioType, int],
      exclude: List[str] = None,
      include_meta: bool = False
  ):
    self.exclude = set(exclude or [])
    self.include_meta = include_meta
    self.scenario_config = scenario_config
    self.feature_adapter = FeatureAdapter(
      feature_cls,
      exclude=list(self.exclude)
    )

    self.regime_adapter = RegimeAggregationAdapter(
      scenario_config=self.scenario_config,
      exclude=list(self.exclude)
    )

  def _meta_values(self, m) -> List[float]:
    return [m.id, m.ego_id, m.t_start, m.t_end, m.duration, m.cluster_id]


  @property
  def meta_feature_names(self) -> List[str]:
    return ["id", "ego_id", "t_start", "t_end", "duration", "cluster_id"]

  def to_vector(self, m) -> List[float]:
    v = []
    if self.include_meta:
      v.extend(self._meta_values(m))
    v.extend(self.feature_adapter.to_vector(m.features))
    v.extend(self.regime_adapter.to_vector(m.regime_aggregation))
    return v

  @property
  def feature_names(self) -> List[str]:
    names = []
    if self.include_meta:
      names.extend(self.meta_feature_names)
    names.extend(self.feature_adapter.feature_names)
    names.extend(self.regime_adapter.feature_names)
    return names


class FeatureMatrixBuilder:
    def __init__(self, adapter, getter: Callable[[Any], Any], drop_none: bool = True):
        self.adapter = adapter
        self.getter = getter
        self.drop_none = drop_none

    def _collect(self, items: Iterable[Any]):
        X = []
        valid_items = []

        for item in items:
            obj = self.getter(item)
            if obj is None and self.drop_none:
                continue
            X.append(self.adapter.to_vector(obj))
            valid_items.append(item)

        return X, valid_items

    def to_numpy(self, items: Iterable[Any]):
        X, valid_items = self._collect(items)
        return np.asarray(X), valid_items

    def to_dataframe(self, items: Iterable[Any]):
        X, valid_items = self._collect(items)
        df = pd.DataFrame(X, columns=self.adapter.feature_names)
        return df, valid_items
