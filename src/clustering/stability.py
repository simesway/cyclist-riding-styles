import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Tuple
from dataclasses import dataclass

from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score


@dataclass
class StabilityResult:
    ari_subsample: float | tuple
    ari_seed: float | tuple
    ari_noise: float | tuple
    min_cluster_frac: float | None = None


class RegimeStabilityTester:
  """Tests clustering stability under subsampling, random seed, and noise perturbations."""
  def __init__(
      self,
      clusterer_factory,
      noise_scale: float = 0.05,
      subsample_frac: float = 0.8,
      random_state: int | None = None,
      metric_subsample_frac: float = 0.8,
  ):
    self.clusterer_factory = clusterer_factory
    self.noise_scale = noise_scale
    self.subsample_frac = subsample_frac
    self.random_state = random_state
    self.metric_subsample_frac = metric_subsample_frac
    self.results = None

  def run(self, X: np.ndarray, labels_ref: np.ndarray, sub_idx, scale, seed: int) -> StabilityResult:
    rng = np.random.default_rng(seed)

    num_clusters = len(set(labels_ref))

    if num_clusters <= 1:
      return StabilityResult(np.nan, np.nan, np.nan)

    # subsample stability
    if len(np.unique(labels_ref[sub_idx])) > 1:
      sub = self.clusterer_factory(random_state=rng.integers(1e9))
      labels_sub = sub.fit_predict(X[sub_idx])
      ari_sub = adjusted_rand_score(labels_ref[sub_idx], labels_sub)

      counts = np.bincount(labels_sub)
      min_cluster_frac = counts.min() / len(labels_sub)
    else:
      ari_sub = np.nan  # single cluster in subsample
      min_cluster_frac = 1.0

    # seed stability
    c1 = self.clusterer_factory(random_state=rng.integers(1e9))
    c2 = self.clusterer_factory(random_state=rng.integers(1e9))
    ari_seed = adjusted_rand_score(c1.fit_predict(X), c2.fit_predict(X))

    # noise stability (fit + predict on noisy data)
    noise = rng.normal(0, self.noise_scale * scale, X.shape)
    noisy = self.clusterer_factory(random_state=rng.integers(1e9))
    labels_noisy = noisy.fit_predict(X + noise)
    ari_noise = adjusted_rand_score(labels_ref, labels_noisy)

    return StabilityResult(ari_sub, ari_seed, ari_noise, min_cluster_frac)

  def run_repeated(self, X: np.ndarray, n_runs: int = 30, pbar=True) -> Tuple[pd.DataFrame, dict]:
    records = []

    rng     = np.random.default_rng(self.random_state)
    seeds   = rng.integers(0,1e9, size=n_runs)

    # reference clustering (deterministic)
    ref = self.clusterer_factory(random_state=self.random_state)
    labels_ref = ref.fit_predict(X)

    scale = np.maximum(X.std(axis=0), 1e-8)

    for i in (tqdm(range(n_runs)) if pbar else range(n_runs)):
      sub_idx = rng.choice(len(X), int(self.subsample_frac * len(X)), replace=False)
      r = self.run(X, labels_ref, sub_idx, scale=scale, seed=seeds[i])
      records.append(r)

    df = pd.DataFrame([r.__dict__ for r in records])

    self.results = df

    sub_rng = np.random.default_rng(self.random_state + 1 if self.random_state is not None else 1)
    sub_idx = sub_rng.choice(len(X), int(self.metric_subsample_frac * len(X)), replace=False)

    silhouette = (
      silhouette_score(X[sub_idx], labels_ref[sub_idx])
      if len(set(labels_ref)) > 1 else np.nan
    )

    davies_bouldin = (
      davies_bouldin_score(X[sub_idx], labels_ref[sub_idx])
      if len(set(labels_ref)) > 1 else np.nan
    )

    calinski_harabasz = (
      calinski_harabasz_score(X[sub_idx], labels_ref[sub_idx])
      if len(set(labels_ref)) > 1 else np.nan
    )

    stability = pd.DataFrame({
      "metric": df.columns,
      "median": df.median().values,
      "mean": df.mean().values,
      "std": df.std().values,
      "p10": df.quantile(0.10).values,
      "p90": df.quantile(0.90).values,
    })

    metrics = {"silhouette": silhouette, "davies_bouldin": davies_bouldin, "calinski_harabasz": calinski_harabasz}

    return stability, metrics


