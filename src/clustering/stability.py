import numpy as np
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm

from sklearn.metrics import adjusted_rand_score


@dataclass
class StabilityResult:
    ari_subsample: float | tuple
    ari_seed: float | tuple
    ari_noise: float | tuple


class RegimeStabilityTester:
  def __init__(
      self,
      clusterer_factory,
      noise_scale: float = 0.05,
      subsample_frac: float = 0.8,
      random_state: int | None = None,
  ):
    self.clusterer_factory = clusterer_factory
    self.noise_scale = noise_scale
    self.subsample_frac = subsample_frac
    self.random_state = random_state
    self.results = None

  def _rng(self, seed_offset: int = 0):
    base = self.random_state or 0
    return np.random.RandomState(base + seed_offset)

  def run(self, X: np.ndarray, run_id: int = 0) -> StabilityResult:
    rng = self._rng(run_id)

    # reference
    ref = self.clusterer_factory(random_state=0)
    labels_ref = ref.fit_predict(X)

    # subsample stability
    idx = rng.choice(len(X), int(self.subsample_frac * len(X)), replace=False)
    sub = self.clusterer_factory(random_state=rng.randint(1e9))
    sub.fit_predict(X[idx])
    labels_sub = sub.predict(X)
    ari_sub = adjusted_rand_score(labels_ref, labels_sub)

    # seed stability
    c1 = self.clusterer_factory(random_state=rng.randint(1e9))
    c2 = self.clusterer_factory(random_state=rng.randint(1e9))
    ari_seed = adjusted_rand_score(
      c1.fit_predict(X),
      c2.fit_predict(X),
    )

    # noise stability (fit + predict on noisy data)
    noise = rng.normal(0, self.noise_scale * X.std(axis=0), X.shape)
    noisy = self.clusterer_factory(random_state=rng.randint(1e9))
    labels_noisy = noisy.fit_predict(X + noise)
    ari_noise = adjusted_rand_score(labels_ref, labels_noisy)

    return StabilityResult(ari_sub, ari_seed, ari_noise)

  def run_repeated(self, X: np.ndarray, n_runs: int = 30) -> pd.DataFrame:
    records = []

    for i in tqdm(range(n_runs)):
      r = self.run(X, run_id=i)
      records.append(r)

    df = pd.DataFrame([r.__dict__ for r in records])

    self.results = df

    return pd.DataFrame({
      "metric": df.columns,
      "median": df.median().values,
      "p10": df.quantile(0.10).values,
      "p90": df.quantile(0.90).values,
    })


