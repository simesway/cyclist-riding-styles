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

  def run(self, X: np.ndarray) -> StabilityResult:
    rng = np.random.RandomState(self.random_state)

    # reference
    full = self.clusterer_factory()
    labels_full = full.fit_predict(X)

    # subsample
    idx = rng.choice(len(X), int(self.subsample_frac * len(X)), replace=False)
    sub = self.clusterer_factory()
    sub.fit_predict(X[idx])
    labels_sub = sub.predict(X)
    ari_sub = adjusted_rand_score(labels_full, labels_sub)

    # rerun
    c1 = self.clusterer_factory()
    c2 = self.clusterer_factory()
    ari_seed = adjusted_rand_score(
      c1.fit_predict(X),
      c2.fit_predict(X)
    )

    # noise
    noise = rng.normal(0, self.noise_scale * X.std(axis=0), X.shape)
    noisy = self.clusterer_factory()
    noisy.fit_predict(X + noise)
    labels_noisy = noisy.predict(X)
    ari_noise = adjusted_rand_score(labels_full, labels_noisy)

    return StabilityResult(
      ari_subsample=ari_sub,
      ari_seed=ari_seed,
      ari_noise=ari_noise
    )

  def run_repeated(self, X, n_runs=30):
    sub, seed, noise = [], [], []

    for _ in tqdm(range(n_runs)):
      r = self.run(X)
      sub.append(r.ari_subsample)
      seed.append(r.ari_seed)
      noise.append(r.ari_noise)

    metrics = {
      "ari_subsample": sub,
      "ari_seed": seed,
      "ari_noise": noise
    }

    rows = []
    for name, values in metrics.items():
      perc = np.percentile(values, [50, 10, 90])
      rows.append({
        "metric": name,
        "median": perc[0],
        "p10": perc[1],
        "p90": perc[2]
      })

    df = pd.DataFrame(rows)
    return df


