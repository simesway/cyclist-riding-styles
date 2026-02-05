import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product
from sklearn.utils import resample

from clustering.clusterer import KMeansClusterer
from clustering.feature_selection import FeatureSelector
from clustering.pca import PCAReducer
from clustering.pipeline import ClusteringPipeline


default_param_grid = {
    "corr_threshold": np.arange(0.7, 0.98, 0.05),
    "vif_threshold": [1, 2, 5, 10, 20, 30, 40, 50, 100, 500, 1000],
    "k": [2, 3],
    "components": np.arange(0.6, 0.95, 0.05)
}


def grid_search(
  data,
  base_df,
  adapter_factory,
  getter,
  param_grid,
  min_cluster_frac=0.05,
  subsample_frac=1.0,
  n_runs=5,
):
  results = []

  grid = list(product(
    param_grid["corr_threshold"],
    param_grid["vif_threshold"],
    param_grid["k"],
    param_grid["components"],
  ))

  for corr, vif, k, comp in tqdm(grid, total=len(grid)):
    try:
        # --- feature selection ---
      selector = FeatureSelector(corr_threshold=corr, vif_threshold=vif)
      drop_feats = selector.fit(base_df)

      adapter = adapter_factory(drop_feats)

      pipeline = ClusteringPipeline(
        adapter=adapter,
        getter=getter,
        pca=PCAReducer(n_components=comp),
        clusterer=KMeansClusterer(k=k),
      )

      X, labels, valid = pipeline.run(data)

      unique, counts = np.unique(labels, return_counts=True)
      if len(unique) < k or counts.min() < min_cluster_frac * len(data):
        continue

      subsample = resample(
        data,
        replace=False,
        n_samples=int(subsample_frac * len(data)),
        stratify=labels
      )

      stability = pipeline.stability_test(subsample, n_runs=n_runs, pbar=False)
      ex_var = pipeline.pca.explained_variance()

      results.append({
        "corr_threshold": corr,
        "vif_threshold": vif,
        "k": k,
        "components": comp,
        "silhouette": stability.loc[
          stability.metric == "silhouette", "median"
        ].iloc[0],
        "ari_subsample": stability.loc[
          stability.metric == "ari_subsample", "median"
        ].iloc[0],
        "ari_noise": stability.loc[
          stability.metric == "ari_noise", "median"
        ].iloc[0],
        "explained_variance": sum(ex_var),
        "num_components": len(ex_var),
        **{f"cluster_{i}": c for i, c in enumerate(counts)},
      })

    except Exception as e:
      print(e)
      continue

  return pd.DataFrame(results)
