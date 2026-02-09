import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product

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
  subsample_frac=1.0,
  silhouette_subsample=0.8,
  n_runs=5,
  seed=0
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

      metrics, silhouette = pipeline.stability_test(
        data,
        subsample_frac=subsample_frac,
        silhouette_subsample=silhouette_subsample,
        n_runs=n_runs,
        pbar=False, seed=seed
      )
      ex_var = pipeline.pca.explained_variance()

      ari_subsample_median = metrics.loc[metrics.metric == "ari_subsample", "median"].iloc[0]
      ari_noise_median = metrics.loc[metrics.metric == "ari_noise", "median"].iloc[0]
      ari_seed_median = metrics.loc[metrics.metric == "ari_seed", "median"].iloc[0]

      results.append({
        "corr_threshold": corr,
        "vif_threshold": vif,
        "k": k,
        "components": comp,
        "silhouette": silhouette,
        "ari_subsample": ari_subsample_median,
        "ari_noise": ari_noise_median,
        "ari_seed": ari_seed_median,
        "min_cluster_frac_median": metrics.loc[
          metrics.metric == "min_cluster_frac", "median"
        ].iloc[0],
        "min_cluster_frac_std": metrics.loc[
          metrics.metric == "min_cluster_frac", "std"
        ].iloc[0],
        "explained_variance": sum(ex_var),
        "num_components": len(ex_var)
      })

    except Exception as e:
      print(e)
      continue

  return pd.DataFrame(results)
