import pandas as pd
from sklearn.decomposition import PCA


class PCAReducer:
  def __init__(self, n_components: float = 0.95):
    self.pca = PCA(n_components=n_components)

  def fit_transform(self, X):
    return self.pca.fit_transform(X)

  def components(self, feature_names, relative=True):
    df_pca_weights = pd.DataFrame(
      data=self.pca.components_,
      columns=feature_names,
      index=[f"PC{i + 1}" for i in range(self.pca.n_components_)]
    )

    if relative:
      return df_pca_weights.abs().div(df_pca_weights.abs().sum(axis=1), axis=0)

    return df_pca_weights