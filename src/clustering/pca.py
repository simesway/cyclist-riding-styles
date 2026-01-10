import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class PCAReducer:
  def __init__(self, n_components: float = 0.95):
    self.pca = PCA(n_components=n_components)

  def fit_transform(self, X):
    return self.pca.fit_transform(X)

  def transform(self, X):
    return self.pca.transform(X)

  def components(self, feature_names, relative=True):
    df_pca_weights = pd.DataFrame(
      data=self.pca.components_,
      columns=feature_names,
      index=[f"PC{i + 1}" for i in range(self.pca.n_components_)]
    )

    if relative:
      return df_pca_weights.abs().div(df_pca_weights.abs().sum(axis=1), axis=0)

    return df_pca_weights

  def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
    return self.pca.inverse_transform(Z)

  def explained_variance(self) -> float:
    return self.pca.explained_variance_ratio_


class SplitPCAReducer:
  def __init__(
      self,
      mag_idx: list[int],
      vol_idx: list[int],
      mag_pca: PCA,
      vol_pca: PCA,
  ):
    self.mag_idx = mag_idx
    self.vol_idx = vol_idx
    self.mag_pca = mag_pca
    self.vol_pca = vol_pca

  def fit_transform(self, X: np.ndarray) -> np.ndarray:
    X_mag = X[:, self.mag_idx]
    X_vol = X[:, self.vol_idx]

    Z_mag = self.mag_pca.fit_transform(X_mag)
    Z_vol = self.vol_pca.fit_transform(X_vol)

    return np.hstack([Z_mag, Z_vol])

  def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
    k_mag = self.mag_pca.n_components_
    Z_mag, Z_vol = Z[:, :k_mag], Z[:, k_mag:]

    X_mag = self.mag_pca.inverse_transform(Z_mag)
    X_vol = self.vol_pca.inverse_transform(Z_vol)

    X = np.zeros((len(Z), len(self.mag_idx) + len(self.vol_idx)))
    X[:, self.mag_idx] = X_mag
    X[:, self.vol_idx] = X_vol
    return X

  def explained_variance(self) -> dict:
    return {
      "magnitude": self.mag_pca.explained_variance_ratio_,
      "volatility": self.vol_pca.explained_variance_ratio_,
    }
