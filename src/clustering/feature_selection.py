import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

class FeatureSelector:
  def __init__(self, corr_threshold=0.9, vif_threshold=5.0):
    self.corr_threshold = corr_threshold
    self.vif_threshold = vif_threshold
    self.drop_features = []

  @staticmethod
  def _compute_vif(X: np.ndarray):
    return np.array([
      variance_inflation_factor(X, i) for i in range(X.shape[1])
    ])

  def fit(self, df: pd.DataFrame):
    # --- Initial feature set ---
    remaining_features = list(df.columns)

    while True:
      changed = False
      df_current = df[remaining_features]

      # --- Step 1: Correlation filtering ---
      corr = df_current.corr().abs()
      np.fill_diagonal(corr.values, 0)

      corr_values = corr.values
      max_corr = corr_values.max()

      if max_corr > self.corr_threshold:
        i, j = np.unravel_index(np.argmax(corr_values), corr_values.shape)
        f1, f2 = corr.columns[i], corr.index[j]

        mean_corr = corr.mean()
        drop = f1 if mean_corr[f1] > mean_corr[f2] else f2

        remaining_features.remove(drop)
        changed = True

      # --- Step 2: VIF filtering ---
      X = StandardScaler().fit_transform(df_current[remaining_features].values)
      vifs = self._compute_vif(X)

      max_vif = vifs.max()
      if max_vif > self.vif_threshold:
        drop = remaining_features[int(np.argmax(vifs))]
        remaining_features.remove(drop)
        changed = True

      if not changed:
        break

    self.drop_features = [c for c in df.columns if c not in remaining_features]
    return self.drop_features

  def transform(self, df: pd.DataFrame):
    return df.drop(columns=self.drop_features, errors='ignore')

  def fit_transform(self, df: pd.DataFrame):
    self.fit(df)
    return self.transform(df)
