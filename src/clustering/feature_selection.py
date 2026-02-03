import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

class FeatureSelector:
  def __init__(self, corr_threshold=0.9, vif_threshold=5.0):
    self.corr_threshold = corr_threshold
    self.vif_threshold = vif_threshold
    self.drop_features = []

  def fit(self, df: pd.DataFrame):
    # --- Initial feature set ---
    remaining_features = list(df.columns)

    while True:
      corr_matrix = df[remaining_features].corr()
      to_drop = set()

      # --- Step 1: Correlation filtering ---
      for i in range(len(corr_matrix.columns)):
        for j in range(i):
          if abs(corr_matrix.iloc[i, j]) > self.corr_threshold:
            to_drop.add(corr_matrix.columns[i])

      if not to_drop:
        break

      remaining_features = [feat for feat in remaining_features if feat not in to_drop]
      df_reduced = df[remaining_features]

      # --- Step 2: VIF filtering ---
      X = df_reduced.values
      for i, feat in enumerate(df_reduced.columns):
        vif = variance_inflation_factor(X, i)
        if vif > self.vif_threshold:
          to_drop.add(feat)

      if not to_drop:
        break

      remaining_features = [feat for feat in remaining_features if feat not in to_drop]

    self.drop_features = [feat for feat in df.columns if feat not in remaining_features]
    return self.drop_features

  def transform(self, df: pd.DataFrame):
    return df.drop(columns=self.drop_features, errors='ignore')

  def fit_transform(self, df: pd.DataFrame):
    self.fit(df)
    return self.transform(df)
