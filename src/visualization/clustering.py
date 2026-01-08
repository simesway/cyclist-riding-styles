import numpy as np
import matplotlib.pyplot as plt


def plot_cluster_radar(df_centers, ax=None):
  """
  df_centers: DataFrame, rows = clusters, columns = features
  ax: optional matplotlib polar axis
  """
  features = df_centers.columns
  clusters = df_centers.index
  n_features = len(features)

  # angles
  angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
  angles += angles[:1]

  if ax is None:
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

  # reference line
  HANGLES = np.linspace(0, 2 * np.pi)
  ax.plot(HANGLES, np.zeros_like(HANGLES), ls=(0, (6, 6)), c="red")

  for cluster_id in clusters:
    values = df_centers.loc[cluster_id].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=f"Cluster {cluster_id}")
    ax.fill(angles, values, alpha=0.25)

  ax.set_xticks(angles[:-1])
  ax.set_xticklabels(features, rotation=45, ha="right")
  ax.set_rlabel_position(0)
  ax.tick_params(axis="y", labelsize=8, colors="grey")
  ax.set_title("Cluster Centers Radar Chart", pad=20)
  ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

  return ax
