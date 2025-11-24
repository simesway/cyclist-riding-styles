import matplotlib.pyplot as plt
from collections import defaultdict

def plot_trajectory(x, y, ax=None):
  if ax is None:
    fig, ax = plt.subplots()

  ax.plot(x, y)
  ax.scatter(x, y, s=8)
  ax.set_aspect("equal", adjustable="box")
  ax.set_xlabel("Translation x")
  ax.set_ylabel("Translation y")

  if ax is None:
    plt.show()
  else:
    return ax

def plot_interactions(df, target_id, interactions, ax=None, figsize=(16, 10), distance=50):
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)


  # collect all interactions grouped by other_id
  windows = defaultdict(list)

  for inter in interactions:
      track_id, other_id, t0, t1 = inter[0], inter[1], inter[2], inter[3]
      windows[other_id].append((t0, t1))

  # 1. plot all other_id trajectories in grey
  for other_id, spans in windows.items():
      df_other = df[df["track_id"] == other_id]
      ax.plot(df_other["translation_x"], df_other["translation_y"], c="grey")

  # 2. plot all highlight windows in color (one color per interaction)
  for other_id, spans in windows.items():
      df_other = df[df["track_id"] == other_id]
      for t0, t1 in spans:
          df_window = df_other[(df_other["timestamp"] >= t0) &
                               (df_other["timestamp"] <= t1)]
          ax.plot(df_window["translation_x"], df_window["translation_y"], linewidth=3.5, label=other_id)

  df_main = df[df["track_id"] == target_id]
  ax.plot(df_main["translation_x"], df_main["translation_y"], c="red", linewidth=3.5, label=f"cyclist:{target_id}")

  x_min_main = df_main["translation_x"].min() - distance
  x_max_main = df_main["translation_x"].max() + distance
  y_min_main = df_main["translation_y"].min() - distance
  y_max_main = df_main["translation_y"].max() + distance

  x_min_current = ax.get_xlim()[0]
  x_max_current = ax.get_xlim()[1]
  y_min_current = ax.get_ylim()[0]
  y_max_current = ax.get_ylim()[1]

  x_min = max(x_min_current, x_min_main)
  x_max = min(x_max_current, x_max_main)
  y_min = max(y_min_current, y_min_main)
  y_max = min(y_max_current, y_max_main)

  ax.set_xlim(x_min, x_max)
  ax.set_ylim(y_min, y_max)

  ax.set_aspect('equal')

  if ax is None:
    plt.show()
  else:
    return ax


