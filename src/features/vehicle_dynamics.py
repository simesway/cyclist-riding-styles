import numpy as np
import pandas as pd


def speed(df: pd.DataFrame) -> np.ndarray:
  vx = df["velocity_x"].to_numpy()
  vy = df["velocity_y"].to_numpy()
  return np.hypot(vx, vy)


def acceleration(df: pd.DataFrame) -> np.ndarray:
  ax = df["acceleration_x"].to_numpy()
  ay = df["acceleration_y"].to_numpy()
  return np.hypot(ax, ay)


def longitudinal_velocity(df: pd.DataFrame) -> np.ndarray:
  yaw = df["rotation_z"].to_numpy()
  vx = df["velocity_x"].to_numpy()
  vy = df["velocity_y"].to_numpy()
  return vx * np.cos(yaw) + vy * np.sin(yaw)


def lateral_velocity(df: pd.DataFrame) -> np.ndarray:
  yaw = df["rotation_z"].to_numpy()
  vx = df["velocity_x"].to_numpy()
  vy = df["velocity_y"].to_numpy()
  return -vx * np.sin(yaw) + vy * np.cos(yaw)


def longitudinal_acceleration(df: pd.DataFrame) -> np.ndarray:
  yaw = df["rotation_z"].to_numpy()
  ax = df["acceleration_x"].to_numpy()
  ay = df["acceleration_y"].to_numpy()
  return ax * np.cos(yaw) + ay * np.sin(yaw)


def lateral_acceleration(df: pd.DataFrame) -> np.ndarray:
  yaw = df["rotation_z"].to_numpy()
  ax = df["acceleration_x"].to_numpy()
  ay = df["acceleration_y"].to_numpy()
  return -ax * np.sin(yaw) + ay * np.cos(yaw)


def yaw_rate(df: pd.DataFrame) -> np.ndarray:
  """
  Compute yaw rate (angular velocity around vertical axis).
  Assumes 'rotation_z' in radians and evenly spaced timestamps.
  """
  yaw = df["rotation_z"].to_numpy()
  t = df["t"].to_numpy()  # time in seconds

  dyaw = np.diff(yaw, prepend=yaw[0])
  dt = np.diff(t, prepend=t[0])
  dt[dt == 0] = np.nan  # avoid division by zero

  return dyaw / dt