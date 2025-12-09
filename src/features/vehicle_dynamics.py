import numpy as np
import pandas as pd

def velocity(df: pd.DataFrame) -> pd.Series:
  """Compute total velocity magnitude."""
  return np.hypot(df['velocity_x'], df['velocity_y'])

def longitudinal_velocity(df: pd.DataFrame) -> pd.Series:
  """
  Compute longitudinal velocity along the vehicle's forward direction.
  """
  cos_yaw = np.cos(df['rotation_z'])
  sin_yaw = np.sin(df['rotation_z'])
  return df['velocity_x'] * cos_yaw + df['velocity_y'] * sin_yaw


def lateral_velocity(df: pd.DataFrame) -> pd.Series:
  """
  Compute lateral velocity perpendicular to the vehicle's forward direction.
  """
  cos_yaw = np.cos(df['rotation_z'])
  sin_yaw = np.sin(df['rotation_z'])
  return -df['velocity_x'] * sin_yaw + df['velocity_y'] * cos_yaw
