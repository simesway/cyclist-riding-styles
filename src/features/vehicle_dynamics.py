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


def acceleration(df: pd.DataFrame) -> pd.Series:
  """Compute total acceleration magnitude."""
  return np.hypot(df['acceleration_x'], df['acceleration_y'])


def longitudinal_acceleration(df: pd.DataFrame) -> pd.Series:
  """
  Compute longitudinal acceleration along the vehicle's forward direction.
  """
  cos_yaw = np.cos(df['rotation_z'])
  sin_yaw = np.sin(df['rotation_z'])
  return df['acceleration_x'] * cos_yaw + df['acceleration_y'] * sin_yaw


def lateral_acceleration(df: pd.DataFrame) -> pd.Series:
  """
  Compute lateral acceleration perpendicular to the vehicle's forward direction.
  """
  cos_yaw = np.cos(df['rotation_z'])
  sin_yaw = np.sin(df['rotation_z'])
  return -df['acceleration_x'] * sin_yaw + df['acceleration_y'] * cos_yaw