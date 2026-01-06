import numpy as np
import pandas as pd


def mean(x: np.ndarray) -> float:
  return float(np.mean(x))


def mad(x: np.ndarray) -> float:
  m = np.mean(x)
  return float(np.mean(np.abs(x - m)))


def std(x: np.ndarray) -> float:
  return float(np.std(x, ddof=0))


def cv(x: np.ndarray) -> float:
  m = np.mean(x)
  return float(np.std(x) / m) if m != 0 else np.nan


def qcv(x: np.ndarray) -> float:
  q1, q3 = np.percentile(x, [25, 75])
  denom = abs(q3) + abs(q1)
  if np.isclose(denom, 0.0):
    return np.nan
  return float((q3 - q1) / denom)


def p90(x: np.ndarray) -> float:
  return float(np.percentile(x, 90))


def p90_abs(x: np.ndarray) -> float:
  return float(np.percentile(np.abs(x), 90))


def max_abs(x: np.ndarray) -> float:
  return float(np.max(np.abs(x)))


# ---------- dataframe helpers ----------

def col(df: pd.DataFrame, name: str) -> np.ndarray:
  """Fast column access as numpy array"""
  return df[name].to_numpy()


def stats_basic(df: pd.DataFrame, name: str) -> dict:
  x = col(df, name)
  return {
    "min": float(np.min(x)),
    "max": float(np.max(x)),
    "mean": mean(x),
    "std": std(x),
    "mad": mad(x),
    "cv": cv(x),
    "qcv": qcv(x),
  }


def stats_acc(df: pd.DataFrame, name: str) -> dict:
  x = col(df, name)
  return {
    "min": float(np.min(x)),
    "max": float(np.max(x)),
    "mad": mad(x),
    "qcv": qcv(x),
  }


def stats_lateral(df: pd.DataFrame, name: str) -> dict:
  x = col(df, name)
  return {
    "p90_abs": p90_abs(x),
    "mad": mad(x),
  }


def stats_yaw(df: pd.DataFrame, name: str) -> dict:
  x = col(df, name)
  return {
    "max_abs": max_abs(x),
    "mad": mad(x),
  }
