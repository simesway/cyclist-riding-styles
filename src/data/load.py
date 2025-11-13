import json
import pandas as pd
from pathlib import Path

def load_metadata(path: str | Path) -> dict:
  """Load metadata JSON file."""
  with open(path, "r") as f:
    return json.load(f)

def load_trajectory_file(path: str | Path) -> pd.DataFrame:
  """Load a single CSV file containing trajectory data."""
  return pd.read_csv(path)

def load_all_trajectories(path: str | Path) -> pd.DataFrame:
  """Load and concatenate all trajectory files from a directory."""
  directory = Path(path)
  csv_files = sorted(directory.glob("*.csv"))
  dfs = []
  for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df)
  return pd.concat(dfs, ignore_index=True)
