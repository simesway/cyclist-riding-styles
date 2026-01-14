import json
import gzip
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from omegaconf import OmegaConf

from features.base import OvertakingFeatures, FollowingFeatures, RidingFeatures, TrafficFeatures, InfrastructureFeatures
from maneuvers.base import OvertakingManeuver, FollowingManeuver, ManeuverMeta, WindowRecord


def load_metadata(path: str | Path) -> dict:
  """Load metadata JSON file."""
  with open(path, "r") as f:
    return json.load(f)

def load_trajectory_file(path: str | Path) -> pd.DataFrame:
  """Load a single CSV file containing trajectory data."""
  return pd.read_csv(path)

def load_all_trajectories(path: str | Path) -> pd.DataFrame:
  """Load and concatenate all trajectory files from a directory and sort by timestamp and track_id."""
  directory = Path(path)
  csv_files = directory.glob("*.csv")
  dfs = []
  for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df)
  df = pd.concat(dfs, ignore_index=True)
  return df.sort_values(by=["timestamp", "track_id"]).reset_index(drop=True)

def save_list(path: str, objs: list):
  """Save a list of dataclass objects to a gzipped JSON file."""
  with gzip.open(path, "wt", encoding="utf-8") as f:
    json.dump([asdict(o) for o in objs], f, separators=(",", ":"))

def load_list(path: str, cls):
  """Load a list of dataclass objects from a gzipped JSON file."""
  with gzip.open(path, "rt", encoding="utf-8") as f:
    data = json.load(f)
  return [cls(**item) for item in data]

def load_overtaking(path):
  """Load a list of OvertakingManeuver objects from gzip JSON with nested features restored."""
  with gzip.open(path, "rt", encoding="utf-8") as f:
    data = json.load(f)

  out = []
  for d in data:
    d['features'] = OvertakingFeatures(**d['features'])
    out.append(OvertakingManeuver(**d))
  return out

def load_following(path):
  """Load a list of FollowingManeuver objects from gzip JSON with nested features restored."""
  with gzip.open(path, "rt", encoding="utf-8") as f:
    data = json.load(f)

  out = []
  for d in data:
    d['features'] = FollowingFeatures(**d['features'])
    out.append(FollowingManeuver(**d))
  return out

def load_windowrecords(path):
    """Load a list of WindowRecord objects from gzip JSON with nested features restored."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    for d in data:
        d['meta'] = ManeuverMeta(**d['meta'])
        if d.get('riding') is not None:
            d['riding'] = RidingFeatures(**d['riding'])
        if d.get('traffic') is not None:
            d['traffic'] = TrafficFeatures(**d['traffic'])
        if d.get('infrastructure') is not None:
            d['infrastructure'] = InfrastructureFeatures(**d['infrastructure'])
        out.append(WindowRecord(**d))
    return out

def load_config(path: str | Path):
  """Load configuration YAML file."""
  return OmegaConf.load(path)