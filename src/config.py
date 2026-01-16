from pathlib import Path
from dotenv import load_dotenv
from omegaconf import OmegaConf
import os

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

cfg = OmegaConf.load(ROOT / "config.yaml")
cfg = OmegaConf.merge(cfg, {"env": dict(os.environ)})

OmegaConf.resolve(cfg)