import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "config" / "weights.json"
DB_PATH = ROOT / "data" / "app.db"
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")


def get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def load_weights() -> dict:
    if not WEIGHTS_PATH.exists():
        return {
            "trend": {"ma": 0.3, "valuation": 0.4, "bond": 0.3},
            "sector": {"rs": 0.4, "flow": 0.3, "momentum": 0.3},
            "stock": {"roe": 0.3, "growth": 0.25, "valuation": 0.25, "dividend": 0.2},
        }
    with open(WEIGHTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_weights(weights: dict) -> None:
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
