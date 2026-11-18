"""Package entry point: run with `uvicorn src.api:app`."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api import app  # noqa: E402,F401
