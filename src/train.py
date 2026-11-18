"""Run the top-level grouped-CV model comparison script."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"), run_name="__main__")
