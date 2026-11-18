"""Run the top-level student-test evaluation script."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / "evaluate.py"), run_name="__main__")
