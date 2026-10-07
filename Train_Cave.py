"""Compatibility entry point; use train_gsno.py for new runs."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train_gsno.py")), run_name="__main__")
else:
    from train_gsno import *  # Keep imports used by existing experiment scripts.
