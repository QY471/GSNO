"""Run the shared GSNO trainer on Harvard."""

from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    trainer = Path(__file__).resolve().with_name("Train_Cave.py")
    sys.argv = [str(trainer), "--dataset", "harvard", *sys.argv[1:]]
    runpy.run_path(str(trainer), run_name="__main__")
