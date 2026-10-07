"""Compatibility entry point; use evaluate_gsno_multiscale.py for new runs."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_gsno_multiscale import *


if __name__ == "__main__":
    main()
