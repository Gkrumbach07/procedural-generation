#!/usr/bin/env python3
"""CLI wrapper: python scripts/inspect_world.py WORLD_DIR [--field NAME]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from globe.cli import inspect_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(inspect_main())
