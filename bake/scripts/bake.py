#!/usr/bin/env python3
"""CLI wrapper: python scripts/bake.py --world NAME [--params p.yaml] [--from STAGE] [--to STAGE]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from globe.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
