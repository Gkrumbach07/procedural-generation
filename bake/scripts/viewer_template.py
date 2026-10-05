"""Refresh a world's viewer page from the template, leaving its data alone.

    python scripts/viewer_template.py worlds/earth-v19

``export_viewer`` encodes every frame again (a minute on the earth preset);
a change to ``globe/viz/viewer.html`` alone only needs ``viewer/index.html``
written again around the script tags it already has.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from globe.viz.viewer import PLACEHOLDER, TEMPLATE  # noqa: E402


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    index = Path(argv[0]) / "viewer" / "index.html"
    tags = re.search(r'<script src="data/meta\.js"></script>.*?<script src="scout\.js"></script>', index.read_text(), re.S)
    if tags is None:
        print(f"{index}: not a viewer export this script knows (no data script tags)")
        return 1
    index.write_text(TEMPLATE.read_text().replace(PLACEHOLDER, tags.group(0)))
    print(f"{index}: page refreshed from {TEMPLATE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
