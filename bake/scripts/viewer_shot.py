#!/usr/bin/env python3
"""Screenshot the HTML viewer headlessly -- a view of a bake without a browser window.

    python scripts/viewer_shot.py worlds/demo --out shot.png
    python scripts/viewer_shot.py worlds/demo --view flat --frame 0 --layer plate --out plates0.png
    python scripts/viewer_shot.py worlds/demo --lat 20 --lon -40 --zoom 4 --out zoom.png
    python scripts/viewer_shot.py worlds/demo --sweep 8 --out sweep.png   # 8 timeline frames in a grid

Any viewer URL state works through ``--hash`` (copy it from the browser's
address bar).  Needs Playwright with Chromium (``pip install playwright &&
playwright install chromium``); WebGL runs on SwiftShader when there is no GPU.
"""
import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("world", help="world directory, or a viewer index.html")
    ap.add_argument("--out", default="viewer_shot.png")
    ap.add_argument("--view", choices=["globe", "flat"], default=None)
    ap.add_argument("--frame", type=int, default=None, help="timeline index (default: the final frame)")
    ap.add_argument("--layer", default=None)
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--zoom", type=float, default=None)
    ap.add_argument("--hash", default="", help="raw URL state, e.g. 'f=3&v=flat&z=2'")
    ap.add_argument("--size", default="1400x900")
    ap.add_argument("--ui", action="store_true", help="keep the panels and timeline in the shot")
    ap.add_argument("--sweep", type=int, default=0, help="grid of N evenly spaced timeline frames")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    p = Path(args.world)
    html = p if p.suffix == ".html" else p / "viewer" / "index.html"
    if not html.exists():
        raise SystemExit(f"no viewer at {html} -- run scripts/export_viewer.py first")
    w, h = (int(v) for v in args.size.split("x"))
    parts = [args.hash] if args.hash else []
    for k, v in (("v", args.view), ("layer", args.layer), ("lat", args.lat), ("lon", args.lon), ("z", args.zoom)):
        if v is not None:
            parts.append(f"{k}={v}")
    if not args.ui:
        parts.append("ui=0")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                           "--allow-file-access-from-files"])
        page = browser.new_page(viewport={"width": w, "height": h})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.goto(html.resolve().as_uri() + "#" + "&".join(parts))
        page.wait_for_function("document.body.dataset.ready === '1'", timeout=300_000)
        n = page.evaluate("GLOBE_VIEWER.meta.frames.length")
        if args.sweep:
            from PIL import Image
            import io

            idx = sorted({round(k * (n - 1) / max(1, args.sweep - 1)) for k in range(args.sweep)})
            shots = []
            for i in idx:
                page.evaluate(f"GLOBE_VIEWER.setFrame({i}); GLOBE_VIEWER.draw()")
                page.wait_for_timeout(150)
                shots.append(Image.open(io.BytesIO(page.screenshot())))
            cols = min(4, len(shots))
            rows = (len(shots) + cols - 1) // cols
            tw, th = w // 2, h // 2
            grid = Image.new("RGB", (cols * tw, rows * th))
            for k, im in enumerate(shots):
                grid.paste(im.resize((tw, th)), ((k % cols) * tw, (k // cols) * th))
            grid.save(args.out)
        else:
            if args.frame is not None:
                page.evaluate(f"GLOBE_VIEWER.setFrame({args.frame}); GLOBE_VIEWER.draw()")
                page.wait_for_timeout(150)
            page.screenshot(path=args.out)
        label = page.evaluate("document.getElementById('tl-label').textContent")
        browser.close()
    for e in errors:
        print("page error:", e, file=sys.stderr)
    print(f"{args.out}  ({n} frames; showing: {label})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
