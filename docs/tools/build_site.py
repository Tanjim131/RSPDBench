#!/usr/bin/env python3
"""Build the project page.

Writes docs/index.html (GitHub Pages) from docs/tools/page.src.html with the
results inlined, and optionally a bare fragment plus assets for previews.

    python docs/tools/build_page_data.py
    python docs/tools/build_site.py --repo-url https://github.com/<you>/RSPDBench \
        --site-url https://<you>.github.io/RSPDBench/
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
PLACEHOLDER_REPO = "https://github.com/TODO/RSPDBench"
DESCRIPTION = ("How do Earth-observation foundation models hold up when the satellite product itself is degraded? "
               "RSPDBench applies physically grounded resolution, calibration, misregistration, noise and "
               "harmonization defects to five datasets and nine models. WACV 2027.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="https://github.com/Tanjim131/RSPDBench")
    ap.add_argument("--site-url", default="https://tanjim131.github.io/RSPDBench/")
    ap.add_argument("--fragment-out", type=Path, help="also write a bare fragment + assets here (for previews)")
    args = ap.parse_args()

    src = (DOCS / "tools" / "page.src.html").read_text()
    data = (DOCS / "assets" / "data.json").read_text().replace("</", "<\\/")
    page = src.replace("/*__DATA__*/", data).replace(PLACEHOLDER_REPO, args.repo_url.rstrip("/"))
    head, body = page.split('<header class="hero"', 1)
    body = '<header class="hero"' + body
    head_lines = [ln for ln in head.splitlines() if not ln.startswith(("<title>", '<meta name="claude:'))]
    site = args.site_url.rstrip("/") + "/"
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>RSPDBench</title>
<meta name="description" content="{DESCRIPTION}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="RSPDBench">
<meta property="og:title" content="RSPDBench: Earth-observation foundation models under real satellite product defects">
<meta property="og:description" content="{DESCRIPTION}">
<meta property="og:url" content="{site}">
<meta property="og:image" content="{site}assets/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="RSPDBench title beside six versions of a satellite burn-scar image: clean, blurred, misregistered, noisy and two compound defects">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{site}assets/og.png">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
{chr(10).join(head_lines).strip()}
</head>
<body>
{body.strip()}
</body>
</html>
"""
    (DOCS / "index.html").write_text(doc)
    print("wrote", DOCS / "index.html", f"({len(doc) / 1024:.0f} KB)")

    if args.fragment_out:
        out = args.fragment_out
        out.mkdir(parents=True, exist_ok=True)
        (out / "index.html").write_text(page)
        for sub in ("explorer", "realism"):
            shutil.copytree(DOCS / "assets" / sub, out / "assets" / sub, dirs_exist_ok=True)
        print("wrote preview fragment and assets to", out)


if __name__ == "__main__":
    main()
