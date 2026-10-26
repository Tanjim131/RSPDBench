#!/usr/bin/env python3
"""Compose the 1200x630 link-preview image (docs/assets/og.png) from the explorer renders."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2] / "docs" / "assets"
NOTO = "/usr/share/fonts/google-noto/"
MONO = "/usr/share/fonts/adobe-source-code-pro/SourceCodePro-Medium.otf"
W, H, BG, INK, INK2, ACCENT = 1200, 630, (10, 15, 25), (238, 242, 248), (169, 179, 198), (255, 93, 143)

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)
title = ImageFont.truetype(NOTO + "NotoSans-Black.ttf", 82)
body = ImageFont.truetype(NOTO + "NotoSans-Medium.ttf", 30)
small = ImageFont.truetype(NOTO + "NotoSans-SemiBold.ttf", 22)
try:
    mono = ImageFont.truetype(MONO, 17)
except OSError:
    mono = ImageFont.truetype(NOTO + "NotoSans-Regular.ttf", 17)

d.text((64, 72), "WACV 2027", font=small, fill=ACCENT)
d.text((60, 110), "RSPDBench", font=title, fill=INK)
lines = ["How do Earth-observation", "foundation models hold up", "when the satellite product", "itself is degraded?"]
for i, line in enumerate(lines):
    d.text((64, 236 + i * 42), line, font=body, fill=INK)
d.text((64, 430), "Blur, calibration drift, band", font=small, fill=INK2)
d.text((64, 460), "misregistration, noise, and the chains", font=small, fill=INK2)
d.text((64, 490), "they form. 5 datasets, 9 models.", font=small, fill=INK2)

tiles = [("clean.jpg", "CLEAN"), ("p1_s3.jpg", "P1 RESOLUTION"), ("p3_s3.jpg", "P3 MISREGISTRATION"),
         ("p4_s3.jpg", "P4 NOISE"), ("c1_gain_down_s3.jpg", "C1 ACQUISITION"), ("c2_gain_down_s3.jpg", "C2 HARMONIZATION")]
HLS = ROOT / "explorer" / "hls-burn-scars"
size, gap, x0, y0 = 172, 10, 600, 92
for i, (name, label) in enumerate(tiles):
    r, c = divmod(i, 3)
    x, y = x0 + c * (size + gap), y0 + r * (size + 44 + gap)
    tile = Image.open(HLS / name).crop((96, 96, 416, 416)).resize((size, size), Image.LANCZOS)
    img.paste(tile, (x, y))
    d.text((x, y + size + 10), label, font=mono, fill=INK2)
img.save(ROOT / "og.png", optimize=True)
print("wrote", ROOT / "og.png")
