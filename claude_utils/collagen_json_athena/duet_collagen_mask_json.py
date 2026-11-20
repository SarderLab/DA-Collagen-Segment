"""Collagen mask of each DUET slide as a DSA (Athena) annotation JSON, for overlay on <slide>_B.svs.

  python duet_collagen_mask_json.py --images <aim3/images> --out <dir> [--only 133H-0] [--jobs 4]

Mask = the collagen mask of feature extraction v5: M > 127 (model probability > 0.5) and CE > 0 (CE has the
glomeruli masked out). CE alone is not used: CE > 0 covers most tissue (dim green haze over tubules).
Built at level 0, then downsampled 4x (2 um/px; a 4x4 block is collagen if >= half its pixels are), traced
with OpenCV (outer contours + holes), simplified to 1 px (2 um), regions < 20 um^2 dropped. Coordinates are
level-0 pixels, so the polygons sit on the B, M and CE slides (same size). Green, so it differs from the
blue/yellow cortex/medulla layers.
Writes <out>/<slide>_B_collagen_mask.json (one annotation document). Existing files are skipped.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import tifffile
import zarr

F, MPP, CHUNK, MIN_UM2 = 4, 0.5, 4096, 20.0
FILL, LINE = "rgba(0, 200, 83, 0.45)", "rgba(0, 200, 83, 1)"
NAME = "collagen mask (M > 0.5, CE > 0)"


def collagen_mask(folder: Path) -> np.ndarray:
    z = {k: zarr.open(tifffile.TiffFile(folder / f"{k}.svs").series[0].aszarr(level=0), mode="r") for k in ("M", "CE")}
    h, w = z["M"].shape[:2]
    out = np.zeros((h // F, w // F), np.uint8)
    for y in range(0, h // F * F, CHUNK):
        for x in range(0, w // F * F, CHUNK):
            m, ce = (np.asarray(z[k][y:y + CHUNK, x:x + CHUNK]) for k in ("M", "CE"))
            m = m[..., 0] if m.ndim == 3 else m
            ce = ce.max(-1) if ce.ndim == 3 else ce
            c = (m > 127) & (ce > 0)
            hh, ww = c.shape[0] // F * F, c.shape[1] // F * F
            block = c[:hh, :ww].reshape(hh // F, F, ww // F, F).mean((1, 3)) >= 0.5
            out[y // F:y // F + block.shape[0], x // F:x // F + block.shape[1]] = block
    return out


def to_points(c: np.ndarray) -> list:
    return [[float(px * F), float(py * F), 0] for px, py in c[:, 0, :]]


def elements(mask: np.ndarray) -> list:
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    min_px = MIN_UM2 / (F * MPP) ** 2
    simple = [cv2.approxPolyDP(c, 1.0, True) for c in contours]
    out = []
    for i, (c, (nxt, prev, child, parent)) in enumerate(zip(simple, hier)):
        if parent != -1 or len(c) < 3 or cv2.contourArea(c) < min_px:
            continue
        holes, j = [], child
        while j != -1:
            if len(simple[j]) >= 3 and cv2.contourArea(simple[j]) >= min_px:
                holes.append(to_points(simple[j]))
            j = hier[j][0]
        out.append({"type": "polyline", "id": uuid4().hex[:24], "closed": True, "points": to_points(c),
                    "holes": holes, "fillColor": FILL, "lineColor": LINE, "lineWidth": 1})
    return out


def run(args):
    folder, out_dir = args
    dst = out_dir / f"{folder.name}_B_collagen_mask.json"
    if dst.exists():
        return f"{folder.name}: exists"
    mask = collagen_mask(folder)
    els = elements(mask)
    doc = {"name": NAME,
           "description": "Collagen segmentation used by feature extraction v5 (M > 127 and CE > 0), "
                          "traced at 2 um/px; regions < 20 um^2 dropped",
           "attributes": {"sample_name": f"{folder.name}_B", "layer": "collagen_mask",
                          "coordinate_system": "level-zero WSI pixels", "downsample": F,
                          "collagen_fraction_of_slide": round(float(mask.mean()), 5)},
           "elements": els}
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps([doc], separators=(",", ":")))
    tmp.replace(dst)
    return f"{folder.name}: {len(els)} polygons, {sum(len(e['points']) for e in els)} vertices"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="*")
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    folders = [a.images / s for s in a.only] if a.only else sorted(
        d for d in a.images.iterdir() if (d / "M.svs").is_file() and (d / "CE.svs").is_file())
    with Pool(a.jobs) as pool:
        for msg in pool.imap_unordered(run, [(f, a.out) for f in folders]):
            print(msg, flush=True)
