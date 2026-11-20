"""
Batch Collagen Feature Extraction — Whole Slide Image (WSI) version 5
---------------------------------------------------------------------
v5 (prediction map M): with `collagen_mask_file` set (e.g. "M.svs", the
collagen segmentation sigmoid output, 0-1 stored as 0-255), the collagen mask
is M > collagen_mask_threshold (127 = probability 0.5) and CE > 0 (CE has the
glomeruli masked out), and features of M itself are added. No fluorescence
intensity enters the M features:
  per tile:  M Sum, M Valid Area, M Mean (soft collagen fraction over tissue),
             M Collagen Mean (model confidence inside the mask),
             M Masked Out Area (M >= 0.25 with CE = 0: glomeruli, excluded)
  per slide: m_density (sum M / valid tissue), m_mean_mean/_iqr,
             m_collagen_mean_mean/_iqr, m_masked_out_share
All v4 columns are kept unchanged, so v4 and v5 results can be compared.

v4 description follows.
---------------------------------------------------------------------
v4 (CE inside collagen): collagen signal is measured ONLY inside the
collagen mask, and the CE signal that lies outside the mask is reported as
a QC measure.

Why: CE is not zero outside collagen. On the local test slide ~85% of
tissue pixels have CE > 0, and ~39% of the summed CE signal is in pixels
below the collagen threshold (a dim haze over tubules, median 24 um from the
nearest collagen pixel). v3's ce_density averages CE over ALL tissue, so it
includes that haze.

  collagen mask   default: CE > slide-level Otsu threshold (as v2/v3).
                  If config `collagen_mask_file` names a file in each slide
                  folder (e.g. the collagen prediction mask, same level-0 size
                  as CE), collagen = mask > collagen_mask_threshold instead.
  per tile:
    CE Collagen Sum    sum of CE/255 over collagen pixels
    CE Collagen Mean   mean CE (0-255) over collagen pixels: how bright the
                       collagen is (CE counterpart of f_mean_intensity)
    CE Collagen Std    spread of CE over collagen pixels
    CE Outside Sum     sum of CE/255 over tissue pixels OUTSIDE the mask
  per slide:
    ce_collagen_density        sum(CE Collagen Sum) / tissue area: collagen
                               amount weighted by CE, haze excluded
    ce_collagen_mean_mean/_iqr tile mean and tile-to-tile IQR of CE
                               Collagen Mean (the CE counterpart of
                               f_mean_intensity_mean / f_mean_intensity_iqr)
    ce_collagen_std_mean/_iqr
    ce_outside_share           sum(CE Outside Sum) / sum(CE Sum): fraction
                               of the slide's CE signal outside collagen (QC)
  QC tile images get a third panel: CE outside the collagen mask (x3).

All v3 columns are kept unchanged, so v3 and v4 results can be compared.

v3 description follows.
---------------------------------------------------------------------
v3 (continuous CE): CE is a continuous image (the collagen prediction
combined with the DUET green signal), not a mask. v3 adds collagen features
computed DIRECTLY from the continuous CE values over tissue, with no
threshold, so collagen intensity and structure (thickness / texture
variation) are kept:

  per tile (over tissue pixels):
    CE Sum         sum of CE/255 (intensity-weighted collagen amount)
    CE Density     CE Sum / tissue pixels (0-1): continuous collagen ratio
    CE Std, CE P90 spread and upper range of CE values
    ce_contrast / ce_homogeneity / ce_energy / ce_correlation
                   GLCM texture of the continuous CE image over tissue
    fw_mean_intensity / fw_std_intensity
                   F green intensity weighted by CE (soft collagen weights)
  per slide:
    ce_density                 area-weighted over all tiles
    ce_density_median / _iqr   across interior tiles
    ce_<feature>_mean / _iqr   for the other per-tile CE and fw features

Everything from v2 is kept unchanged (binary-mask collagen_* features,
b_* and f_* features measured inside the thresholded collagen mask), so v2
and v3 results can be compared directly. Output columns: v2 columns + ce_* +
fw_*.

v2 description follows.
--------------------------------------------------------------------
Extracts CE-derived collagen geometry PLUS CE-masked B (H&E) and F (DUET
fluorescence) intensity/texture features, per tile and aggregated to slide
level for the multimodal combination analysis.

For each slide folder (e.g. Root/images/10H-0/) containing B.svs, F.svs,
and CE.svs:
1. Tissue is detected on the B (H&E) slide using a saturation-based Otsu
   threshold on a low-resolution thumbnail.
2. A grid of tile coordinates (level-0 pixel space) covering the detected
   tissue is generated.
3. B, F, and CE level-0 dimensions are checked for an exact match before any
   tiling happens — mismatched slide sets are skipped with an error message.
4. For every tissue tile coordinate, the SAME (x, y) region is read from B,
   F, and CE at level 0, guaranteeing pixel-for-pixel spatial alignment.
5. A level-0 tissue mask is computed for each tile from the B tile (same
   saturation criterion as step 1, threshold fixed per slide). This is the
   denominator of the collagen ratio, so glass in edge tiles no longer
   dilutes it. Lumens follow one rule regardless of the tile edge
   (lumen_mode), and CE-collagen pixels next to tissue are added to it.
6. The CE tile is binarized with ONE slide-level threshold (Otsu over
   tissue pixels sampled from across the slide; see ce_threshold_mode) and
   intersected with the tissue mask to obtain the collagen mask (or a fixed
   threshold, e.g. CE > 0 for a pre-extracted collagen image). Collagen
   geometry AND thickness come from that mask; B is color-deconvolved
   (skimage's rgb2hed, Ruifrok & Johnston stain separation) to isolate the
   Eosin channel — collagen is predominantly eosinophilic in H&E — and F is
   used as its green channel (the DUET fluorescence signal); both are summarized by intensity +
   GLCM texture statistics computed over collagen pixels only.
7. Per-tile features are written to
   ./data/tiles/<slide>_Tile_Collagen_Features.csv (patch-level labels for
   the later H&E distillation model).
8. Slide-level scores are aggregated:
      collagen_ratio      = sum(collagen_area) / sum(tissue_area), all tiles
      collagen_median     = median of per-tile collagen_area_ratio  (*)
      collagen_iqr        = IQR of per-tile collagen_area_ratio     (*)
      collagen_thickness_* = mean / median / IQR (across tiles) of each
                             tile's mean collagen strand thickness (see
                             Feature_Extractor_WSI.thickness_features) (*)
      b_/f_ features      = mean and IQR of each per-tile feature  (*)
   (*) over "interior" tiles only: Tissue Fraction >= min_tile_tissue_fraction
   (falls back to all tiles, flagged, if too few qualify). Per-tile ratios
   from mostly-glass tiles are noisy and were the main source of the
   tile-to-tile IQR.
   All slides are collected into ./data/collagen_scores.csv.

Folder structure expected:
Root/
└── images/
     ├── 10H-0/
     │    ├── B.svs
     │    ├── CE.svs
     │    └── F.svs
     └── 10H-1/
          └── ...

Timepoint convention: a trailing '-<digit>' in the folder name is parsed as
the timepoint ('10H-0' -> T0, '10H-1' -> T1); folders without a suffix
default to T0. Set "timepoint" in the config JSON to override for the whole
batch.

Usage (config-driven):
    python FeatureExtractor_wsi_v5.py --config ./config/feat_extract_wsi_v5.json

Config JSON keys (all optional unless noted):
    root_dir                  (required) root folder containing images/
    out_dir                   output folder, default "./data"
    tile_size                 default 3000
    stride                    default = tile_size
    tissue_threshold          default 0.10
    thumbnail_size            default 2048
    min_tissue_object_size    default 500
    save_qc_overlay           default true
    timepoint                 optional, overrides parsed timepoint label
    eosin_od_max              default 1.0 — OD value mapped to 255 when rescaling the
                              color-deconvolved Eosin channel used for B features (see
                              Feature_Extractor_WSI._deconvolve_eosin). A run-summary
                              warning is printed per slide if too many collagen-masked
                              pixels saturate at this value — raise it if so.
    ce_threshold_mode         "slide" (default): one Otsu threshold per slide over tissue
                              pixels sampled from ce_threshold_sample_tiles tiles, so every
                              tile is binarized consistently. "tile": Otsu per tile, but
                              over that tile's tissue pixels only (not glass). "fixed":
                              collagen = CE > ce_fixed_threshold — use with 0 if CE is a
                              pre-extracted collagen image whose non-collagen pixels are 0
                              (the per-slide log line "CE tissue px == 0" tells you).
    ce_fixed_threshold        default 0, used only when ce_threshold_mode == "fixed"
    ce_threshold_sample_tiles default 24
    min_tile_tissue_fraction  default 0.5 — tiles with less tissue than this still count
                              toward collagen_ratio (area-weighted) but are excluded from the
                              per-tile distribution stats (median / IQR / means). Applied at
                              aggregation only, so changing it does not force re-extraction.
    min_interior_tiles        default 5 — if fewer tiles pass min_tile_tissue_fraction,
                              distribution stats fall back to all tiles and the slide gets
                              interior_fallback = 1. With a narrow biopsy core, use a
                              smaller tile_size (e.g. 1000-1500) so enough tiles qualify.
    lumen_mode                "exclude" (default): denominator = stained tissue; all lumens left
                              out. "include": lumens up to max_lumen_diameter_um count as
                              tissue (cortical-area convention), computed on a padded read so
                              lumens cut by the tile edge are treated like enclosed ones.
    max_lumen_diameter_um     default 150 (include mode only)
    speckle_hole_um2          default 25 — unstained holes smaller than this are filled in
                              both modes (pale cytoplasm, not lumens)
    collagen_near_tissue_um   default 20 — CE-collagen within this distance of stained tissue
                              is kept and added to the tissue mask (pale collagen)
    n_qc_tiles                default 4 — per slide, save PNGs of the tiles with the lowest
                              and highest tissue fraction showing the tissue and collagen masks

Requirements:
pip install openslide-python scikit-image
(plus the system OpenSlide library, e.g. `apt install openslide-tools`
on Ubuntu/Debian, or `conda install -c conda-forge openslide`)

Re-running on the same output folder: each slide's per-tile CSV is skipped
and reused only if BOTH the feature-computation code version AND every
tiling/threshold/deconvolution parameter match the cached file exactly (see
Feature_Extractor_WSI._extraction_signature). Any change to those — including
edits to this script that change what a feature means while keeping the same
column name — invalidates the cache for that slide and it is re-extracted
automatically; you don't need to manually clear ./data/tiles/ between runs
with different settings.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.color import rgb2hed
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.morphology import (binary_closing, disk, remove_small_holes, remove_small_objects,
                                skeletonize)
from tqdm import tqdm

try:
    import openslide
except ImportError as e:
    raise ImportError(
        "This script requires openslide-python. Install with:\n"
        "  pip install openslide-python\n"
        "and make sure the system OpenSlide library is installed "
        "(e.g. `apt install openslide-tools` or "
        "`conda install -c conda-forge openslide`)."
    ) from e


# Bump this whenever a change to feature-computation LOGIC (not just
# tiling/threshold parameters, which are captured separately) would make an
# old per-tile CSV's columns mean something different than they used to,
# even though the column names haven't changed. Used in
# Feature_Extractor_WSI._extraction_signature() to detect stale cached
# per-tile CSVs on rerun. History:
#   v1              original C-only pipeline
#   v2-bf-texture   added B/F intensity + GLCM texture (CE-masked)
#   v3-eosin-deconv added collagen thickness; switched B from RGB-mean
#                   inversion to color-deconvolved Eosin channel
#   v4-tissue-denom per-tile tissue mask (from B) is now the ratio
#                   denominator (was: hole-filled collagen mask); CE
#                   threshold is slide-level Otsu over tissue pixels (was:
#                   per-tile Otsu over all pixels incl. glass); collagen
#                   mask restricted to tissue; B/F intensity + GLCM computed
#                   over collagen pixels only (masked-out zeros used to be
#                   included); collagen-free tissue tiles are kept (ratio 0);
#                   new per-tile 'Tissue Fraction' column
#   v5-lumen-rule   lumen handling no longer depends on where the tile edge
#                   falls (was: enclosed lumens filled, border-cut ones not):
#                   lumen_mode "exclude" (stained tissue only) or "include"
#                   (lumens up to max_lumen_diameter_um filled on a padded
#                   read); CE-collagen pixels near tissue are added to the
#                   tissue mask so pale collagen is never dropped
#   v6-continuous-ce adds collagen features from the continuous CE values
#                   (no threshold) and CE-weighted F intensity; v5 columns kept
#   v7-f-green      F features (f_*, fw_*) use the green channel of F (was:
#                   mean of R, G, B)
#   v8-ce-in-collagen CE inside / outside the collagen mask; optional
#                   collagen prediction mask file; v7 columns kept
#   v9-m-values     with a collagen_mask_file: features of the prediction
#                   map M itself (m_density, m_mean_*, m_collagen_mean_*)
TILE_SCHEMA_VERSION = "v9-m-values"


# =====================================================================
#                       WSI TILE FEATURE EXTRACTOR
# =====================================================================
class Feature_Extractor_WSI:
    """
    Tiles a whole-slide set on the fly, restricted to tissue regions
    detected on the B (H&E) thumbnail, and computes per-tile collagen
    geometry from the CE (collagen, glomeruli removed) slide.
    """

    def __init__(self,
                 slide_name: str,
                 b_path: Path,
                 f_path: Path,
                 ce_path: Path,
                 output_dir: Path,
                 tile_size: int = 3000,
                 stride: Optional[int] = None,
                 tissue_threshold: float = 0.10,
                 thumbnail_size: int = 2048,
                 min_tissue_object_size: int = 500,
                 save_qc_overlay: bool = True,
                 eosin_od_max: float = 1.0,
                 ce_threshold_mode: str = "slide",
                 ce_threshold_sample_tiles: int = 24,
                 ce_fixed_threshold: float = 0,
                 n_qc_tiles: int = 4,
                 lumen_mode: str = "exclude",
                 max_lumen_diameter_um: float = 150.0,
                 speckle_hole_um2: float = 25.0,
                 collagen_near_tissue_um: float = 20.0,
                 collagen_mask_path: Optional[Path] = None,
                 collagen_mask_threshold: float = 0):

        self.slide_name = slide_name
        self.b_path = b_path
        self.ce_path = ce_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tile_size = tile_size
        self.stride = stride or tile_size
        self.tissue_threshold = tissue_threshold
        self.thumbnail_size = thumbnail_size
        self.min_tissue_object_size = min_tissue_object_size
        self.save_qc_overlay = save_qc_overlay
        # OD value that maps to 255 when rescaling the deconvolved Eosin
        # channel (see _deconvolve_eosin). Tracked here (not a module
        # constant) so it's included in the extraction signature below —
        # changing it on a rerun correctly triggers re-extraction.
        self.eosin_od_max = eosin_od_max
        self._eosin_total_px = 0
        self._eosin_saturated_px = 0

        if ce_threshold_mode not in ("slide", "tile", "fixed"):
            raise ValueError(f"ce_threshold_mode must be 'slide', 'tile' or 'fixed', got {ce_threshold_mode!r}")
        self.ce_threshold_mode = ce_threshold_mode
        self.ce_fixed_threshold = ce_fixed_threshold
        self.ce_threshold_sample_tiles = ce_threshold_sample_tiles
        self.n_qc_tiles = n_qc_tiles
        if lumen_mode not in ("exclude", "include"):
            raise ValueError(f"lumen_mode must be 'exclude' or 'include', got {lumen_mode!r}")
        self.lumen_mode = lumen_mode
        self.max_lumen_diameter_um = max_lumen_diameter_um
        self.speckle_hole_um2 = speckle_hole_um2
        self.collagen_near_tissue_um = collagen_near_tissue_um
        # Set by _compute_tissue_mask (thumbnail saturation Otsu) and reused
        # for every level-0 tile so the tissue criterion is fixed per slide.
        self._saturation_thresh: Optional[float] = None
        # Set lazily in run() when ce_threshold_mode == "slide".
        self.ce_threshold: Optional[float] = None

        # ---- open slides ----
        self.b_slide = openslide.OpenSlide(str(self.b_path))
        self.f_slide = openslide.OpenSlide(str(f_path))
        self.ce_slide = openslide.OpenSlide(str(self.ce_path))
        # optional collagen prediction mask: collagen = mask > threshold
        self.collagen_mask_path = collagen_mask_path
        self.collagen_mask_threshold = collagen_mask_threshold
        self.mask_slide = (openslide.OpenSlide(str(collagen_mask_path))
                           if collagen_mask_path else None)

        # ---- mandatory dimension check (B vs F vs CE) ----
        self._check_dimensions()
        self.mpp = self._read_mpp()

        # ---- tissue detection + tile grid, from B only ----
        self._thumb_arr = None
        self._tissue_mask = None
        self.tile_coords: List[Tuple[int, int]] = self._find_tissue_tiles()

        if self.save_qc_overlay:
            self._save_qc_overlay()

    # ------------------- Extraction fingerprint (for resume/staleness check) -------------------
    def _extraction_signature(self) -> str:
        """
        Fingerprint of everything that changes what ends up IN the per-tile
        CSV: the feature-computation code version (TILE_SCHEMA_VERSION —
        bump this by hand whenever you change what a feature means, e.g.
        today's switch from RGB-invert to color-deconvolved Eosin for B)
        plus every tiling/thresholding parameter that affects which pixels
        get read and masked. Comparing this against a cached CSV's stored
        signature is what lets `run()` tell a genuinely reusable previous
        result apart from a stale one — matching column NAMES alone (the
        previous version of this check) can't detect either a code change
        that keeps the same column names, or a parameter change like a
        different tile_size on a rerun into the same output folder.
        """
        return (f"{TILE_SCHEMA_VERSION}"
                f"|tile={self.tile_size}"
                f"|stride={self.stride}"
                f"|tissue_thr={self.tissue_threshold}"
                f"|thumb={self.thumbnail_size}"
                f"|min_obj={self.min_tissue_object_size}"
                f"|eosin_od_max={self.eosin_od_max}"
                f"|ce_thr={self.ce_threshold_mode}:{self.ce_threshold_sample_tiles}"
                f":{self.ce_fixed_threshold}"
                f"|lumen={self.lumen_mode}:{self.max_lumen_diameter_um}:{self.speckle_hole_um2}"
                f"|near={self.collagen_near_tissue_um}"
                f"|mask={Path(self.collagen_mask_path).name if self.collagen_mask_path else None}"
                f":{self.collagen_mask_threshold}&ce>0")

    # ------------------- Cleanup -------------------
    def close(self):
        for s in (self.b_slide, self.f_slide, self.ce_slide, self.mask_slide):
            if s is None:
                continue
            try:
                s.close()
            except Exception:
                pass

    # ------------------- Dimension check -------------------
    def _check_dimensions(self):
        b_dim = self.b_slide.level_dimensions[0]
        f_dim = self.f_slide.level_dimensions[0]
        ce_dim = self.ce_slide.level_dimensions[0]

        if self.mask_slide is not None and self.mask_slide.level_dimensions[0] != ce_dim:
            raise ValueError(
                f"[DIMENSION MISMATCH] Slide set '{self.slide_name}': collagen mask "
                f"{self.mask_slide.level_dimensions[0]} vs CE {ce_dim} (level 0).")
        if not (b_dim == f_dim == ce_dim):
            raise ValueError(
                f"[DIMENSION MISMATCH] Slide set '{self.slide_name}': "
                f"B={b_dim} (level 0), F={f_dim} (level 0), CE={ce_dim} (level 0). "
                f"All three modalities must have identical level-0 pixel dimensions "
                f"for coordinate-aligned tiling to be valid."
            )
        self.level0_width, self.level0_height = b_dim

    def _read_mpp(self) -> float:
        """Level-0 microns per pixel of B, needed to express lumen/speckle
        sizes in µm. Falls back to 0.5 (20x Aperio) with a warning."""
        props = self.b_slide.properties
        for key in ("openslide.mpp-x", "tiffslide.mpp-x", "aperio.MPP"):
            try:
                v = float(props.get(key))
                if v > 0:
                    return v
            except (TypeError, ValueError):
                continue
        print(f"[WARNING] {self.slide_name}: MPP not found in slide metadata; assuming 0.5 µm/px")
        return 0.5

    # ------------------- Tissue detection (saturation-based Otsu on B) -------------------
    def _compute_tissue_mask(self, thumbnail_rgb: np.ndarray) -> np.ndarray:
        """
        Standard H&E tissue detection: convert to HSV and Otsu-threshold
        the saturation channel. Stained tissue has notably higher
        saturation than the (near-white, low-saturation) glass background.
        """
        hsv = cv2.cvtColor(thumbnail_rgb, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]

        try:
            thresh = threshold_otsu(saturation)
        except Exception:
            return np.zeros(saturation.shape, dtype=bool)
        self._saturation_thresh = float(thresh)

        tissue_mask = saturation > thresh
        tissue_mask = binary_fill_holes(tissue_mask)
        tissue_mask = binary_closing(tissue_mask, disk(2))
        tissue_mask = remove_small_objects(tissue_mask, min_size=self.min_tissue_object_size)
        return tissue_mask

    # Level-0 tissue masks are computed at 1/TILE_TISSUE_DOWNSAMPLE resolution
    # (morphology on a 3000x3000 mask is slow and unnecessary for a tissue
    # outline) and upsampled back with nearest-neighbour.
    TILE_TISSUE_DOWNSAMPLE = 4
    # Specks smaller than this (level-0 px) are dropped from the tile tissue
    # mask. min_tissue_object_size is in thumbnail px, a different scale.
    TILE_MIN_OBJECT_PX = 2000

    def _stained_mask_ds(self, b_rgb: np.ndarray, fill_holes_um2: float) -> np.ndarray:
        """
        Stained-tissue mask at 1/TILE_TISSUE_DOWNSAMPLE resolution: saturation
        above the slide's thumbnail threshold, with holes smaller than
        fill_holes_um2 filled. remove_small_holes fills by SIZE only, so an
        unstained region is treated the same whether or not it is enclosed.
        """
        ds = self.TILE_TISSUE_DOWNSAMPLE
        h, w = b_rgb.shape[:2]
        small = cv2.resize(b_rgb, (max(1, w // ds), max(1, h // ds)), interpolation=cv2.INTER_AREA)
        mask = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[:, :, 1] > self._saturation_thresh
        mask = binary_closing(mask, disk(1))
        mask = remove_small_objects(mask, min_size=max(1, self.TILE_MIN_OBJECT_PX // (ds * ds)))
        px_area_um2 = (self.mpp * ds) ** 2
        return remove_small_holes(mask, area_threshold=max(1, int(fill_holes_um2 / px_area_um2)))

    def tile_tissue_mask(self, x: int, y: int, b_tile: np.ndarray) -> np.ndarray:
        """
        Level-0 tissue mask (the ratio denominator) for the tile at (x, y).
        Lumens are handled by one rule everywhere, independent of where the
        tile edge falls:

        lumen_mode "exclude": stained tissue only. Every lumen — enclosed or
            cut by the tile edge — is left out; only speckle-sized unstained
            holes (< speckle_hole_um2, e.g. pale cytoplasm) are filled.
        lumen_mode "include": unstained regions up to max_lumen_diameter_um
            across are counted as tissue (as in cortical area for IF%). The
            mask is computed on a read padded by that diameter and cropped, so
            a lumen cut by the tile edge is seen whole and filled like an
            enclosed one; larger regions (tears, glass) stay excluded.
        """
        ds = self.TILE_TISSUE_DOWNSAMPLE
        T = self.tile_size
        if self._saturation_thresh is None:
            return np.ones(b_tile.shape[:2], dtype=bool)
        if self.lumen_mode == "exclude":
            mask = self._stained_mask_ds(b_tile, self.speckle_hole_um2)
            return cv2.resize(mask.astype(np.uint8), (T, T), interpolation=cv2.INTER_NEAREST).astype(bool)

        max_area_um2 = np.pi * (self.max_lumen_diameter_um / 2) ** 2
        m = int(np.ceil(self.max_lumen_diameter_um / self.mpp / ds)) * ds  # margin, multiple of ds
        x0, y0 = max(0, x - m), max(0, y - m)
        x1, y1 = min(self.level0_width, x + T + m), min(self.level0_height, y + T + m)
        padded = np.array(self.b_slide.read_region((x0, y0), 0, (x1 - x0, y1 - y0)).convert("RGB"))
        mask = self._stained_mask_ds(padded, max_area_um2)
        full = cv2.resize(mask.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        return full[y - y0:y - y0 + T, x - x0:x - x0 + T].astype(bool)

    def near_tissue(self, tissue: np.ndarray) -> np.ndarray:
        """Tissue dilated by collagen_near_tissue_um. CE-collagen pixels in this
        zone are added to the tissue mask: pale collagen has low H&E
        saturation and would otherwise fall out of both the numerator and the
        denominator. CE is 0 on glass, so far-away CE signal is not admitted."""
        r = max(1, int(round(self.collagen_near_tissue_um / self.mpp)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate(tissue.astype(np.uint8), kernel).astype(bool)

    # ------------------- Tile grid restricted to tissue -------------------
    def _find_tissue_tiles(self) -> List[Tuple[int, int]]:
        thumb = self.b_slide.get_thumbnail((self.thumbnail_size, self.thumbnail_size))
        thumb_arr = np.array(thumb.convert("RGB"))
        self._thumb_arr = thumb_arr

        tissue_mask = self._compute_tissue_mask(thumb_arr)
        self._tissue_mask = tissue_mask

        thumb_h, thumb_w = tissue_mask.shape
        scale_x = self.level0_width / thumb_w
        scale_y = self.level0_height / thumb_h

        tile_coords = []
        for y0 in range(0, self.level0_height - self.tile_size + 1, self.stride):
            ty0 = int(y0 / scale_y)
            ty1 = max(ty0 + 1, int((y0 + self.tile_size) / scale_y))
            for x0 in range(0, self.level0_width - self.tile_size + 1, self.stride):
                tx0 = int(x0 / scale_x)
                tx1 = max(tx0 + 1, int((x0 + self.tile_size) / scale_x))

                patch = tissue_mask[ty0:ty1, tx0:tx1]
                if patch.size == 0:
                    continue
                if patch.mean() >= self.tissue_threshold:
                    tile_coords.append((x0, y0))

        print(f"[INFO] {self.slide_name}: {len(tile_coords)} tissue tiles found "
              f"(tile_size={self.tile_size}, stride={self.stride}, "
              f"tissue_threshold={self.tissue_threshold})")
        return tile_coords

    # ------------------- QC overlay -------------------
    def _save_qc_overlay(self):
        """Save a thumbnail with the tissue mask and sampled tile grid drawn on it."""
        if self._thumb_arr is None:
            return

        overlay = self._thumb_arr.copy()
        thumb_h, thumb_w = self._tissue_mask.shape
        scale_x = self.level0_width / thumb_w
        scale_y = self.level0_height / thumb_h

        # tint tissue mask green
        mask_color = np.zeros_like(overlay)
        mask_color[..., 1] = (self._tissue_mask * 80).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 1.0, mask_color, 0.5, 0)

        # draw sampled tile boxes in red
        draw_coords = self.tile_coords[::max(1, len(self.tile_coords) // 500)]
        for (x0, y0) in draw_coords:
            tx0, ty0 = int(x0 / scale_x), int(y0 / scale_y)
            tx1 = int((x0 + self.tile_size) / scale_x)
            ty1 = int((y0 + self.tile_size) / scale_y)
            cv2.rectangle(overlay, (tx0, ty0), (tx1, ty1), (255, 0, 0), 1)

        out_path = self.output_dir / f"{self.slide_name}_tissue_qc.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    def _save_tile_qc(self, records: List[Dict], width: int = 600):
        """
        For the tiles with the lowest and highest Tissue Fraction, save
        [B with tissue outline incl. excluded holes (green) + collagen (red)] | [CE],
        so the tissue mask and CE threshold can be checked on edge vs
        interior tiles.
        """
        order = sorted(records, key=lambda r: r['Tissue Fraction'])
        k = max(1, self.n_qc_tiles // 2)
        picks = [('edge', r) for r in order[:k]] + [('interior', r) for r in order[-k:]]
        seen = set()
        for tag, r in picks:
            x, y = r['Tile X'], r['Tile Y']
            if (x, y) in seen:
                continue
            seen.add((x, y))
            b_tile = self._read_tile(self.b_slide, x, y)
            ce_tile = self._read_tile(self.ce_slide, x, y)
            tissue, collagen = self.tile_masks(x, y, b_tile, ce_tile)

            size = (width, width)
            b_small = cv2.resize(b_tile, size, interpolation=cv2.INTER_AREA)
            ce_small = cv2.resize(ce_tile, size, interpolation=cv2.INTER_AREA)
            col_small = cv2.resize(collagen.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
            tis_small = cv2.resize(tissue.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)

            overlay = b_small.copy()
            overlay[col_small] = (0.5 * overlay[col_small] + [127, 0, 0]).astype(np.uint8)
            # RETR_LIST: draw hole outlines too, so excluded lumens are visible
            contours, _ = cv2.findContours(tis_small, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 200, 0), 2)
            label = (f"{tag} tissue={r['Tissue Fraction']:.2f} "
                     f"ratio={r['Collagen Area Ratio']:.3f}")
            cv2.putText(overlay, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            # v4: CE outside the collagen mask, x3 so the haze is visible
            outside = ce_small.astype(np.float32)
            outside[col_small] = 0
            outside = np.clip(outside * 3, 0, 255).astype(np.uint8)
            cv2.putText(outside, "CE outside collagen mask (x3)", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            panel = np.hstack([overlay, ce_small, outside])
            out_path = self.output_dir / f"{self.slide_name}_qc_tile_{tag}_x{x}_y{y}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    def tile_masks(self, x: int, y: int, b_tile: np.ndarray, ce_tile: np.ndarray):
        """(tissue, collagen) for one tile. Collagen is CE above threshold
        within near_tissue(stained tissue); tissue = stained tissue (+ lumens
        per lumen_mode) UNION collagen, so collagen is always a subset of
        tissue and the ratio is <= 1."""
        tissue = self.tile_tissue_mask(x, y, b_tile)
        if self.mask_slide is not None:
            # v4: collagen from the prediction mask, same near-tissue rule.
            # CE > 0 keeps the glomeruli that were masked out of CE (CE = 0
            # there) out of the collagen mask too; M itself still has them.
            m = self._to_gray_uint8(self._read_tile(self.mask_slide, x, y))
            self._m_tile = m                                 # reused by m_features
            collagen = ((m > self.collagen_mask_threshold) & (self._to_gray_uint8(ce_tile) > 0)
                        & self.near_tissue(tissue))
        else:
            collagen = self.binarize(ce_tile, tissue, allowed=self.near_tissue(tissue)).astype(bool)
        return tissue | collagen, collagen

    # ------------------- Tile reading -------------------
    def _read_tile(self, slide: "openslide.OpenSlide", x: int, y: int) -> np.ndarray:
        """Read a tile at level-0 coordinates (x, y), native resolution."""
        tile = slide.read_region((x, y), 0, (self.tile_size, self.tile_size)).convert("RGB")
        return np.array(tile)

    # ------------------- Binarization -------------------
    @staticmethod
    def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
        img_gray = np.mean(image, axis=-1) if image.ndim == 3 else image.astype(np.float32)
        # decide by dtype, not by value: an 8-bit CE tile whose brightest pixel
        # is 1 (faint, near-empty tile) must not be rescaled to 0/255
        if np.issubdtype(image.dtype, np.floating) and img_gray.max() <= 1.0:
            return (img_gray * 255).astype(np.uint8)
        return img_gray.astype(np.uint8)

    @staticmethod
    def _otsu_or_none(values: np.ndarray) -> Optional[float]:
        """Otsu threshold of a 1-D pixel sample; None when there's no
        distinct bright class (constant input, or Otsu at the minimum)."""
        if values.size == 0 or values.min() == values.max():
            return None
        thresh = float(threshold_otsu(values))
        return None if thresh <= values.min() else thresh

    def compute_slide_ce_threshold(self, max_px_per_tile: int = 200_000, seed: int = 0) -> Optional[float]:
        """
        One CE threshold for the whole slide: Otsu over TISSUE pixels pooled
        from up to ce_threshold_sample_tiles tiles spread evenly over the
        tile grid. The old per-tile Otsu over all pixels split glass vs tissue
        in edge tiles but collagen vs non-collagen in interior tiles, so the
        same collagen got a different mask depending on where the tile sat.
        """
        if not self.tile_coords:
            return None
        n = min(self.ce_threshold_sample_tiles, len(self.tile_coords))
        idx = np.unique(np.linspace(0, len(self.tile_coords) - 1, n).astype(int))
        rng = np.random.default_rng(seed)
        samples = []
        for i in idx:
            x, y = self.tile_coords[i]
            tissue = self.tile_tissue_mask(x, y, self._read_tile(self.b_slide, x, y))
            vals = self._to_gray_uint8(self._read_tile(self.ce_slide, x, y))[tissue]
            if vals.size > max_px_per_tile:
                vals = rng.choice(vals, max_px_per_tile, replace=False)
            samples.append(vals)
        pooled = np.concatenate(samples) if samples else np.array([], dtype=np.uint8)
        thresh = self._otsu_or_none(pooled)
        print(f"[INFO] {self.slide_name}: slide-level CE threshold = {thresh} "
              f"(Otsu over {pooled.size} tissue px from {len(idx)} tiles)")
        if pooled.size:
            # If CE is a pre-extracted collagen image (non-collagen == 0),
            # most tissue px are exactly 0 and ce_threshold_mode="fixed" with
            # ce_fixed_threshold=0 (any non-zero px = collagen) is the right
            # choice; Otsu would then discard dim collagen.
            above = pooled[pooled > 0]
            print(f"[INFO] {self.slide_name}: CE tissue px == 0: {np.mean(pooled == 0):.1%}; "
                  f"non-zero px percentiles 5/50/95 = "
                  f"{np.percentile(above, [5, 50, 95]).round(1).tolist() if above.size else 'n/a'}; "
                  f"fraction of tissue px above Otsu = {np.mean(pooled > thresh) if thresh is not None else 0:.1%}, "
                  f"above 0 = {np.mean(pooled > 0):.1%}")
        return thresh

    def binarize(self, image: np.ndarray, tissue_mask: Optional[np.ndarray] = None,
                 allowed: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Collagen mask for a CE tile, restricted to `allowed` (default:
        tissue_mask). run() passes allowed = near_tissue(tissue) so pale
        collagen just outside the stained-tissue outline is kept.

        ce_threshold_mode == "slide": uses self.ce_threshold (set once per
        slide in run()). "tile": Otsu over this tile's tissue pixels only.
        Returns an all-zero mask when no threshold exists (no distinct
        bright class — e.g. dim tissue with no collagen signal), rather than
        masking the whole tile as collagen.
        """
        img_gray = self._to_gray_uint8(image)
        if tissue_mask is None:
            tissue_mask = np.ones(img_gray.shape, dtype=bool)

        if self.ce_threshold_mode == "slide":
            thresh = self.ce_threshold
        elif self.ce_threshold_mode == "fixed":
            thresh = self.ce_fixed_threshold
        else:
            thresh = self._otsu_or_none(img_gray[tissue_mask])
        if thresh is None:
            return np.zeros_like(img_gray, dtype=np.uint8)

        # strict '>' per skimage's Otsu convention (foreground is > thresh);
        # '>=' puts the whole lower class in the mask on near-binary CE images
        allowed = tissue_mask if allowed is None else allowed
        return ((img_gray > thresh) & allowed).astype(np.uint8)

    # ------------------- Color deconvolution (B / H&E) -------------------
    def _deconvolve_eosin(self, rgb_tile: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Color-deconvolve an H&E RGB tile via skimage's rgb2hed (Ruifrok &
        Johnston stain-separation matrix) and return the Eosin channel,
        rescaled to an approximate 0-255 range for compatibility with
        intensity_features/texture_features (GLCM needs discrete gray
        levels). Collagen in H&E is predominantly eosinophilic, so this
        targets collagen-relevant staining specifically, instead of the
        previous flat RGB-average-then-invert grayscale (which mixed in
        hematoxylin/nuclear staining and any other RGB variation equally).

        rgb2hed's raw output is in optical-density-like units, not 0-255,
        and can go slightly negative in near-unstained regions due to
        matrix-inversion noise — both are handled below (clip negatives to
        0, then linearly rescale [0, eosin_od_max] -> [0, 255]).

        self.eosin_od_max sets the OD value that maps to 255. The default
        (1.0) is a common practical ceiling for Eosin OD, but actual stain
        intensity varies by scanner, protocol, and staining batch — this is
        an assumption, not a calibrated constant for your data. If `mask`
        is given (the collagen mask, so saturation is tracked only where it
        matters), this method accumulates saturation stats that `run()`
        reports at the end of each slide; a high saturation fraction is
        your signal to raise eosin_od_max for this dataset.
        """
        hed = rgb2hed(rgb_tile)
        eosin_od = hed[..., 1]

        if mask is not None:
            region = eosin_od[mask.astype(bool)]
            if region.size:
                self._eosin_total_px += int(region.size)
                self._eosin_saturated_px += int(np.sum(region >= self.eosin_od_max))

        eosin_od = np.clip(eosin_od, 0, self.eosin_od_max)
        return (eosin_od / self.eosin_od_max * 255).astype(np.float32)

    # ------------------- Geometry features -------------------
    def global_features(self, bin_image: np.ndarray, tissue_mask: np.ndarray) -> Dict[str, float]:
        """
        Collagen geometry for one tile. The denominator is the tile's tissue
        area (from B), not the tile area and not the hole-filled collagen
        mask (the pre-v4 denominator, which ignored tissue not enclosed by
        collagen and made edge-tile ratios depend on mask topology).

        'Tissue Area' includes the collagen pixels (collagen is a subset of
        tissue, see tile_masks), so Collagen Area Ratio = Collagen Area /
        Tissue Area. What counts as tissue for lumens is set by lumen_mode.
        """
        collagen_area = int(np.sum(bin_image))
        tissue_area = int(np.sum(tissue_mask))
        return {
            'Collagen Area': collagen_area,
            'Tissue Area': tissue_area,
            'Tissue Fraction': tissue_area / tissue_mask.size,
            'Collagen Area Ratio': collagen_area / tissue_area if tissue_area > 0 else np.nan,
        }

    # ------------------- Thickness features -------------------
    def thickness_features(self, bin_image: np.ndarray) -> Dict[str, float]:
        """
        Per-tile collagen strand thickness, via distance transform + skeleton:
        for every pixel on the mask's morphological skeleton (medial axis),
        local thickness ~= 2 * (distance to nearest background pixel) at
        that point.

        This is a standard, cheap proxy for local thickness in irregular,
        non-tubular structures like collagen deposits — it is NOT the exact
        maximum-inscribed-sphere "local thickness" used in bone morphometry
        (e.g. BoneJ), which requires a more expensive per-pixel search.
        Treat these values as a RELATIVE / comparative thickness measure
        across tiles and slides (e.g. for ranking or as a model feature),
        not a calibrated micron measurement, unless you additionally
        multiply by the slide's microns-per-pixel (MPP) — level-0 MPP is
        not currently read from the .svs metadata in this script.

        Returns NaN for all three stats on an empty mask, or on a mask too
        small/thin for skeletonize to identify any medial-axis pixel.
        """
        if np.sum(bin_image) == 0:
            return {'Collagen Thickness Mean': np.nan,
                    'Collagen Thickness Median': np.nan,
                    'Collagen Thickness IQR': np.nan}

        dist = distance_transform_edt(bin_image)
        skeleton = skeletonize(bin_image.astype(bool))
        thickness_px = 2.0 * dist[skeleton]

        if thickness_px.size == 0:
            return {'Collagen Thickness Mean': np.nan,
                    'Collagen Thickness Median': np.nan,
                    'Collagen Thickness IQR': np.nan}

        return {
            'Collagen Thickness Mean': float(np.mean(thickness_px)),
            'Collagen Thickness Median': float(np.median(thickness_px)),
            'Collagen Thickness IQR': float(np.percentile(thickness_px, 75) -
                                            np.percentile(thickness_px, 25)),
        }

    # ------------------- Intensity features (CE-masked) -------------------
    def intensity_features(self, gray_image: np.ndarray, mask: np.ndarray, prefix: str) -> Dict[str, float]:
        """
        Intensity statistics over collagen pixels only. (Pre-v4 this averaged
        the whole masked tile, zeros included, so the "mean intensity" was
        really intensity x collagen coverage and dropped in edge tiles.)
        """
        vals = gray_image[mask.astype(bool)].astype(float)
        if vals.size == 0:
            return {f'{prefix}_mean_intensity': np.nan,
                    f'{prefix}_std_intensity': np.nan}
        return {
            f'{prefix}_mean_intensity': float(np.mean(vals)),
            f'{prefix}_std_intensity': float(np.std(vals)),
        }

    # ------------------- Texture features (CE-masked) -------------------
    def texture_features(self, gray_image: np.ndarray, mask: np.ndarray, prefix: str) -> Dict[str, float]:
        """
        GLCM texture over collagen pixels only. Masked pixels are encoded as
        gray level 0 and in-mask values shifted into 1..255; GLCM row/column 0
        is then zeroed so only pairs of adjacent collagen pixels count. (Pre-v4
        the masked-out zeros were counted, so texture mostly measured how
        much of the tile was masked.) graycoprops renormalizes the matrix.
        """
        nan_out = {f'{prefix}_contrast': np.nan, f'{prefix}_homogeneity': np.nan,
                   f'{prefix}_energy': np.nan, f'{prefix}_correlation': np.nan}
        mask = mask.astype(bool)
        if not mask.any():
            return nan_out
        coded = np.zeros(gray_image.shape, dtype=np.uint8)
        coded[mask] = 1 + np.clip(gray_image[mask], 0, 254).astype(np.uint8)
        glcm = graycomatrix(coded, [1], [0], levels=256, symmetric=True, normed=False).astype(np.float64)
        glcm[0, :] = 0
        glcm[:, 0] = 0
        if glcm.sum() == 0:
            return nan_out
        return {
            f'{prefix}_contrast': float(graycoprops(glcm, 'contrast')[0, 0]),
            f'{prefix}_homogeneity': float(graycoprops(glcm, 'homogeneity')[0, 0]),
            f'{prefix}_energy': float(graycoprops(glcm, 'energy')[0, 0]),
            f'{prefix}_correlation': float(graycoprops(glcm, 'correlation')[0, 0]),
        }

    # ------------------- Continuous CE features (no threshold) -------------------
    def ce_features(self, ce_tile: np.ndarray, f_green: np.ndarray, tissue: np.ndarray,
                    collagen: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Collagen features from the continuous CE image over TISSUE pixels,
        without binarising: every tissue pixel contributes in proportion to its
        CE value, so faint and bright collagen, and the gradient across a
        strand, are all kept.

        CE Sum / CE Density   amount of collagen signal (CE/255 summed; per
                              tissue pixel). Continuous counterpart of
                              Collagen Area / Collagen Area Ratio.
        CE Std / CE P90       spread and upper range of CE in the tile
        ce_* texture          GLCM of CE over tissue pixels (same encoding as
                              texture_features: non-tissue pixels excluded)
        fw_* intensity        F green weighted by CE/255: the soft-weighted
                              counterpart of f_mean_intensity
        CE Collagen Sum/Mean/Std  CE over the collagen mask pixels only:
                              amount, brightness and spread of collagen CE
                              (Mean is the CE counterpart of f_mean_intensity;
                              CE Sum / Collagen Area is NOT, because a large
                              share of CE lies outside the mask)
        CE Outside Sum        CE/255 summed over tissue outside the mask (QC)
        """
        t = tissue.astype(bool)
        ce = self._to_gray_uint8(ce_tile).astype(np.float32)
        out = {"CE Sum": np.nan, "CE Density": np.nan, "CE Std": np.nan, "CE P90": np.nan,
               "CE Collagen Sum": 0.0, "CE Collagen Mean": np.nan, "CE Collagen Std": np.nan,
               "CE Outside Sum": np.nan, "fw_mean_intensity": np.nan, "fw_std_intensity": np.nan}
        c = np.zeros_like(t) if collagen is None else collagen.astype(bool)
        if c.any():
            inside = ce[c]
            out["CE Collagen Sum"] = float(inside.sum() / 255.0)
            out["CE Collagen Mean"] = float(inside.mean())
            out["CE Collagen Std"] = float(inside.std())
        out["CE Outside Sum"] = float(ce[t & ~c].sum() / 255.0)
        if not t.any():
            out.update(self.texture_features(ce, t, "ce"))
            return out
        v = ce[t]
        w = v / 255.0
        out["CE Sum"] = float(w.sum())
        out["CE Density"] = float(w.mean())
        out["CE Std"] = float(v.std())
        out["CE P90"] = float(np.percentile(v, 90))
        if w.sum() > 0:
            fv = f_green[t].astype(np.float64)
            m = float(np.sum(w * fv) / w.sum())
            out["fw_mean_intensity"] = m
            out["fw_std_intensity"] = float(np.sqrt(np.sum(w * (fv - m) ** 2) / w.sum()))
        out.update(self.texture_features(ce, t, "ce"))
        return out

    @staticmethod
    def m_features(m: np.ndarray, ce_tile: np.ndarray, tissue: np.ndarray,
                   collagen: np.ndarray) -> Dict[str, float]:
        """Features of the collagen prediction map M itself (sigmoid output,
        stored 0-255): no fluorescence intensity enters them.

        Glomeruli are masked out of CE but not of M; pixels with M >= 0.25 and
        CE = 0 are taken as masked out (if CE = M x green, CE rounds to 0 there
        only when green < 2) and excluded.
        M Sum / M Valid Area  sum of M (0-1) and pixel count over tissue
                              minus masked-out pixels
        M Mean                M Sum / M Valid Area: soft collagen fraction
        M Collagen Mean       mean M (0-1) over the collagen mask: model
                              confidence inside collagen
        M Masked Out Area     masked-out (glomerulus) pixels, QC
        """
        ce = Feature_Extractor_WSI._to_gray_uint8(ce_tile)
        t = tissue.astype(bool)
        masked_out = t & (ce == 0) & (m >= 64)
        valid = t & ~masked_out
        p = m.astype(np.float64) / 255.0
        out = {"M Sum": float(p[valid].sum()), "M Valid Area": float(valid.sum()),
               "M Mean": float(p[valid].mean()) if valid.any() else np.nan,
               "M Collagen Mean": float(p[collagen].mean()) if collagen.any() else np.nan,
               "M Masked Out Area": float(masked_out.sum())}
        return out

    # ------------------- Main runner -------------------
    def run(self) -> pd.DataFrame:
        self.output_file_path = self.output_dir / f"{self.slide_name}_Tile_Collagen_Features.csv"
        signature = self._extraction_signature()

        # resume: skip only if the existing per-tile CSV was produced with
        # the exact same schema version AND extraction parameters (see
        # _extraction_signature). This catches both a code change that
        # keeps the same column names (e.g. today's B-channel change below)
        # and a parameter change (different tile_size, tissue_threshold,
        # eosin_od_max, etc.) on a rerun into the same output folder —
        # either would otherwise be silently missed by a column-name-only
        # check.
        if self.output_file_path.exists():
            existing = pd.read_csv(self.output_file_path)
            existing_sig = (existing['_extraction_signature'].iloc[0]
                             if '_extraction_signature' in existing.columns and len(existing)
                             else None)
            if existing_sig == signature:
                print(f"[INFO] {self.slide_name}: output file already exists with matching "
                      f"extraction settings, skipping: {self.output_file_path}")
                return existing
            print(f"[INFO] {self.slide_name}: existing per-tile CSV is stale "
                  f"(old={existing_sig!r}, current={signature!r}) — re-extracting.")

        if self.ce_threshold_mode in ("slide", "fixed"):
            # also run in "fixed" mode: logs the CE intensity diagnostics
            self.ce_threshold = self.compute_slide_ce_threshold()
            if self.ce_threshold_mode == "slide" and self.ce_threshold is None:
                print(f"[WARNING] {self.slide_name}: no slide-level CE threshold "
                      f"(no distinct bright class in tissue); all collagen masks will be empty.")

        records = []
        for (x, y) in tqdm(self.tile_coords, total=len(self.tile_coords), desc=self.slide_name):
            b_tile = self._read_tile(self.b_slide, x, y)
            ce_tile = self._read_tile(self.ce_slide, x, y)
            tissue, collagen = self.tile_masks(x, y, b_tile, ce_tile)
            # the grid is chosen on the thumbnail; at level 0 a sliver tile
            # can turn out to contain no tissue at all
            if not tissue.any():
                continue
            bin_image = collagen.astype(np.uint8)

            # Collagen-free tissue tiles are KEPT (ratio 0): dropping them
            # biases the slide ratio and IQR upward.
            tile_features = self.global_features(bin_image, tissue)
            tile_features.update(self.thickness_features(bin_image))

            # F (DUET fluorescence): green channel only (the fluorescence
            # signal; R/B are not averaged in), read for every tile (v3: also
            # needed for the CE-weighted fw_* features)
            f_green = self._read_tile(self.f_slide, x, y)[..., 1].astype(np.float32)
            tile_features.update(self.ce_features(ce_tile, f_green, tissue, bin_image))
            if self.mask_slide is not None:
                tile_features.update(self.m_features(self._m_tile, ce_tile, tissue, collagen))
            if bin_image.any():
                # B (H&E): color-deconvolved Eosin channel (targets
                # collagen-relevant eosinophilic staining specifically, rather
                # than overall RGB darkness — see _deconvolve_eosin)
                b_gray = self._deconvolve_eosin(b_tile, mask=bin_image)
            else:
                b_gray = np.zeros(bin_image.shape, dtype=np.float32)

            # all four are computed over collagen pixels only (NaN if none)
            tile_features.update(self.intensity_features(b_gray, bin_image, 'b'))
            tile_features.update(self.texture_features(b_gray, bin_image, 'b'))
            tile_features.update(self.intensity_features(f_green, bin_image, 'f'))
            tile_features.update(self.texture_features(f_green, bin_image, 'f'))

            tile_features['Slide Name'] = self.slide_name
            tile_features['Tile X'] = x
            tile_features['Tile Y'] = y
            records.append(tile_features)

        if self.save_qc_overlay and records and self.n_qc_tiles > 0:
            self._save_tile_qc(records)

        if self._eosin_total_px > 0:
            sat_frac = self._eosin_saturated_px / self._eosin_total_px
            if sat_frac > 0.05:
                print(f"[WARNING] {self.slide_name}: {sat_frac:.1%} of collagen-masked "
                      f"Eosin-channel pixels saturated at eosin_od_max={self.eosin_od_max}. "
                      f"b_mean_intensity may be compressed for this slide — consider raising "
                      f"eosin_od_max in the config.")

        if records:
            feature_df = pd.DataFrame.from_records(records)
            feature_df['_extraction_signature'] = signature
            feature_df.to_csv(self.output_file_path, index=False)
            print(f"[INFO] {self.slide_name}: wrote {len(records)} tiles with collagen signal "
                  f"-> {self.output_file_path}")
        else:
            feature_df = pd.DataFrame()
            print(f"[WARNING] {self.slide_name}: no tiles with collagen signal found; no CSV written.")

        return feature_df


# =====================================================================
#                    SLIDE-LEVEL AGGREGATION
# =====================================================================
def aggregate_slide(tile_df: pd.DataFrame,
                    min_tile_tissue_fraction: float = 0.5,
                    min_interior_tiles: int = 5) -> Optional[Dict[str, float]]:
    """
    Aggregate per-tile collagen geometry, thickness, and B/F imaging
    features into slide-level scores.

    collagen_ratio is area-weighted over ALL tiles: sum of collagen pixels
    over sum of tissue pixels. Edge tiles contribute in proportion to the
    tissue they contain, so they need no special handling here.

    Every per-tile DISTRIBUTION statistic (collagen_median/iqr, thickness
    mean/median/iqr, B/F mean and IQR) uses only "interior" tiles with
    Tissue Fraction >= min_tile_tissue_fraction. A tile that is 90% glass
    gives a per-tile ratio from a small, boundary-dominated tissue area; those
    noisy values were what drove collagen_iqr. If fewer than
    min_interior_tiles qualify, all tiles are used and interior_fallback = 1
    (reduce tile_size for narrow cores rather than relying on this).

    collagen_thickness_* summarizes each tile's mean strand thickness (a
    single scalar per tile, from Feature_Extractor_WSI.thickness_features).
    """
    if tile_df is None or len(tile_df) == 0:
        return None

    total_collagen = float(tile_df['Collagen Area'].sum())
    total_tissue = float(tile_df['Tissue Area'].sum())
    collagen_ratio = total_collagen / total_tissue if total_tissue > 0 else np.nan

    interior = tile_df[tile_df['Tissue Fraction'] >= min_tile_tissue_fraction]
    fallback = len(interior) < min_interior_tiles
    dist_df = tile_df if fallback else interior

    ratios = dist_df['Collagen Area Ratio'].to_numpy(dtype=float)
    ratios = ratios[~np.isnan(ratios)]

    out = {
        'collagen_ratio': collagen_ratio,
        'collagen_median': float(np.median(ratios)) if len(ratios) else np.nan,
        'collagen_iqr': (float(np.percentile(ratios, 75) - np.percentile(ratios, 25))
                         if len(ratios) else np.nan),
        'n_tiles_total': int(len(tile_df)),
        'n_tiles_interior': int(len(interior)),
        'n_tiles_collagen_positive': int((tile_df['Collagen Area'] > 0).sum()),
        'mean_tile_tissue_fraction': float(tile_df['Tissue Fraction'].mean()),
        'interior_fallback': int(fallback),
    }

    # slide-level collagen thickness: mean/median/IQR across tiles' per-tile
    # mean thickness. NaN-tile guard mirrors the B/F block below.
    if 'Collagen Thickness Mean' in dist_df.columns:
        thick_vals = dist_df['Collagen Thickness Mean'].to_numpy(dtype=float)
        thick_vals = thick_vals[~np.isnan(thick_vals)]
        if len(thick_vals):
            out['collagen_thickness_mean'] = float(np.mean(thick_vals))
            out['collagen_thickness_median'] = float(np.median(thick_vals))
            out['collagen_thickness_iqr'] = float(np.percentile(thick_vals, 75) -
                                                   np.percentile(thick_vals, 25))

    # v3: continuous CE features
    if 'CE Sum' in tile_df.columns:
        out['ce_density'] = (float(tile_df['CE Sum'].sum() / total_tissue)
                             if total_tissue > 0 else np.nan)
        dens = dist_df['CE Density'].to_numpy(dtype=float)
        dens = dens[~np.isnan(dens)]
        if len(dens):
            out['ce_density_median'] = float(np.median(dens))
            out['ce_density_iqr'] = float(np.percentile(dens, 75) - np.percentile(dens, 25))
        if 'CE Collagen Sum' in tile_df.columns:
            ce_total = float(tile_df['CE Sum'].sum())
            out['ce_collagen_density'] = (float(tile_df['CE Collagen Sum'].sum() / total_tissue)
                                          if total_tissue > 0 else np.nan)
            out['ce_outside_share'] = (float(tile_df['CE Outside Sum'].sum() / ce_total)
                                       if ce_total > 0 else np.nan)
        for col, name in [('CE Std', 'ce_std'), ('CE P90', 'ce_p90'),
                          ('CE Collagen Mean', 'ce_collagen_mean'),
                          ('CE Collagen Std', 'ce_collagen_std'),
                          ('ce_contrast', 'ce_contrast'), ('ce_homogeneity', 'ce_homogeneity'),
                          ('ce_energy', 'ce_energy'), ('ce_correlation', 'ce_correlation'),
                          ('fw_mean_intensity', 'fw_mean_intensity'),
                          ('fw_std_intensity', 'fw_std_intensity')]:
            vals = dist_df[col].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals):
                out[f'{name}_mean'] = float(np.mean(vals))
                out[f'{name}_iqr'] = float(np.percentile(vals, 75) - np.percentile(vals, 25))

    # v5: collagen prediction map M (only with a collagen_mask_file)
    if 'M Sum' in tile_df.columns:
        valid = float(tile_df['M Valid Area'].sum())
        out['m_density'] = float(tile_df['M Sum'].sum() / valid) if valid > 0 else np.nan
        out['m_masked_out_share'] = (float(tile_df['M Masked Out Area'].sum() / total_tissue)
                                     if total_tissue > 0 else np.nan)
        for col, name in [('M Mean', 'm_mean'), ('M Collagen Mean', 'm_collagen_mean')]:
            vals = dist_df[col].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals):
                out[f'{name}_mean'] = float(np.mean(vals))
                out[f'{name}_iqr'] = float(np.percentile(vals, 75) - np.percentile(vals, 25))

    # slide-level B/F features: mean and IQR across (interior) tiles
    for prefix in ('b', 'f'):
        for feat in ['mean_intensity', 'std_intensity', 'contrast',
                     'homogeneity', 'energy', 'correlation']:
            col = f'{prefix}_{feat}'
            if col in dist_df.columns:
                vals = dist_df[col].to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    out[f'{col}_mean'] = float(np.mean(vals))
                    out[f'{col}_iqr'] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    return out


def parse_timepoint(slide_name: str, override: Optional[str] = None) -> str:
    """'10H-0' -> T0, '10H-1' -> T1; no suffix -> T0. override wins if given."""
    if override:
        return override
    m = re.search(r'-(\d+)$', slide_name)
    return f"T{m.group(1)}" if m else "T0"


# =====================================================================
#                    BATCH PROCESSING LOGIC
# =====================================================================
def find_bfce_sets(root_dir: Path, mask_file: Optional[str] = None) -> List[Dict[str, Path]]:
    """
    Match B/CE whole-slide images by looking for B.svs, F.svs, CE.svs
    directly inside each slide subfolder.

    Expected layout:
        Root/
        ├── 10H-0/
        │    ├── B.svs
        │    ├── CE.svs
        │    └── F.svs
        └── 10H-1/
             └── ...
    """
    sets = []
    missing = []

    for slide_dir in sorted(root_dir.iterdir()):
        if not slide_dir.is_dir() or slide_dir.name.startswith("Features_"):
            continue

        b_svs = slide_dir / "B.svs"
        f_svs = slide_dir / "F.svs"
        ce_svs = slide_dir / "CE.svs"

        mask = slide_dir / mask_file if mask_file else None
        if b_svs.exists() and f_svs.exists() and ce_svs.exists() and (mask is None or mask.exists()):
            sets.append({"name": slide_dir.name,
                         "B": b_svs,
                         "F": f_svs,
                         "CE": ce_svs,
                         "mask": mask})
        else:
            missing.append(slide_dir.name)

    if missing:
        print(f"[WARNING] {len(missing)} slide folder(s) missing B/F/CE .svs"
              f"{' or ' + mask_file if mask_file else ''}: {missing}")

    if not sets:
        raise ValueError(f"No complete B/F/CE slide sets found under {root_dir}")

    print(f"[INFO] Found {len(sets)} matched B/F/CE slide set(s)")
    return sets


def main(args):
    root_dir = Path(args.root_dir)
    sets = find_bfce_sets(root_dir / "images", getattr(args, "collagen_mask_file", None))

    out_dir = Path(getattr(args, "out_dir", "./data"))
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    slide_scores = []
    for s in sets:
        print(f"\n================ Processing Slide: {s['name']} ================")

        try:
            extractor = Feature_Extractor_WSI(
                slide_name=s["name"],
                b_path=s["B"],
                f_path=s["F"],
                ce_path=s["CE"],
                output_dir=tiles_dir,
                tile_size=getattr(args, "tile_size", 3000),
                stride=getattr(args, "stride", None),
                tissue_threshold=getattr(args, "tissue_threshold", 0.10),
                thumbnail_size=getattr(args, "thumbnail_size", 2048),
                min_tissue_object_size=getattr(args, "min_tissue_object_size", 500),
                save_qc_overlay=getattr(args, "save_qc_overlay", True),
                eosin_od_max=getattr(args, "eosin_od_max", 1.0),
                ce_threshold_mode=getattr(args, "ce_threshold_mode", "slide"),
                ce_threshold_sample_tiles=getattr(args, "ce_threshold_sample_tiles", 24),
                ce_fixed_threshold=getattr(args, "ce_fixed_threshold", 0),
                n_qc_tiles=getattr(args, "n_qc_tiles", 4),
                lumen_mode=getattr(args, "lumen_mode", "exclude"),
                max_lumen_diameter_um=getattr(args, "max_lumen_diameter_um", 150.0),
                speckle_hole_um2=getattr(args, "speckle_hole_um2", 25.0),
                collagen_near_tissue_um=getattr(args, "collagen_near_tissue_um", 20.0),
                collagen_mask_path=s["mask"],
                collagen_mask_threshold=getattr(args, "collagen_mask_threshold", 0),
            )
        except ValueError as e:
            print(f"[ERROR] Skipping '{s['name']}': {e}")
            continue

        start_time = time.time()
        tile_df = extractor.run()
        extractor.close()
        elapsed = time.time() - start_time
        print(f"[INFO] Finished {s['name']} in {int(elapsed // 60)}:{int(elapsed % 60):02d} min")

        score = aggregate_slide(tile_df,
                                min_tile_tissue_fraction=getattr(args, "min_tile_tissue_fraction", 0.5),
                                min_interior_tiles=getattr(args, "min_interior_tiles", 5))
        if score is not None and score['interior_fallback']:
            print(f"[WARNING] {s['name']}: only {score['n_tiles_interior']} tile(s) with tissue "
                  f"fraction >= min_tile_tissue_fraction; distribution stats use all tiles. "
                  f"Consider a smaller tile_size.")
        if score is not None:
            score['slide_name'] = s['name']
            score['timepoint'] = parse_timepoint(s['name'], getattr(args, "timepoint", None))
            slide_scores.append(score)

    if slide_scores:
        scores_df = pd.DataFrame(slide_scores)
        # Lead with slide_name/timepoint; keep every other produced column
        # (collagen_*, thickness_*, b_*, f_*, and anything added later)
        # rather than a fixed whitelist, so new features are never silently
        # dropped here.
        lead_cols = ['slide_name', 'timepoint']
        other_cols = sorted(c for c in scores_df.columns if c not in lead_cols)
        scores_df = scores_df[lead_cols + other_cols]
        out_path = out_dir / "collagen_scores.csv"
        scores_df.to_csv(out_path, index=False)
        print(f"\n[INFO] Wrote slide-level scores for {len(scores_df)} slide(s) -> {out_path}")
    else:
        print("\n[WARNING] No slide-level scores aggregated; collagen_scores.csv not written.")


# =====================================================================
#                            ENTRY POINT
# =====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="WSI-based Batch Collagen Feature Extraction (v5, prediction map M)")
    parser.add_argument('--config', type=str, default="./config/feat_extract_wsi_v5.json",
                        help='Path to config JSON')
    cli_args, _ = parser.parse_known_args()
    with open(cli_args.config) as f:
        data = json.load(f)
    run_args = argparse.Namespace(**data)
    main(run_args)