"""
AlphaEarth: Google Satellite Embedding V1 -> alphaearth_imgfeat.npy per city.

For each GEOID:
  1. Read tract shapefile  ({shp_dir}/{GEOID}/*.shp)
  2. Fetch GEE hex embeddings for the county boundary (2019-2021 mean)
  3. Aggregate: hex centroids spatial-joined to tracts -> mean of hex embeddings per tract
  4. Save [N_tracts, 64] -> {out_dir}/{GEOID}/nfeat/alphaearth_imgfeat.npy

Row order matches shapefile row order (= nfeat.npy row order).

Run from AlphaEarth/ directory:
    python build_imgfeat.py \
        --shp_dir  /path/to/US_countyfile_TractBorder \
        --out_dir  data/US \
        --ref_dir  ../RemoteCLIP/data/US \
        --ee_project ee-yourproject
"""

import argparse
import glob
import io
import math
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from tqdm import tqdm


# Constants
SAT_COLLECTION                 = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EE_HIGH_VOLUME_URL             = "https://earthengine-highvolume.googleapis.com"
N_BANDS                        = 64
MAX_WORKERS                    = 8
MAX_TILE_BYTES                 = 48 * 1024 * 1024
COMPUTE_PIXELS_BYTES_PER_VALUE = 10
CHECKPOINT_EVERY               = 500
DATE_START                     = "2019"
DATE_END                       = "2021"

H3_EDGE_LENGTH_M = {
    4: 22_606, 5: 8_545, 6: 3_229, 7: 1_220,
    8:    461, 9:   174, 10:  65,  11:   25,
}


# GEE tile helpers
def _resolution_scale(resolution: int) -> float:
    return max(10.0, H3_EDGE_LENGTH_M.get(resolution, 174) / 4.0)


def _assign_pixels_to_hexes(H, W, minx, miny, maxx, maxy, resolution):
    import h3
    lons = np.linspace(minx, maxx, W, endpoint=False) + (maxx - minx) / (2 * W)
    lats = np.linspace(maxy, miny, H, endpoint=False) - (maxy - miny) / (2 * H)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    assign = np.vectorize(h3.latlng_to_cell)
    return assign(lat_grid.ravel(), lon_grid.ravel(), resolution)


def _aggregate_tile(arr, pixel_hex_ids):
    band_names = arr.dtype.names
    valid = ~np.isnan(arr[band_names[0]].ravel())
    if not valid.any():
        return None, None, None
    ids_valid = pixel_hex_ids[valid]
    unique_hexes, local_inv = np.unique(ids_valid, return_inverse=True)
    n_local = len(unique_hexes)
    band_sums = np.array([
        np.bincount(local_inv,
                    weights=arr[b].ravel()[valid].astype(np.float64),
                    minlength=n_local)
        for b in band_names
    ]).T
    pixel_counts = np.bincount(local_inv, minlength=n_local)
    return unique_hexes, band_sums, pixel_counts


def _fetch_tile(image, tx0, ty1, scale_deg, tw, th, max_retries=6):
    import ee
    request = {
        "expression": image,
        "fileFormat": "NPY",
        "grid": {
            "dimensions": {"width": tw, "height": th},
            "affineTransform": {
                "scaleX":  scale_deg, "shearX": 0, "translateX": tx0,
                "shearY":  0, "scaleY": -scale_deg, "translateY": ty1,
            },
            "crsCode": "EPSG:4326",
        },
    }
    for attempt in range(max_retries):
        try:
            data = ee.data.computePixels(request)
            return np.load(io.BytesIO(data))
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # 1, 2, 4, 8, 16, 32 seconds
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _fetch_hex_embeddings(boundary_geom, ee_project, resolution=9,
                          scale=None, workers=MAX_WORKERS,
                          ckpt_path=None, resume=True):
    """
    Fetch GEE Satellite Embedding V1 for boundary_geom.
    Returns dict {hex_id: np.array shape (64,) float32}.
    """
    import ee
    ee.Initialize(project=ee_project, opt_url=EE_HIGH_VOLUME_URL)

    hull_coords = list(boundary_geom.convex_hull.exterior.coords)
    aoi = ee.Geometry.Polygon([[list(c) for c in hull_coords]])

    image = (
        ee.ImageCollection(SAT_COLLECTION)
        .filterDate(f"{DATE_START}-01-01", f"{DATE_END}-12-31")
        .filterBounds(aoi)
        .mean()
    )

    if scale is None:
        scale = _resolution_scale(resolution)
    scale_deg = scale / 111_320.0

    minx, miny, maxx, maxy = boundary_geom.bounds
    full_W = max(1, math.ceil((maxx - minx) / scale_deg))
    full_H = max(1, math.ceil((maxy - miny) / scale_deg))

    all_bands = image.bandNames().getInfo()
    tile_px = max(32, int(math.sqrt(
        MAX_TILE_BYTES / (len(all_bands) * COMPUTE_PIXELS_BYTES_PER_VALUE)
    )))

    n_tiles_x = math.ceil(full_W / tile_px)
    n_tiles_y = math.ceil(full_H / tile_px)
    total_tiles = n_tiles_x * n_tiles_y

    sums, counts, completed_tiles = {}, {}, set()
    if resume and ckpt_path and Path(ckpt_path).exists():
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        if ckpt.get("total_tiles") == total_tiles:
            sums = ckpt["sums"]
            counts = ckpt["counts"]
            completed_tiles = ckpt["completed"]

    lock = threading.Lock()
    done_count = [len(completed_tiles)]

    def process_tile(ix, iy):
        if (ix, iy) in completed_tiles:
            return
        tx0 = minx + ix * tile_px * scale_deg
        ty1 = maxy - iy * tile_px * scale_deg
        tw  = min(tile_px, full_W - ix * tile_px)
        th  = min(tile_px, full_H - iy * tile_px)
        if tw <= 0 or th <= 0:
            return
        tx1 = tx0 + tw * scale_deg
        ty0 = ty1 - th * scale_deg

        arr = _fetch_tile(image, tx0, ty1, scale_deg, tw, th)
        tile_hex_ids = _assign_pixels_to_hexes(th, tw, tx0, ty0, tx1, ty1, resolution)
        unique_hexes, band_sums, pixel_counts = _aggregate_tile(arr, tile_hex_ids)
        if unique_hexes is None:
            return

        with lock:
            for i, hid in enumerate(unique_hexes):
                if hid in sums:
                    sums[hid]   += band_sums[i]
                    counts[hid] += int(pixel_counts[i])
                else:
                    sums[hid]   = band_sums[i].copy()
                    counts[hid] = int(pixel_counts[i])
            completed_tiles.add((ix, iy))
            done_count[0] += 1
            if ckpt_path and done_count[0] % CHECKPOINT_EVERY == 0:
                tmp = str(ckpt_path) + ".tmp"
                with open(tmp, "wb") as f:
                    pickle.dump({"sums": sums, "counts": counts,
                                 "completed": completed_tiles,
                                 "total_tiles": total_tiles}, f, protocol=4)
                os.replace(tmp, ckpt_path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(process_tile, ix, iy)
            for iy in range(n_tiles_y)
            for ix in range(n_tiles_x)
        ]
        with tqdm(total=total_tiles, initial=len(completed_tiles),
                  desc="  Tiles", unit="tile", leave=False) as pbar:
            for f in as_completed(futures):
                f.result()
                pbar.update(1)

    hex_embeddings = {
        hid: (sums[hid] / cnt).astype(np.float32)
        for hid, cnt in counts.items() if cnt > 0
    }

    if ckpt_path and Path(ckpt_path).exists():
        os.remove(ckpt_path)

    return hex_embeddings


# Tract-level aggregation
def _hex_embeddings_to_tract_features(tract_gdf, hex_embeddings, resolution=9):
    """
    Aggregate hex embeddings to tract level via centroid-within spatial join.
    Returns np.array [n_tracts, 64] float32, ordered by tract_gdf row order.
    Tracts with no hex centroid inside get zeros.
    """
    import h3

    if not hex_embeddings:
        return np.zeros((len(tract_gdf), N_BANDS), dtype=np.float32)

    hex_ids = list(hex_embeddings.keys())
    centroids = []
    for hid in hex_ids:
        lat, lng = h3.cell_to_latlng(hid)
        centroids.append(Point(lng, lat))

    hex_gdf = gpd.GeoDataFrame({"hex_id": hex_ids}, geometry=centroids, crs="EPSG:4326")

    if tract_gdf.crs is None:
        tract_gdf = tract_gdf.set_crs(epsg=4326)
    elif tract_gdf.crs.to_epsg() != 4326:
        tract_gdf = tract_gdf.to_crs(epsg=4326)

    tract_gdf_reset = tract_gdf.reset_index(drop=True).copy()
    tract_gdf_reset["_tract_i"] = np.arange(len(tract_gdf_reset))

    joined = gpd.sjoin(hex_gdf, tract_gdf_reset[["_tract_i", "geometry"]],
                       how="left", predicate="within")

    feat_matrix = np.zeros((len(tract_gdf), N_BANDS), dtype=np.float32)
    for local_i in range(len(tract_gdf)):
        matched = joined[joined["_tract_i"] == local_i]["hex_id"].tolist()
        if matched:
            vecs = np.stack([hex_embeddings[hid] for hid in matched], axis=0)
            feat_matrix[local_i] = vecs.mean(axis=0)

    return feat_matrix


# Per-city pipeline
def process_city(geoid, shp_dir, out_dir, ee_project, resolution, workers):
    out_path  = os.path.join(out_dir, geoid, "nfeat", "alphaearth_imgfeat.npy")
    ckpt_path = os.path.join(out_dir, geoid, "nfeat", ".alphaearth_checkpoint.pkl")

    if os.path.exists(out_path):
        return "skip"

    shp_files = glob.glob(os.path.join(shp_dir, geoid, "**", "*.shp"), recursive=True)
    if not shp_files:
        print(f"  [WARN] no .shp for {geoid}")
        return "fail"

    tract_gdf = gpd.read_file(shp_files[0])
    if len(tract_gdf) == 0:
        print(f"  [WARN] empty shapefile for {geoid}")
        return "fail"

    if tract_gdf.crs is None:
        tract_gdf = tract_gdf.set_crs(epsg=4326)
    elif tract_gdf.crs.to_epsg() != 4326:
        tract_gdf = tract_gdf.to_crs(epsg=4326)

    county_boundary = unary_union(tract_gdf.geometry)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    hex_embeddings = _fetch_hex_embeddings(
        boundary_geom=county_boundary,
        ee_project=ee_project,
        resolution=resolution,
        workers=workers,
        ckpt_path=ckpt_path,
        resume=True,
    )
    print(f"  {len(hex_embeddings)} hexes fetched")

    feat_matrix = _hex_embeddings_to_tract_features(tract_gdf, hex_embeddings, resolution)

    n_zero = (feat_matrix.sum(axis=1) == 0).sum()
    if n_zero > 0:
        print(f"  [WARN] {n_zero}/{len(tract_gdf)} tracts had no hex centroid inside")

    np.save(out_path, feat_matrix)
    print(f"  Saved {feat_matrix.shape} -> {out_path}")
    return "ok"


# Main
def main():
    parser = argparse.ArgumentParser(
        description="AlphaEarth: GEE Satellite Embedding V1 -> alphaearth_imgfeat.npy"
    )
    parser.add_argument("--shp_dir",    required=True,
                        help="Base dir with {GEOID}/*.shp per city")
    parser.add_argument("--out_dir",    required=True,
                        help="Output base: {out_dir}/{GEOID}/nfeat/alphaearth_imgfeat.npy")
    parser.add_argument("--ref_dir",    required=True,
                        help="Dir with valid GEOIDs to process (e.g. RemoteCLIP/data/US/)")
    parser.add_argument("--ee_project", required=True,
                        help="Google Earth Engine project ID (e.g. ee-yourname)")
    parser.add_argument("--resolution", type=int, default=9,
                        help="H3 resolution (default: 9)")
    parser.add_argument("--workers",    type=int, default=MAX_WORKERS,
                        help=f"Parallel tile workers (default: {MAX_WORKERS})")
    args = parser.parse_args()

    geoids = sorted([
        d for d in os.listdir(args.ref_dir)
        if os.path.isdir(os.path.join(args.ref_dir, d))
    ])
    print(f"Found {len(geoids)} cities (from ref_dir).")

    ok = skip = fail = 0
    for i, geoid in enumerate(geoids, 1):
        print(f"\n[{i}/{len(geoids)}] {geoid}")
        try:
            result = process_city(geoid, args.shp_dir, args.out_dir,
                                  args.ee_project, args.resolution, args.workers)
        except Exception as e:
            print(f"  [ERROR] {geoid}: {e}")
            result = "fail"
        if result == "ok":
            ok += 1
        elif result == "skip":
            skip += 1
            print("  already exists, skipping")
        else:
            fail += 1

    print(f"\nDone. ok={ok}  skipped={skip}  failed={fail}")
    print("Feature dim: 64  (img_dim in config = 66 after worldpop concat)")


if __name__ == "__main__":
    main()
