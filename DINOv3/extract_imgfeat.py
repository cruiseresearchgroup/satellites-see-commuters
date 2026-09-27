"""
DINOv3 satellite image feature extraction.
Saves dino_imgfeat.npy [N_regions, 1024] per city into {out_dir}/{GEOID}/nfeat/dino_imgfeat.npy

Run from DINOv3/ directory:
    python extract_imgfeat.py \
        --shp_dir /path/to/shapefiles \
        --out_dir data/US \
        --ref_dir ../RemoteCLIP/data/US \
        --token $ESRI_TOKEN \
        --model_name facebook/dinov3-vitl16-pretrain-sat493m
"""

import os
import glob
import argparse
import time
import requests
from io import BytesIO
from multiprocessing import Pool

import numpy as np
import torch
from tqdm import tqdm
import geopandas as gpd
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb check for large county canvases
from scipy.ndimage import binary_fill_holes
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import login

from utils import XY2deg, get_YX_area, lon_lat_to_pixel


# Tile download + stitch  (same logic as RemoteCLIP pipeline)
def _download_one_tile(args):
    X, Y, token = args
    base_url = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/41468/15/"
    url = base_url + str(Y) + "/" + str(X) + "?token=" + token
    filename = str(Y) + "_" + str(X)
    for attempt in range(2):
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return filename, BytesIO(response.content)
        except Exception:
            if attempt == 0:
                time.sleep(1)
    return filename  # failed


def _download_all_tiles(Y_X, token, num_proc=50):
    tile_args = [(int(X), int(Y), token) for Y, X in [t.split("_") for t in Y_X]]
    cached, failed = {}, []
    with Pool(processes=num_proc) as pool:
        for result in tqdm(pool.imap(_download_one_tile, tile_args),
                           total=len(Y_X), desc="  Downloading tiles"):
            if isinstance(result, tuple):
                cached[result[0]] = result[1]
            else:
                failed.append(result)
    return cached, failed


def _concat_one_region(args):
    region, tiles, files = args

    ys, xs = zip(*[(int(t.split("_")[0]), int(t.split("_")[1])) for t in tiles])
    row_min, col_min, row_max, col_max = min(ys), min(xs), max(ys), max(xs)
    mask_w = col_max - col_min + 1
    mask_h = row_max - row_min + 1
    canvas = np.zeros([mask_h * 256, mask_w * 256, 3], dtype=np.int32)
    pixel_coords = []

    for tile in tiles:
        if tile not in files:
            continue
        try:
            img_temp = Image.open(files[tile])
            row, col = int(tile.split("_")[0]), int(tile.split("_")[1])
            minlat, minlon = XY2deg(col, row, 15)
            maxlat, maxlon = XY2deg(col + 1, row + 1, 15)
            geom = region.iloc[0].geometry
            polys = [geom] if geom.geom_type != "MultiPolygon" else list(geom.geoms)
            for poly in polys:
                for coord in poly.exterior.coords:
                    pixel_coords.append(lon_lat_to_pixel(
                        coord, minlon, maxlon, minlat, maxlat,
                        col, col_min, row, row_min, mask_w, mask_h))
            canvas[(row - row_min)*256:(row - row_min + 1)*256,
                   (col - col_min)*256:(col - col_min + 1)*256, :] = img_temp
        except Exception:
            continue

    if canvas.sum() == 0:
        return int(region.index[0])

    im = np.uint8(canvas)
    im_zero = Image.new("L", (mask_w * 256, mask_h * 256), 0)
    draw = ImageDraw.Draw(im_zero)
    for i in range(len(pixel_coords)):
        draw.line([pixel_coords[i], pixel_coords[(i + 1) % len(pixel_coords)]],
                  fill="red", width=5)
    filled = binary_fill_holes(im_zero)
    im = im * filled.reshape(*filled.shape, 1)
    img_mem = BytesIO()
    Image.fromarray(np.uint8(im)).save(img_mem, format="PNG")
    img_mem.seek(0)
    return int(region.index[0]), img_mem


def _concat_all_regions(region_list, area_shp, cached_tiles, num_proc=20):
    region_list = sorted(region_list)
    args = [
        (area_shp[area_shp.index == i],
         area_shp.loc[area_shp.index == i, "Y_X"].item(),
         {k: cached_tiles[k] for k in area_shp.loc[area_shp.index == i, "Y_X"].item()
          if k in cached_tiles})
        for i in region_list
    ]
    imgs, failed = {}, []
    for arg in tqdm(args, desc="  Stitching regions"):
        result = _concat_one_region(arg)
        if isinstance(result, tuple):
            imgs[result[0]] = result[1]
        else:
            failed.append(result)
    return imgs, failed


def download_area_sateimgs(area_shp, token, num_proc=50):
    if area_shp.crs is None:
        area_shp = area_shp.set_crs(epsg=4326)
    elif area_shp.crs.to_string() != "EPSG:4326":
        area_shp = area_shp.to_crs(epsg=4326)

    # uses proper sjoin intersection (not just bounding box)
    area_shp, Y_X = get_YX_area(area_shp)

    cached = {}
    remaining = Y_X
    while remaining:
        new_tiles, remaining = _download_all_tiles(remaining, token, num_proc)
        cached.update(new_tiles)
        if remaining:
            print(f"  {len(remaining)} tiles failed, retrying...")

    regional_imgs = {}
    remaining = area_shp.index.to_list()
    while remaining:
        new_imgs, remaining = _concat_all_regions(remaining, area_shp, cached, num_proc)
        regional_imgs.update(new_imgs)

    return regional_imgs


# DINOv3 feature extraction
def load_model(model_name, device):
    # gated HF models need HF_TOKEN set
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager").to(device)
    model.eval()
    return model, processor


def extract_imgfeats(model, processor, regional_imgs, device, use_cls_token=True):
    """
    Returns [N_regions, feat_dim] in ascending region-index order
    (matches nfeat.npy row order).
    """
    region_ids = sorted(regional_imgs.keys())
    feats = []
    use_amp = device.type == "cuda"

    for rid in tqdm(region_ids, desc="  Extracting features"):
        img_io = regional_imgs[rid]
        img_io.seek(0)
        image = Image.open(img_io).convert("RGB")
        inputs = {k: v.to(device) for k, v in
                  processor(images=image, return_tensors="pt").items()}

        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            if use_cls_token:
                feat = outputs.last_hidden_state[:, 0, :]            # CLS token [1, 1024]
            else:
                feat = outputs.last_hidden_state[:, 1:, :].mean(dim=1)  # mean patch tokens

            feats.append(feat.squeeze(0).float().cpu().numpy())

    return np.stack(feats, axis=0)  # [N_regions, 1024]


# Per-city processing
def process_city(geoid, shp_dir, out_dir, token, model, processor, device, use_cls_token):
    out_path = os.path.join(out_dir, geoid, "nfeat", "dino_imgfeat.npy")
    if os.path.exists(out_path):
        return "skip"

    shp_files = glob.glob(os.path.join(shp_dir, geoid, "**", "*.shp"), recursive=True)
    if not shp_files:
        print(f"  [WARN] no .shp found for {geoid}")
        return "fail"

    area_shp = gpd.read_file(shp_files[0])

    try:
        regional_imgs = download_area_sateimgs(area_shp, token)
    except Exception as e:
        print(f"  [WARN] {geoid} tile download failed: {e}")
        return "fail"

    if not regional_imgs:
        print(f"  [WARN] {geoid} produced no images")
        return "fail"

    feats = extract_imgfeats(model, processor, regional_imgs, device, use_cls_token)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, feats)
    return "ok"


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shp_dir",    required=True,
                        help="Base dir with {GEOID}/*.shp per city")
    parser.add_argument("--out_dir",    required=True,
                        help="Output base: {out_dir}/{GEOID}/nfeat/imgfeat.npy")
    parser.add_argument("--ref_dir",    required=True,
                        help="Dir with valid GEOIDs to process (e.g. RemoteCLIP/data/US/)")
    parser.add_argument("--token",      required=True,
                        help="Esri World Imagery token")
    parser.add_argument("--model_name", default="facebook/dinov3-vitl16-pretrain-sat493m",
                        help="HuggingFace model ID")
    parser.add_argument("--use_cls_token", dest="use_cls_token",
                        action="store_true",  default=True)
    parser.add_argument("--no_use_cls_token", dest="use_cls_token", action="store_false")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model: {args.model_name}")
    model, processor = load_model(args.model_name, device)
    print("Model loaded.")

    all_geoids = sorted([
        d for d in os.listdir(args.ref_dir)
        if os.path.isdir(os.path.join(args.ref_dir, d))
    ])
    print(f"Found {len(all_geoids)} cities (from ref_dir). Pre-scanning tile counts...")
    def _tile_count(geoid):
        shp_files = glob.glob(os.path.join(args.shp_dir, geoid, "**", "*.shp"), recursive=True)
        if not shp_files:
            return 0
        try:
            _, Y_X = get_YX_area(gpd.read_file(shp_files[0]))
            return len(Y_X)
        except Exception:
            return 0
    tile_counts = {g: _tile_count(g) for g in all_geoids}
    geoids = sorted(all_geoids, key=lambda g: tile_counts[g])
    print(f"Sorted {len(geoids)} cities by tile count (smallest first).")
    print(f"Found {len(geoids)} cities (from ref_dir).")

    ok = skip = fail = 0
    for i, geoid in enumerate(geoids, 1):
        print(f"\n[{i}/{len(geoids)}] {geoid}")
        result = process_city(
            geoid, args.shp_dir, args.out_dir, args.token,
            model, processor, device, args.use_cls_token
        )
        if result == "ok":
            ok += 1
        elif result == "skip":
            skip += 1
            print("  already exists, skipping")
        else:
            fail += 1

    print(f"\nDone. ok={ok}  skipped={skip}  failed={fail}")


if __name__ == "__main__":
    main()
