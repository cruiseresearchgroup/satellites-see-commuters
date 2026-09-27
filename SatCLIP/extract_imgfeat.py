"""
SatCLIP ViT-S/16 satellite image feature extraction.
Saves satclip_imgfeat.npy [N_regions, 384] per city into {out_dir}/{GEOID}/nfeat/satclip_imgfeat.npy

Run from SatCLIP/ directory:
    python extract_imgfeat.py \
        --shp_dir /path/to/shapefiles \
        --out_dir data/US \
        --ref_dir ../RemoteCLIP/data/US \
        --token $ESRI_TOKEN \
        --model_name microsoft/SatCLIP-ViT16-L40
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
from transformers import CLIPModel, CLIPProcessor

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


# SatCLIP feature extraction
def load_model(model_name, device):
    # Processor: standard ViT-B/16 CLIP processor (224px, 3-channel RGB)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Load SatCLIP visual encoder from .ckpt
    # Checkpoint is ViT-S/16 (embed_dim=384, 12 blocks) trained on 13-band Sentinel-2.
    import timm
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(model_name, "satclip-vit16-l40.ckpt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    visual_sd = {
        k.replace("model.visual.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.visual.")
    }

    # patch_embed.proj.weight is [384, 13, 16, 16] (13 Sentinel-2 bands).
    # Adapt to 3-channel RGB by averaging across spectral bands then repeating.
    pe_w = visual_sd["patch_embed.proj.weight"]        # [384, 13, 16, 16]
    pe_mean = pe_w.mean(dim=1, keepdim=True)            # [384,  1, 16, 16]
    visual_sd["patch_embed.proj.weight"] = pe_mean.repeat(1, 3, 1, 1)  # [384, 3, 16, 16]

    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0, in_chans=3)
    # use manual attention, fused attention fails on some GPUs
    for m in model.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
    model.load_state_dict(visual_sd, strict=False)
    model = model.to(device).eval()
    return model, processor


def extract_imgfeats(model, processor, regional_imgs, device):
    """
    Returns [N_regions, 384] CLS token features.
    """
    region_ids = sorted(regional_imgs.keys())
    feats = []
    use_amp = device.type == "cuda"

    for rid in tqdm(region_ids, desc="  Extracting features"):
        img_io = regional_imgs[rid]
        img_io.seek(0)
        image = Image.open(img_io).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    feat = model(pixel_values)  # [1, 1024]
            else:
                feat = model(pixel_values)

            feats.append(feat.squeeze(0).float().cpu().numpy())

    return np.stack(feats, axis=0)


# Per-city processing
def process_city(geoid, shp_dir, out_dir, token, model, processor, device):
    out_path = os.path.join(out_dir, geoid, "nfeat", "satclip_imgfeat.npy")
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

    feats = extract_imgfeats(model, processor, regional_imgs, device)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, feats)
    return "ok"


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shp_dir",    required=True)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--ref_dir",    required=True,
                        help="Dir with valid GEOIDs to process (e.g. RemoteCLIP/data/US/)")
    parser.add_argument("--token",      required=True)
    parser.add_argument("--model_name", default="microsoft/SatCLIP-ViT16-L40")
    args = parser.parse_args()

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
            model, processor, device
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
