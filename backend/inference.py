"""
GAN-SCA 3-Band inference core.

This module preserves the EXACT neural network and sliding-window inference
logic from the original `predict.py` (the single source of truth). The only
changes versus the CLI script are structural, so the same code can be driven
by a web server:

  * `rich` console output is replaced by an optional `on_stage()` progress
    callback (so the API can report live progress to the browser).
  * `run_prediction()` returns a summary dict instead of only printing it.
  * `load_model()` caches the loaded model so the 125 MB checkpoint is read
    once, not on every request.

The numerical pipeline -- percentile normalization, Gaussian-weighted
sliding window, FP16/FP32 handling, 0.5 threshold, RGBA mask, GeoJSON
vectorization with CRS preservation -- is unchanged from the original.
"""

import os
import json
import argparse
import tempfile

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import torch
import torch.nn as nn


# =========================================================================
# Neural Network Architecture (GeneratorSCA)  -- verbatim from predict.py
# =========================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class SCABlock(nn.Module):
    def __init__(self, in_channels):
        super(SCABlock, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class GeneratorSCA(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(GeneratorSCA, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.sca_bot = SCABlock(1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        self.sca_up1 = SCABlock(512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        self.outc = nn.Sequential(nn.Conv2d(64, out_channels, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        bn = self.down4(x4)
        bn = self.sca_bot(bn)

        u1 = self.up1(bn)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.conv_up1(u1)
        u1 = self.sca_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv_up3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.conv_up4(u4)
        return self.outc(u4)


# =========================================================================
# Device + model management
# =========================================================================

_MODEL_CACHE = {}  # path -> loaded GeneratorSCA (on the resolved device)


def get_device():
    """Return the best available torch device (CUDA if present, else CPU)."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device('cuda')
    return torch.device('cpu')


def get_device_info():
    """Human-readable device description for the /api/health endpoint."""
    if torch.cuda.is_available():
        return {"type": "cuda", "name": torch.cuda.get_device_name(0)}
    return {"type": "cpu", "name": "CPU"}


def load_model(model_path, device=None):
    """
    Load (and cache) the GeneratorSCA generator weights.

    The checkpoint is a plain state_dict of the 3-band generator. Loading is
    cached per path so repeated predictions reuse the in-memory model.
    """
    if device is None:
        device = get_device()
    key = os.path.abspath(model_path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model = GeneratorSCA(in_channels=3, out_channels=1).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def model_is_loaded(model_path):
    return os.path.abspath(model_path) in _MODEL_CACHE


# =========================================================================
# Inference engine helpers  -- verbatim numerical logic from predict.py
# =========================================================================

def generate_gaussian_weights(patch_size):
    center = patch_size // 2
    y, x = np.ogrid[-center:center, -center:center]
    sigma = patch_size / 4.0
    weight = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return weight.astype(np.float32)


def _estimate_percentiles(src, n_samples=100, patch_size=256, bands=3):
    """
    Estimate per-band 2nd/98th percentiles by reading random small windows
    from the source raster. Never loads the full image.
    """
    h, w = src.height, src.width
    rng = np.random.default_rng(42)
    samples = {c: [] for c in range(bands)}

    for _ in range(n_samples):
        ry = rng.integers(0, max(1, h - patch_size))
        rx = rng.integers(0, max(1, w - patch_size))
        win = Window(rx, ry, min(patch_size, w - rx), min(patch_size, h - ry))
        data = src.read(indexes=list(range(1, bands + 1)), window=win).astype(np.float32)
        for c in range(bands):
            samples[c].append(data[c].ravel())

    percentiles = []
    for c in range(bands):
        all_px = np.concatenate(samples[c])
        p2, p98 = np.percentile(all_px, (2, 98))
        percentiles.append((float(p2), float(p98)))
    return percentiles


def _noop(stage, frac=None):
    pass


def run_prediction(model, img_path, out_tif_path, out_geojson_path,
                   patch_size=256, batch_size=None, threshold=0.5,
                   building_color=(255, 0, 0), device=None, on_stage=None):
    """
    Memory-efficient sliding-window Gaussian-weighted inference.

    Identical numerical behaviour to the original `predict_large_image`;
    console output is replaced by the `on_stage(stage, frac)` callback and a
    summary dict is returned.

    Returns
    -------
    dict with keys: building_count, width, height, crs_epsg, crs_raw
    """
    if on_stage is None:
        on_stage = _noop
    if device is None:
        device = get_device()

    # Ensure output directories exist
    for p in (out_tif_path, out_geojson_path):
        if p:
            d = os.path.dirname(p)
            if d:
                os.makedirs(d, exist_ok=True)

    use_cuda = (device.type == 'cuda')

    on_stage("preparing", None)
    with rasterio.open(img_path) as src:
        h, w = src.height, src.width
        bands = min(src.count, 3)
        src_crs = src.crs
        src_transform = src.transform
        src_meta = src.meta.copy()

    # -- Step 1: Estimate percentiles from random windows -----------------
    on_stage("normalizing", None)
    with rasterio.open(img_path) as src:
        percentiles = _estimate_percentiles(src, n_samples=100, patch_size=patch_size, bands=bands)

    # -- Step 2: Create temporary memory-mapped accumulators --------------
    tmp_dir = tempfile.mkdtemp(prefix="gansca_pred_")
    pred_sum_path = os.path.join(tmp_dir, "pred_sum.dat")
    weight_sum_path = os.path.join(tmp_dir, "weight_sum.dat")

    pred_sum = np.memmap(pred_sum_path, dtype='float32', mode='w+', shape=(h, w))
    weight_sum = np.memmap(weight_sum_path, dtype='float32', mode='w+', shape=(h, w))

    gaussian_weights = generate_gaussian_weights(patch_size)
    stride = patch_size // 2

    y_steps = list(range(0, h, stride))
    x_steps = list(range(0, w, stride))

    patch_coords = []
    for y in y_steps:
        for x in x_steps:
            rh = min(patch_size, h - y)
            rw = min(patch_size, w - x)
            patch_coords.append((y, x, rh, rw))

    BATCH = batch_size if batch_size is not None else (32 if use_cuda else 4)
    model.eval()

    if use_cuda:
        model.half()  # FP16 weights
        batch_buf = torch.zeros(BATCH, bands, patch_size, patch_size,
                                dtype=torch.float16, pin_memory=True)
    else:
        model.float()  # FP32 weights on CPU
        batch_buf = torch.zeros(BATCH, bands, patch_size, patch_size,
                                dtype=torch.float32, pin_memory=False)

    # -- Step 3: Batched sliding-window inference -------------------------
    on_stage("inference", 0.0)
    total_batches = max(1, (len(patch_coords) + BATCH - 1) // BATCH)
    done_batches = 0

    with rasterio.open(img_path) as src:
        with torch.no_grad():
            for batch_start in range(0, len(patch_coords), BATCH):
                batch_slice = patch_coords[batch_start: batch_start + BATCH]
                cur_bs = len(batch_slice)

                # -- Read + normalize patches into buffer ------
                for bi, (py, px, rh, rw) in enumerate(batch_slice):
                    win = Window(px, py, rw, rh)
                    patch = src.read(
                        indexes=list(range(1, bands + 1)), window=win,
                    ).astype(np.float32)

                    for c in range(bands):
                        p2, p98 = percentiles[c]
                        if p98 > p2:
                            np.subtract(patch[c], p2, out=patch[c])
                            np.divide(patch[c], (p98 - p2), out=patch[c])
                            np.clip(patch[c], 0.0, 1.0, out=patch[c])
                        else:
                            patch[c] = 0.0

                    batch_buf[bi] = 0  # zero padding region
                    if use_cuda:
                        batch_buf[bi, :, :rh, :rw] = torch.from_numpy(patch).half()
                    else:
                        batch_buf[bi, :, :rh, :rw] = torch.from_numpy(patch).float()

                # -- Forward pass --------------------------------
                batch_tensor = batch_buf[:cur_bs].to(device, non_blocking=True)
                if use_cuda:
                    with torch.amp.autocast('cuda'):
                        preds = model(batch_tensor)
                    preds_np = preds.float().squeeze(1).cpu().numpy()
                else:
                    preds = model(batch_tensor)
                    preds_np = preds.squeeze(1).cpu().numpy()

                # -- Scatter results back to accumulators -------------
                for bi, (py, px, rh, rw) in enumerate(batch_slice):
                    pred_np = preds_np[bi]
                    pred_sum[py:py + rh, px:px + rw] += pred_np[:rh, :rw] * gaussian_weights[:rh, :rw]
                    weight_sum[py:py + rh, px:px + rw] += gaussian_weights[:rh, :rw]

                done_batches += 1
                on_stage("inference", done_batches / total_batches)

    model.float()  # restore FP32 weights

    # -- Step 4: Finalize probability map -> binary mask -------------------
    on_stage("writing_mask", None)
    np.clip(weight_sum, 1e-8, None, out=weight_sum)
    np.divide(pred_sum, weight_sum, out=pred_sum)  # reuse pred_sum as probability

    # -- Step 5: Write RGBA TIF in chunks (memory-safe) -------------------
    R, G, B = building_color
    out_meta = src_meta.copy()
    out_meta.update({
        'count': 4,
        'dtype': 'uint8',
        'driver': 'GTiff',
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'compress': 'deflate',
    })

    CHUNK = 2048  # rows at a time
    with rasterio.open(out_tif_path, 'w', **out_meta) as dst:
        for row_start in range(0, h, CHUNK):
            row_end = min(row_start + CHUNK, h)
            chunk_h = row_end - row_start

            prob_chunk = pred_sum[row_start:row_end, :]
            binary = (prob_chunk > threshold).astype(np.uint8)

            rgba = np.zeros((4, chunk_h, w), dtype=np.uint8)
            rgba[0][binary == 1] = R
            rgba[1][binary == 1] = G
            rgba[2][binary == 1] = B
            rgba[3][binary == 1] = 200

            win = Window(0, row_start, w, chunk_h)
            dst.write(rgba, window=win)

    # -- Step 6: Vectorise to GeoJSON (process in chunks) -----------------
    on_stage("vectorizing", None)
    features = []

    for row_start in range(0, h, CHUNK):
        row_end = min(row_start + CHUNK, h)
        chunk_h = row_end - row_start

        prob_chunk = pred_sum[row_start:row_end, :]
        binary_chunk = ((prob_chunk > threshold).astype(np.uint8)) * 255

        chunk_transform = rasterio.transform.from_bounds(
            *rasterio.transform.array_bounds(chunk_h, w,
                rasterio.windows.transform(Window(0, row_start, w, chunk_h), src_transform)),
            w, chunk_h,
        )

        mask_valid = binary_chunk > 0
        if not mask_valid.any():
            continue

        for geom, val in shapes(binary_chunk, mask=mask_valid, transform=chunk_transform):
            features.append({
                "type": "Feature",
                "properties": {"class": "building", "value": int(val)},
                "geometry": geom,
            })

    # CRS handling
    crs_epsg = None
    if src_crs is not None:
        try:
            crs_epsg = src_crs.to_epsg()
        except Exception:
            crs_epsg = None

    geojson_dict = {
        "type": "FeatureCollection",
        "name": "building_footprints",
        "crs": {
            "type": "name",
            "properties": {
                "name": f"urn:ogc:def:crs:EPSG::{crs_epsg}" if isinstance(crs_epsg, int) else str(src_crs)
            }
        },
        "features": features,
    }

    with open(out_geojson_path, 'w', encoding='utf-8') as f:
        json.dump(geojson_dict, f)

    # -- Cleanup temp files -----------------------------------------------
    del pred_sum, weight_sum
    try:
        os.remove(pred_sum_path)
        os.remove(weight_sum_path)
        os.rmdir(tmp_dir)
    except Exception:
        pass

    on_stage("done", 1.0)
    return {
        "building_count": len(features),
        "width": w,
        "height": h,
        "crs_epsg": crs_epsg,
        "crs_raw": str(src_crs) if src_crs is not None else None,
    }


# =========================================================================
# Backward-compatible CLI (mirrors the original predict.py)
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="GAN-SCA 3-Band Prediction (importable core)")
    parser.add_argument("--model", type=str, default="model/best_gansca_3band.pth")
    parser.add_argument("--input", type=str, default="../input/sample.tif")
    parser.add_argument("--out_tif", type=str, default="data/results/cli/prediction_mask.tif")
    parser.add_argument("--out_vector", type=str, default="data/results/cli/prediction_vector.geojson")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=None)
    opts = parser.parse_args()

    device = get_device()
    print(f"[device] {get_device_info()}")
    model = load_model(opts.model, device)
    print("[model] weights loaded")

    def cli_stage(stage, frac=None):
        msg = stage if frac is None else f"{stage} {frac * 100:5.1f}%"
        print(f"  -> {msg}", end="\r" if stage == "inference" else "\n")

    summary = run_prediction(
        model, opts.input, opts.out_tif, opts.out_vector,
        patch_size=opts.patch_size, batch_size=opts.batch_size,
        device=device, on_stage=cli_stage,
    )
    print("\n[done]", summary)


if __name__ == "__main__":
    main()
