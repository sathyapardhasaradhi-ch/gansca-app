
# GAN-SCA · Building Footprint Extraction

A complete web application for extracting building footprints from
high-resolution satellite imagery using the **GAN-SCA** generator
(GAN with Spatial-Channel Attention). Upload a georeferenced GeoTIFF, run
detection, and explore the extracted footprints on an interactive map with
downloadable raster masks and GeoJSON vectors.

Built around the original GAN-SCA inference pipeline — the model and its
numerical behaviour are preserved exactly; this project adds the web API and
frontend on top.

---

## What's inside

```
gansca-app/
├── backend/                 Flask API + inference core
│   ├── app.py               HTTP server: jobs, results, static hosting
│   ├── inference.py         GeneratorSCA + sliding-window pipeline (from source backend)
│   ├── geo_utils.py         UTM→WGS84 reprojection, areas, raster meta, PNG preview
│   ├── config.py            All paths/limits/thresholds (env-overridable)
│   ├── requirements.txt
│   ├── model/               ← place best_gansca_3band.pth here
│   ├── samples/             shipped Kolkata demo (mask + geojson)
│   └── data/                runtime uploads + results (auto-created)
└── frontend/                static client (HTML/CSS/JS, no build step)
    ├── index.html
    ├── css/styles.css
    └── js/{config,api,map,app}.js
```

---

## Prerequisites

- **Python 3.10+**
- ~2 GB free disk for the PyTorch install (CPU or CUDA build)
- Optional: an NVIDIA GPU + CUDA drivers for fast inference (the app
  auto-detects and uses CUDA when present, otherwise runs on CPU)

---

## Setup

From the `backend/` directory:

```bash
cd backend

# 1. (recommended) virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt
```

`requirements.txt` installs the original inference stack (numpy, rasterio,
torch, rich) plus the web layer (Flask, pyproj, Pillow).

### Place the model

Copy your trained checkpoint into `backend/model/`:

```
backend/model/best_gansca_3band.pth
```

(Or point `GANSCA_MODEL` at it anywhere on disk — see Configuration.)

Without the checkpoint the server still runs in **demo-only mode**: the
bundled Kolkata sample scene works, but new uploads can't be processed.

---

## Run

```bash
cd backend
python app.py
```

Then open **http://127.0.0.1:5000/** in a browser.

On startup the server prints whether the inference stack and model were
found, warms the model in the background, and seeds the Kolkata demo scene.

---

## Using it

1. **Load demo** — click *Load demo* to view the pre-computed Kolkata scene
   (578 footprints) immediately, without a model. Good for a first look.
2. **Run detection** — drop a **3-band GeoTIFF** onto the ingest panel (or
   click to browse) and press *Detect buildings*. A progress bar reports the
   live stage (reading → band statistics → detection → vectorizing →
   reprojecting). When it finishes, the footprints render on the map and the
   readout fills in.
3. **Explore** — toggle the footprint layer, switch between satellite
   imagery and street basemaps, and click any building for its area.
4. **Download** — grab the RGBA mask (GeoTIFF), the footprints in the
   source CRS (for GIS), or the footprints in WGS84 (for web maps).

The map basemap defaults to Esri World Imagery so footprints sit directly on
top of the satellite view they were extracted from.

---

## Configuration

All settings are environment variables (with sensible defaults in
`config.py`):

| Variable           | Default                         | Purpose                               |
|--------------------|---------------------------------|---------------------------------------|
| `GANSCA_HOST`      | `127.0.0.1`                     | Bind address                          |
| `GANSCA_PORT`      | `5000`                          | Port                                  |
| `GANSCA_MODEL`     | `backend/model/best_gansca_3band.pth` | Checkpoint path                 |
| `GANSCA_MAX_MB`    | `512`                           | Max upload size (MB)                  |
| `GANSCA_PATCH`     | `256`                           | Sliding-window patch size             |
| `GANSCA_THRESHOLD` | `0.5`                           | Probability → building threshold      |
| `GANSCA_WARMUP`    | `1`                             | Preload model at startup              |
| `GANSCA_DEBUG`     | `0`                             | Flask debug mode                      |

Example:

```bash
GANSCA_PORT=8080 GANSCA_MODEL=/data/models/gansca.pth python app.py
```

---

## How inference works (unchanged from the source backend)

1. **Percentile normalization** — per-band 2nd/98th percentiles estimated
   from 100 random windows (never loads the whole image).
2. **Sliding-window inference** — 256×256 patches, 50 % overlap, blended
   with Gaussian weights; FP16 autocast on GPU, FP32 on CPU; memory-mapped
   accumulators so arbitrarily large rasters fit in RAM.
3. **Mask** — probabilities thresholded at 0.5 → tiled, deflate-compressed
   RGBA GeoTIFF (buildings red, background transparent).
4. **Vectors** — the mask is polygonized to GeoJSON with the source CRS
   preserved.

The web layer then reprojects the vectors to **WGS84** for the map (the
source imagery is typically in a UTM projection, e.g. EPSG:32645), computes
per-building areas in the projected CRS, and renders a PNG preview of the
mask for the browser.

---

## Troubleshooting

- **Status chip says "demo only"** — PyTorch/rasterio aren't importable.
  Re-run `pip install -r requirements.txt` in the active environment.
- **Status chip says "no model file"** — the checkpoint isn't at
  `GANSCA_MODEL`. Drop `best_gansca_3band.pth` into `backend/model/`.
- **Footprints don't line up with the basemap** — the input GeoTIFF has no
  CRS, so reprojection can't run. Ensure the raster is georeferenced.
- **Prediction is slow** — you're on CPU. Expected. A CUDA GPU is used
  automatically when available.
- **Large image seems stuck** — big scenes legitimately take minutes; the
  progress bar switches to the *Detecting buildings* stage with a percentage.

See `API_DOCUMENTATION.md` for the HTTP contract and
`PROJECT_SPECIFICATION.md` for architecture, data flow, and test status.
