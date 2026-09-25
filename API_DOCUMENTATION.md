# GAN-SCA API Reference

Base URL: the same origin that serves the frontend (default
`http://127.0.0.1:5000`). All responses are JSON unless noted. Uploaded
files and results are keyed by a 12-character hex id.

The prediction endpoint is **asynchronous**: it queues a background job and
returns immediately. The client polls the job endpoint until the job is
`done`, then fetches the result. This avoids long-held requests and lets the
UI show live progress.

---

## `GET /api/health`

Server and inference status. Safe to call anytime (used by the UI on load).

**200**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "inference_available": true,
  "model_file_present": true,
  "device": { "type": "cuda", "name": "NVIDIA RTX A4000" },
  "model_loaded": true
}
```
- `inference_available` — PyTorch + rasterio importable.
- `model_file_present` — checkpoint exists at the configured path.
- If inference is unavailable, `device`/`model_loaded` are omitted and a
  `reason` string is included instead.

---

## `POST /api/predict`

Queue a building-extraction job for one GeoTIFF.

**Request** — `multipart/form-data`

| Field   | Type | Required | Notes                          |
|---------|------|----------|--------------------------------|
| `image` | file | yes      | 3-band GeoTIFF (`.tif`/`.tiff`) |

**202 Accepted**
```json
{ "job_id": "3d9aed95918d", "status": "queued" }
```

**Errors**

| Status | When                                                        |
|--------|-------------------------------------------------------------|
| `400`  | No file, empty filename, wrong extension, or unreadable raster |
| `503`  | Inference stack unavailable, or model checkpoint missing    |

Error bodies include `error` and often a `hint`:
```json
{ "error": "Model checkpoint not found.",
  "hint": "Place best_gansca_3band.pth at: .../backend/model/best_gansca_3band.pth" }
```

---

## `GET /api/jobs/<job_id>`

Poll a job. Recommended interval ≈ 1.2 s.

**200 (running)**
```json
{
  "job_id": "3d9aed95918d",
  "status": "running",
  "stage": "inference",
  "stage_label": "Detecting buildings",
  "progress": 0.62
}
```

**200 (done)** — adds `result_id` (equal to `job_id`):
```json
{ "job_id": "3d9aed95918d", "status": "done", "stage": "done",
  "stage_label": "Complete", "progress": 1.0, "result_id": "3d9aed95918d" }
```

**200 (error)** — adds `error`.

**404** — unknown `job_id`.

### Stages (`stage` → `stage_label`)

| stage          | label                       | progress        |
|----------------|-----------------------------|-----------------|
| `queued`       | Queued                      | null            |
| `preparing`    | Reading raster              | null            |
| `normalizing`  | Estimating band statistics  | null            |
| `inference`    | Detecting buildings         | 0.0 → 1.0       |
| `writing_mask` | Building mask raster        | null            |
| `vectorizing`  | Vectorizing footprints      | null            |
| `reprojecting` | Reprojecting to WGS84       | null            |
| `done`         | Complete                    | 1.0             |
| `error`        | Failed                      | —               |

`progress` is only numeric during `inference`; other stages report `null`
(the UI shows an indeterminate bar).

---

## `GET /api/results/<result_id>/meta`

Result metadata.

**200**
```json
{
  "result_id": "3d9aed95918d",
  "building_count": 578,
  "width": 1024,
  "height": 1024,
  "crs_epsg": 32645,
  "crs_raw": "EPSG:32645",
  "total_area_m2": 64202.31,
  "center_latlon": [22.6244, 88.3914],
  "bounds_latlon": [[22.6223, 88.3891], [22.6265, 88.3936]],
  "has_preview": true,
  "source_filename": "sample.tif",
  "processing_time_sec": 12.4,
  "device": { "type": "cuda", "name": "NVIDIA RTX A4000" },
  "created_at": "2026-01-01T00:00:00+00:00"
}
```
`bounds_latlon` is `[[south, west], [north, east]]` (Leaflet order).
Demo results carry `"is_demo": true` with `processing_time_sec` / `device`
null. **404** if the id is unknown.

---

## `GET /api/results/<result_id>/vector_wgs84`

Building footprints as a GeoJSON `FeatureCollection` in **WGS84**
(CRS84 / EPSG:4326) — this is what the map renders. Each feature carries
`class`, `value`, and `area_m2` (computed in the source projected CRS).
Content-Type `application/geo+json`. **404** if unknown.

## `GET /api/results/<result_id>/vector`

Same footprints in the **source CRS** (e.g. EPSG:32645), served as a
download for GIS tools. **404** if unknown.

## `GET /api/results/<result_id>/mask`

The RGBA GeoTIFF mask (buildings red, background transparent), served as a
download. Content-Type `image/tiff`. **404** if unknown.

## `GET /api/results/<result_id>/preview.png`

A downscaled, browser-viewable PNG of the mask (longest side ≤ 1400 px).
Content-Type `image/png`. **404** if unknown or preview generation failed.

---

## Typical client flow

```
POST /api/predict            → { job_id }
loop:
  GET /api/jobs/<job_id>      → status/stage/progress
  until status == "done"
GET  /api/results/<id>/meta          → counts, CRS, bounds, timing
GET  /api/results/<id>/vector_wgs84  → draw footprints on the map
(download links point at /mask, /vector, /vector_wgs84)
```
