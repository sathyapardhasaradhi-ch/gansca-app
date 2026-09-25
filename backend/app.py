"""
GAN-SCA Building Footprint Extraction -- web application server.

Responsibilities
----------------
* Serve the static frontend (single origin -> no CORS needed by default).
* Expose the existing GAN-SCA inference pipeline over HTTP.
* Run inference as a background job (it is slow) and let the browser poll
  for live progress instead of holding a long request open.
* Prepare model output for display: reproject vectors to WGS84, compute
  areas/bounds, and render a PNG preview of the raster mask.

The model itself is never re-implemented here -- this file only orchestrates
`inference.run_prediction` (the single source of truth) and `geo_utils`.

Inference depends on PyTorch. If torch/rasterio are not installed the server
still boots and serves the frontend + the pre-seeded Kolkata demo; only the
live `/api/predict` route reports that inference is unavailable. This keeps
the app demonstrable everywhere while requiring the full stack only to run
new predictions.
"""

import os
import json
import shutil
import threading
import traceback
import uuid
from datetime import datetime, timezone

from flask import Flask, request, jsonify, send_file, send_from_directory, abort

from config import Config

# --- Optional heavy imports (torch/rasterio) -----------------------------
try:
    import inference
    import geo_utils
    INFERENCE_AVAILABLE = True
    INFERENCE_IMPORT_ERROR = None
except Exception as e:  # torch or rasterio missing
    inference = None
    try:
        import geo_utils  # geo_utils only needs rasterio; try independently
    except Exception:
        geo_utils = None
    INFERENCE_AVAILABLE = False
    INFERENCE_IMPORT_ERROR = f"{type(e).__name__}: {e}"


app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
Config.ensure_dirs()

# -------------------------------------------------------------------------
# In-process job registry (adequate for a single-instance research tool;
# a multi-user deployment would swap this for Celery/RQ + Redis).
# -------------------------------------------------------------------------
_JOBS = {}
_JOBS_LOCK = threading.Lock()

_STAGE_LABELS = {
    "queued": "Queued",
    "preparing": "Reading raster",
    "normalizing": "Estimating band statistics",
    "inference": "Detecting buildings",
    "writing_mask": "Building mask raster",
    "vectorizing": "Vectorizing footprints",
    "reprojecting": "Reprojecting to WGS84",
    "finalizing": "Finalizing outputs",
    "done": "Complete",
    "error": "Failed",
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _set_job(job_id, **fields):
    with _JOBS_LOCK:
        job = _JOBS.setdefault(job_id, {})
        job.update(fields)


def _get_job(job_id):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


# -------------------------------------------------------------------------
# Result helpers
# -------------------------------------------------------------------------
def _result_dir(result_id):
    return os.path.join(Config.RESULTS_DIR, result_id)


def _result_path(result_id, name):
    return os.path.join(_result_dir(result_id), name)


def _write_result_bundle(result_id, source_geojson, src_epsg, mask_tif_path,
                         summary, extra_meta):
    """
    Given a raw source-CRS geojson + mask tif, produce the full served bundle:
    enriched source vector, WGS84 vector, preview PNG, and meta.json.
    Returns the meta dict.
    """
    rdir = _result_dir(result_id)
    os.makedirs(rdir, exist_ok=True)

    wgs84, enriched_src, stats = geo_utils.reproject_and_enrich(source_geojson, src_epsg)

    with open(_result_path(result_id, "vector.geojson"), "w", encoding="utf-8") as f:
        json.dump(enriched_src, f)
    with open(_result_path(result_id, "vector_wgs84.geojson"), "w", encoding="utf-8") as f:
        json.dump(wgs84, f)

    # mask tif into the bundle (copy if produced elsewhere)
    bundle_mask = _result_path(result_id, "mask.tif")
    if os.path.abspath(mask_tif_path) != os.path.abspath(bundle_mask):
        shutil.copyfile(mask_tif_path, bundle_mask)

    # browser preview
    preview_ok = False
    try:
        geo_utils.mask_to_png(bundle_mask, _result_path(result_id, "preview.png"))
        preview_ok = True
    except Exception:
        preview_ok = False

    meta = {
        "result_id": result_id,
        "building_count": summary.get("building_count", stats["count"]),
        "width": summary.get("width"),
        "height": summary.get("height"),
        "crs_epsg": src_epsg,
        "crs_raw": summary.get("crs_raw"),
        "total_area_m2": stats["total_area_m2"],
        "center_latlon": stats["center_latlon"],
        "bounds_latlon": stats["bounds_latlon"],
        "has_preview": preview_ok,
        "created_at": _now(),
    }
    meta.update(extra_meta or {})
    with open(_result_path(result_id, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


# -------------------------------------------------------------------------
# Background prediction job
# -------------------------------------------------------------------------
def _run_prediction_job(job_id, upload_path, source_filename):
    started = datetime.now(timezone.utc)
    try:
        _set_job(job_id, status="running", stage="preparing", progress=None)

        device = inference.get_device()
        model = inference.load_model(Config.MODEL_PATH, device)

        result_id = job_id  # reuse the job id as the result id
        rdir = _result_dir(result_id)
        os.makedirs(rdir, exist_ok=True)
        raw_mask = _result_path(result_id, "mask.tif")
        raw_vector = _result_path(result_id, "_vector_src.geojson")

        def on_stage(stage, frac=None):
            _set_job(job_id, status="running", stage=stage, progress=frac)

        summary = inference.run_prediction(
            model, upload_path, raw_mask, raw_vector,
            patch_size=Config.PATCH_SIZE, batch_size=Config.BATCH_SIZE,
            threshold=Config.PROB_THRESHOLD, building_color=Config.BUILDING_COLOR,
            device=device, on_stage=on_stage,
        )

        _set_job(job_id, stage="reprojecting", progress=None)
        with open(raw_vector, "r", encoding="utf-8") as f:
            source_geojson = json.load(f)

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        _write_result_bundle(
            result_id, source_geojson, summary.get("crs_epsg"), raw_mask, summary,
            extra_meta={
                "source_filename": source_filename,
                "processing_time_sec": round(elapsed, 2),
                "device": inference.get_device_info(),
            },
        )

        # tidy intermediate + upload
        for p in (raw_vector, upload_path):
            try:
                os.remove(p)
            except Exception:
                pass

        _set_job(job_id, status="done", stage="done", progress=1.0,
                 result_id=result_id, finished_at=_now())
    except Exception as e:
        _set_job(job_id, status="error", stage="error",
                 error=f"{type(e).__name__}: {e}", finished_at=_now())
        app.logger.error("Prediction job %s failed:\n%s", job_id, traceback.format_exc())


# =========================================================================
# API routes
# =========================================================================

@app.get("/api/health")
def health():
    model_exists = os.path.exists(Config.MODEL_PATH)
    info = {
        "status": "ok",
        "version": Config.VERSION,
        "inference_available": INFERENCE_AVAILABLE,
        "model_file_present": model_exists,
    }
    if INFERENCE_AVAILABLE:
        info["device"] = inference.get_device_info()
        info["model_loaded"] = inference.model_is_loaded(Config.MODEL_PATH)
    else:
        info["reason"] = INFERENCE_IMPORT_ERROR
    return jsonify(info)


@app.post("/api/predict")
def predict():
    if not INFERENCE_AVAILABLE:
        return jsonify({
            "error": "Inference stack unavailable on this server.",
            "detail": INFERENCE_IMPORT_ERROR,
            "hint": "Install requirements.txt (torch, rasterio, pyproj) to enable predictions.",
        }), 503
    if not os.path.exists(Config.MODEL_PATH):
        return jsonify({
            "error": "Model checkpoint not found.",
            "hint": f"Place best_gansca_3band.pth at: {Config.MODEL_PATH}",
        }), 503

    if "image" not in request.files:
        return jsonify({"error": "No file uploaded. Send a GeoTIFF in the 'image' field."}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{ext}'.",
            "hint": "Upload a 3-band GeoTIFF (.tif / .tiff).",
        }), 400

    job_id = uuid.uuid4().hex[:12]
    upload_path = os.path.join(Config.UPLOAD_DIR, f"{job_id}{ext}")
    f.save(upload_path)

    # validate it is actually a readable raster before queuing
    if geo_utils is not None and not geo_utils.is_readable_raster(upload_path):
        try:
            os.remove(upload_path)
        except Exception:
            pass
        return jsonify({"error": "File is not a readable GeoTIFF raster."}), 400

    _set_job(job_id, status="queued", stage="queued", progress=None,
             source_filename=f.filename, created_at=_now())
    threading.Thread(
        target=_run_prediction_job, args=(job_id, upload_path, f.filename), daemon=True
    ).start()

    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = _get_job(job_id)
    if job is None:
        return jsonify({"error": "Unknown job id."}), 404
    stage = job.get("stage", job.get("status"))
    out = {
        "job_id": job_id,
        "status": job.get("status"),
        "stage": stage,
        "stage_label": _STAGE_LABELS.get(stage, stage),
        "progress": job.get("progress"),
    }
    if job.get("status") == "done":
        out["result_id"] = job.get("result_id")
    if job.get("status") == "error":
        out["error"] = job.get("error")
    return jsonify(out)


@app.get("/api/results/<result_id>/meta")
def result_meta(result_id):
    p = _result_path(result_id, "meta.json")
    if not os.path.exists(p):
        return jsonify({"error": "Unknown result id."}), 404
    with open(p, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.get("/api/results/<result_id>/vector_wgs84")
def result_vector_wgs84(result_id):
    p = _result_path(result_id, "vector_wgs84.geojson")
    if not os.path.exists(p):
        return jsonify({"error": "Not found."}), 404
    return send_file(p, mimetype="application/geo+json")


@app.get("/api/results/<result_id>/vector")
def result_vector(result_id):
    p = _result_path(result_id, "vector.geojson")
    if not os.path.exists(p):
        return jsonify({"error": "Not found."}), 404
    return send_file(p, mimetype="application/geo+json",
                     as_attachment=True, download_name=f"{result_id}_buildings.geojson")


@app.get("/api/results/<result_id>/mask")
def result_mask(result_id):
    p = _result_path(result_id, "mask.tif")
    if not os.path.exists(p):
        return jsonify({"error": "Not found."}), 404
    return send_file(p, mimetype="image/tiff",
                     as_attachment=True, download_name=f"{result_id}_mask.tif")


@app.get("/api/results/<result_id>/preview.png")
def result_preview(result_id):
    p = _result_path(result_id, "preview.png")
    if not os.path.exists(p):
        return jsonify({"error": "Not found."}), 404
    return send_file(p, mimetype="image/png")


# =========================================================================
# Static frontend (served from the same origin as the API)
# =========================================================================

@app.get("/")
def index():
    return send_from_directory(Config.FRONTEND_DIR, "index.html")


@app.get("/<path:path>")
def static_proxy(path):
    full = os.path.join(Config.FRONTEND_DIR, path)
    if os.path.isfile(full):
        return send_from_directory(Config.FRONTEND_DIR, path)
    abort(404)


# =========================================================================
# Startup: seed the Kolkata demo + warm the model
# =========================================================================

def seed_sample_result():
    """
    Build the pre-computed Kolkata demo result from the shipped sample assets
    (mask.tif + source geojson) so the frontend has something to display
    without running the model. Also exercises geo_utils on real data at boot.
    """
    if geo_utils is None:
        return
    rid = Config.SAMPLE_RESULT_ID
    if os.path.exists(_result_path(rid, "meta.json")):
        return  # already seeded

    src_tif = os.path.join(Config.SAMPLE_ASSETS_DIR, "prediction_mask.tif")
    src_geo = os.path.join(Config.SAMPLE_ASSETS_DIR, "prediction_vector.geojson")
    if not (os.path.exists(src_tif) and os.path.exists(src_geo)):
        return

    try:
        with open(src_geo, "r", encoding="utf-8") as f:
            source_geojson = json.load(f)
        # recover EPSG from the geojson crs urn
        src_epsg = None
        try:
            urn = source_geojson["crs"]["properties"]["name"]
            if "EPSG::" in urn:
                src_epsg = int(urn.split("EPSG::")[-1])
        except Exception:
            src_epsg = None

        meta_extra = {
            "source_filename": "Kolkata sample tile (CartoSat-3, 1024x1024)",
            "processing_time_sec": None,
            "device": None,
            "is_demo": True,
        }
        summary = {
            "building_count": len(source_geojson.get("features", [])),
            "width": 1024, "height": 1024,
            "crs_epsg": src_epsg, "crs_raw": f"EPSG:{src_epsg}" if src_epsg else None,
        }
        _write_result_bundle(rid, source_geojson, src_epsg, src_tif, summary, meta_extra)
        app.logger.info("Seeded demo result '%s'.", rid)
    except Exception:
        app.logger.warning("Could not seed demo result:\n%s", traceback.format_exc())


def warm_model():
    if not (INFERENCE_AVAILABLE and Config.WARMUP_MODEL_ON_START):
        return
    if not os.path.exists(Config.MODEL_PATH):
        return

    def _warm():
        try:
            inference.load_model(Config.MODEL_PATH, inference.get_device())
            app.logger.info("Model warmed up.")
        except Exception:
            app.logger.warning("Model warm-up failed:\n%s", traceback.format_exc())

    threading.Thread(target=_warm, daemon=True).start()


seed_sample_result()
warm_model()


if __name__ == "__main__":
    print(f"GAN-SCA server  v{Config.VERSION}")
    print(f"  inference available : {INFERENCE_AVAILABLE}"
          + ("" if INFERENCE_AVAILABLE else f"  ({INFERENCE_IMPORT_ERROR})"))
    print(f"  model checkpoint    : {Config.MODEL_PATH} "
          f"({'present' if os.path.exists(Config.MODEL_PATH) else 'MISSING'})")
    print(f"  serving frontend    : {Config.FRONTEND_DIR}")
    print(f"  open                : http://{Config.HOST}:{Config.PORT}/")
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG, threaded=True)
