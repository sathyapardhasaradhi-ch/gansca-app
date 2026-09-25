"""
Central configuration for the GAN-SCA web application.

Every tunable lives here so nothing is hard-coded across the codebase. All
values can be overridden with environment variables, which keeps the same
build working on a laptop, a lab workstation, or a server without edits.
"""

import os

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)


class Config:
    # -- Server ----------------------------------------------------------
    HOST = os.environ.get("GANSCA_HOST", "127.0.0.1")
    PORT = int(os.environ.get("GANSCA_PORT", "5000"))
    DEBUG = os.environ.get("GANSCA_DEBUG", "0") == "1"

    # -- Model + data locations -----------------------------------------
    MODEL_PATH = os.environ.get(
        "GANSCA_MODEL", os.path.join(BACKEND_DIR, "model", "best_gansca_3band.pth")
    )
    DATA_DIR = os.path.join(BACKEND_DIR, "data")
    UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
    RESULTS_DIR = os.path.join(DATA_DIR, "results")
    SAMPLE_ASSETS_DIR = os.path.join(BACKEND_DIR, "samples")   # shipped demo tif+geojson
    FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")

    # -- Upload validation ----------------------------------------------
    ALLOWED_EXTENSIONS = {".tif", ".tiff"}
    MAX_CONTENT_MB = int(os.environ.get("GANSCA_MAX_MB", "512"))
    MAX_CONTENT_LENGTH = MAX_CONTENT_MB * 1024 * 1024

    # -- Inference parameters (defaults match the validated backend) -----
    PATCH_SIZE = int(os.environ.get("GANSCA_PATCH", "256"))
    BATCH_SIZE = None  # None => 32 on GPU, 4 on CPU (backend default)
    PROB_THRESHOLD = float(os.environ.get("GANSCA_THRESHOLD", "0.5"))
    BUILDING_COLOR = (255, 0, 0)

    # -- Housekeeping ----------------------------------------------------
    WARMUP_MODEL_ON_START = os.environ.get("GANSCA_WARMUP", "1") == "1"
    SAMPLE_RESULT_ID = "sample"
    VERSION = "1.0.0"

    @classmethod
    def ensure_dirs(cls):
        for d in (cls.DATA_DIR, cls.UPLOAD_DIR, cls.RESULTS_DIR):
            os.makedirs(d, exist_ok=True)
