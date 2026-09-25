#!/bin/bash
set -e

MODEL_PATH="backend/model/best_gansca_3band.pth"

if [ ! -f "$MODEL_PATH" ]; then
  if [ -z "$HF_MODEL_REPO" ]; then
    echo "No local model and HF_MODEL_REPO is not set — starting in demo-only mode."
  else
    echo "Downloading model checkpoint from Hugging Face repo: $HF_MODEL_REPO ..."
    python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil, os

path = hf_hub_download(
    repo_id=os.environ["HF_MODEL_REPO"],
    filename=os.environ.get("HF_MODEL_FILENAME", "best_gansca_3band.pth"),
    token=os.environ.get("HF_TOKEN"),
)
os.makedirs("backend/model", exist_ok=True)
shutil.copy(path, "backend/model/best_gansca_3band.pth")
print("Model downloaded to backend/model/best_gansca_3band.pth")
PY
  fi
fi

exec gunicorn -w 1 --threads 8 --timeout 600 -b 0.0.0.0:${GANSCA_PORT:-8080} backend.app:app