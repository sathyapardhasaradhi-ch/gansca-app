/* Application orchestration: UI state + the predict/poll/render flow.
   Depends on GANSCA_CONFIG, GANSCA_API, GANSCA_MAP. */
(function () {
  const C = window.GANSCA_CONFIG;
  const API = window.GANSCA_API;
  const MAP = window.GANSCA_MAP;

  // ---- DOM ----
  const $ = (id) => document.getElementById(id);
  const el = {
    statusDot: $("statusDot"), statusLabel: $("statusLabel"),
    dropzone: $("dropzone"), fileInput: $("fileInput"),
    fileTag: $("fileTag"), fileName: $("fileName"), fileClear: $("fileClear"),
    runBtn: $("runBtn"), demoBtn: $("demoBtn"),
    progress: $("progress"), progStage: $("progStage"), progPct: $("progPct"), progBar: $("progBar"),
    notice: $("notice"),
    readout: $("readout"),
    rCount: $("rCount"), rSource: $("rSource"), rDims: $("rDims"), rCrs: $("rCrs"),
    rCenter: $("rCenter"), rArea: $("rArea"), rDevice: $("rDevice"), rTime: $("rTime"),
    previewWrap: $("previewWrap"), previewImg: $("previewImg"),
    dlMask: $("dlMask"), dlVector: $("dlVector"), dlVectorWgs: $("dlVectorWgs"),
    emptyState: $("emptyState"), mapCtrl: $("mapCtrl"),
    toggleBuildings: $("toggleBuildings"), basemapBtn: $("basemapBtn"), fitBtn: $("fitBtn"),
  };

  let selectedFile = null;
  let busy = false;
  let pollTimer = null;
  let inferenceAvailable = true;

  // ---- Small UI helpers ----
  function showNotice(msg, kind) {
    el.notice.textContent = msg;
    el.notice.className = "notice" + (kind ? ` is-${kind}` : "");
    el.notice.hidden = false;
  }
  function clearNotice() { el.notice.hidden = true; }

  function setProgress(label, frac) {
    el.progress.hidden = false;
    el.progStage.textContent = label || "Working…";
    if (frac == null || isNaN(frac)) {
      el.progBar.classList.add("is-indeterminate");
      el.progPct.textContent = "";
    } else {
      el.progBar.classList.remove("is-indeterminate");
      const pct = Math.max(0, Math.min(100, Math.round(frac * 100)));
      el.progBar.style.width = pct + "%";
      el.progPct.textContent = pct + "%";
    }
  }
  function hideProgress() {
    el.progress.hidden = true;
    el.progBar.classList.remove("is-indeterminate");
    el.progBar.style.width = "0%";
  }

  function setBusy(state) {
    busy = state;
    el.runBtn.disabled = state || !selectedFile || !inferenceAvailable;
    el.demoBtn.disabled = state;
    el.dropzone.style.pointerEvents = state ? "none" : "";
    el.dropzone.style.opacity = state ? "0.5" : "";
  }

  function fmtArea(m2) {
    if (m2 == null) return "—";
    if (m2 >= 1e6) return (m2 / 1e6).toFixed(2) + " km²";
    return Math.round(m2).toLocaleString() + " m²";
  }
  function fmtCenter(c) {
    if (!c) return "—";
    const [lat, lon] = c;
    const ns = lat >= 0 ? "N" : "S", ew = lon >= 0 ? "E" : "W";
    return `${Math.abs(lat).toFixed(4)}°${ns}, ${Math.abs(lon).toFixed(4)}°${ew}`;
  }

  // ---- File selection ----
  function pickFile(file) {
    if (!file) return;
    const ok = /\.tiff?$/i.test(file.name);
    if (!ok) { showNotice("Please choose a GeoTIFF (.tif or .tiff).", "warn"); return; }
    selectedFile = file;
    el.fileName.textContent = file.name;
    el.fileTag.hidden = false;
    el.runBtn.disabled = !inferenceAvailable || busy;
    clearNotice();
    if (!inferenceAvailable) {
      showNotice("Inference is offline on this server, so new tiles can't be processed. The demo still works.", "warn");
    }
  }
  function clearFile() {
    selectedFile = null;
    el.fileInput.value = "";
    el.fileTag.hidden = true;
    el.runBtn.disabled = true;
  }

  // ---- Rendering a completed result ----
  async function renderResult(resultId) {
    const meta = await API.resultMeta(resultId);
    const geojson = await API.vectorWgs84(resultId);

    // telemetry
    el.rCount.textContent = Number(meta.building_count || 0).toLocaleString();
    el.rSource.textContent = meta.source_filename || resultId;
    el.rDims.textContent = (meta.width && meta.height) ? `${meta.width} × ${meta.height} px` : "—";
    el.rCrs.textContent = meta.crs_epsg ? `EPSG:${meta.crs_epsg}` : (meta.crs_raw || "—");
    el.rCenter.textContent = fmtCenter(meta.center_latlon);
    el.rArea.textContent = fmtArea(meta.total_area_m2);
    el.rDevice.textContent = meta.device ? `${meta.device.name}` : (meta.is_demo ? "pre-computed" : "—");
    el.rTime.textContent = (meta.processing_time_sec != null)
      ? `${meta.processing_time_sec.toFixed(1)} s` : (meta.is_demo ? "—" : "—");

    // preview
    if (meta.has_preview) {
      el.previewImg.src = API.urls.preview(resultId);
      el.previewWrap.hidden = false;
    } else {
      el.previewWrap.hidden = true;
    }

    // downloads
    el.dlMask.href = API.urls.mask(resultId);
    el.dlVector.href = API.urls.vector(resultId);
    el.dlVectorWgs.href = API.urls.vectorWgs84(resultId);

    el.readout.hidden = false;

    // map
    el.emptyState.classList.add("is-hidden");
    MAP.invalidate();
    MAP.loadBuildings(geojson);
    el.mapCtrl.hidden = false;
    el.toggleBuildings.setAttribute("aria-pressed", "true");
  }

  // ---- Poll a running job ----
  function pollJob(jobId, startedAt) {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(async () => {
      try {
        const s = await API.jobStatus(jobId);
        if (s.status === "running" || s.status === "queued") {
          setProgress(s.stage_label || "Working…", s.progress);
          if (Date.now() - startedAt > C.POLL_TIMEOUT_MS) {
            throw new Error("Timed out waiting for the prediction to finish.");
          }
          pollJob(jobId, startedAt);
        } else if (s.status === "done") {
          setProgress("Complete", 1);
          await renderResult(s.result_id);
          hideProgress();
          setBusy(false);
        } else if (s.status === "error") {
          throw new Error(s.error || "Prediction failed.");
        }
      } catch (err) {
        hideProgress();
        setBusy(false);
        showNotice(err.message || "Something went wrong.", "error");
      }
    }, C.POLL_INTERVAL_MS);
  }

  // ---- Actions ----
  async function runPrediction() {
    if (!selectedFile || busy) return;
    clearNotice();
    setBusy(true);
    setProgress("Uploading…", null);
    try {
      const { job_id } = await API.predict(selectedFile);
      setProgress("Queued", null);
      pollJob(job_id, Date.now());
    } catch (err) {
      hideProgress();
      setBusy(false);
      showNotice(err.message || "Upload failed.", "error");
    }
  }

  async function loadDemo() {
    if (busy) return;
    clearNotice();
    setBusy(true);
    setProgress("Loading demo scene…", null);
    try {
      await renderResult(C.SAMPLE_RESULT_ID);
      hideProgress();
      setBusy(false);
    } catch (err) {
      hideProgress();
      setBusy(false);
      showNotice(
        "Demo scene isn't available on this server yet. " + (err.message || ""),
        "warn"
      );
    }
  }

  // ---- Health ----
  async function checkHealth() {
    try {
      const h = await API.health();
      inferenceAvailable = !!h.inference_available && !!h.model_file_present;
      if (h.inference_available && h.model_file_present) {
        const dev = h.device ? h.device.type.toUpperCase() : "READY";
        el.statusDot.className = "status-chip__dot is-ok";
        el.statusLabel.textContent = `online · ${dev}`;
      } else if (h.inference_available && !h.model_file_present) {
        el.statusDot.className = "status-chip__dot is-warn";
        el.statusLabel.textContent = "no model file";
      } else {
        el.statusDot.className = "status-chip__dot is-warn";
        el.statusLabel.textContent = "demo only";
      }
    } catch (_) {
      inferenceAvailable = false;
      el.statusDot.className = "status-chip__dot is-down";
      el.statusLabel.textContent = "offline";
    }
    el.runBtn.disabled = !selectedFile || !inferenceAvailable || busy;
  }

  // ---- Wire up ----
  function bind() {
    el.dropzone.addEventListener("click", () => el.fileInput.click());
    el.dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.fileInput.click(); }
    });
    el.fileInput.addEventListener("change", (e) => pickFile(e.target.files[0]));
    el.fileClear.addEventListener("click", clearFile);

    ["dragenter", "dragover"].forEach((ev) =>
      el.dropzone.addEventListener(ev, (e) => {
        e.preventDefault(); e.stopPropagation();
        el.dropzone.classList.add("is-drag");
      })
    );
    ["dragleave", "drop"].forEach((ev) =>
      el.dropzone.addEventListener(ev, (e) => {
        e.preventDefault(); e.stopPropagation();
        el.dropzone.classList.remove("is-drag");
      })
    );
    el.dropzone.addEventListener("drop", (e) => {
      const f = e.dataTransfer.files && e.dataTransfer.files[0];
      pickFile(f);
    });

    el.runBtn.addEventListener("click", runPrediction);
    el.demoBtn.addEventListener("click", loadDemo);

    el.toggleBuildings.addEventListener("click", () => {
      const pressed = el.toggleBuildings.getAttribute("aria-pressed") === "true";
      const next = !pressed;
      MAP.toggleBuildings(next);
      el.toggleBuildings.setAttribute("aria-pressed", String(next));
    });
    el.basemapBtn.addEventListener("click", () => {
      const label = MAP.cycleBasemap();
      el.basemapBtn.textContent = label;
    });
    el.fitBtn.addEventListener("click", () => MAP.fit());

    window.addEventListener("resize", () => MAP.invalidate());
  }

  // ---- Boot ----
  document.addEventListener("DOMContentLoaded", () => {
    MAP.init("map");
    el.basemapBtn.textContent = MAP.currentBasemapLabel();
    bind();
    checkHealth();
  });
})();
