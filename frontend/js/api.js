/* API client. Every network call to the backend goes through here so error
   handling and URL construction live in exactly one place. */
window.GANSCA_API = (function () {
  const C = window.GANSCA_CONFIG;

  async function _json(res) {
    let body = null;
    try { body = await res.json(); } catch (_) { /* non-JSON */ }
    if (!res.ok) {
      const msg = (body && (body.error || body.detail)) || `Request failed (${res.status})`;
      const err = new Error(msg);
      err.status = res.status;
      err.body = body;
      throw err;
    }
    return body;
  }

  return {
    async health() {
      return _json(await fetch(C.API_BASE + C.ENDPOINTS.health));
    },

    /** Upload a GeoTIFF and start a prediction job. Resolves to { job_id }. */
    async predict(file) {
      const fd = new FormData();
      fd.append("image", file);
      return _json(await fetch(C.API_BASE + C.ENDPOINTS.predict, {
        method: "POST",
        body: fd,
      }));
    },

    async jobStatus(jobId) {
      return _json(await fetch(C.API_BASE + C.ENDPOINTS.job(jobId)));
    },

    async resultMeta(resultId) {
      return _json(await fetch(C.API_BASE + C.ENDPOINTS.meta(resultId)));
    },

    async vectorWgs84(resultId) {
      const res = await fetch(C.API_BASE + C.ENDPOINTS.vectorWgs84(resultId));
      if (!res.ok) throw new Error(`Could not load footprints (${res.status})`);
      return res.json();
    },

    // Direct URLs for <a download> and <img>.
    urls: {
      mask:        (id) => C.API_BASE + C.ENDPOINTS.mask(id),
      vector:      (id) => C.API_BASE + C.ENDPOINTS.vector(id),
      vectorWgs84: (id) => C.API_BASE + C.ENDPOINTS.vectorWgs84(id),
      preview:     (id) => C.API_BASE + C.ENDPOINTS.preview(id),
    },
  };
})();
