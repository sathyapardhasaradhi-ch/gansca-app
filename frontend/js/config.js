/* Central frontend configuration. No logic here -- just constants so the
   rest of the app never hard-codes URLs, colours, or timings. */
window.GANSCA_CONFIG = {
  // Same origin as the Flask server that serves this page.
  API_BASE: "",

  POLL_INTERVAL_MS: 1200,
  POLL_TIMEOUT_MS: 20 * 60 * 1000, // give large scenes up to 20 min

  ENDPOINTS: {
    health:      "/api/health",
    predict:     "/api/predict",
    job:         (id) => `/api/jobs/${id}`,
    meta:        (id) => `/api/results/${id}/meta`,
    vectorWgs84: (id) => `/api/results/${id}/vector_wgs84`,
    vector:      (id) => `/api/results/${id}/vector`,
    mask:        (id) => `/api/results/${id}/mask`,
    preview:     (id) => `/api/results/${id}/preview.png`,
  },

  SAMPLE_RESULT_ID: "sample",

  MAP: {
    initialCenter: [22.9, 88.4], // eastern India fallback
    initialZoom: 5,
    basemaps: {
      imagery: {
        label: "Imagery",
        url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution: "Tiles &copy; Esri, Maxar, Earthstar Geographics",
        maxZoom: 21,
      },
      streets: {
        label: "Streets",
        url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
        maxZoom: 20,
      },
    },
    defaultBasemap: "imagery",
  },

  BUILDING_STYLE: {
    color: "#ff4d4d",
    weight: 1,
    opacity: 0.9,
    fillColor: "#ff4d4d",
    fillOpacity: 0.28,
  },
  BUILDING_STYLE_HOVER: {
    weight: 2,
    fillOpacity: 0.5,
  },
};
