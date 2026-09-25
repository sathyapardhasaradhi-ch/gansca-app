/* Leaflet map wrapper. Owns the map instance, basemap layers, and the
   building footprint overlay. The rest of the app talks to it through a
   small, intention-revealing surface (load / clear / fit / toggle). */
window.GANSCA_MAP = (function () {
  const C = window.GANSCA_CONFIG;

  let map = null;
  let basemapLayers = {};
  let currentBasemapKey = C.MAP.defaultBasemap;
  let buildingLayer = null;
  let lastBounds = null;

  function init(elementId) {
    map = L.map(elementId, { zoomControl: true, attributionControl: true })
      .setView(C.MAP.initialCenter, C.MAP.initialZoom);

    Object.entries(C.MAP.basemaps).forEach(([key, cfg]) => {
      basemapLayers[key] = L.tileLayer(cfg.url, {
        attribution: cfg.attribution,
        maxZoom: cfg.maxZoom,
      });
    });
    basemapLayers[currentBasemapKey].addTo(map);
    return map;
  }

  function setBasemap(key) {
    if (!basemapLayers[key] || key === currentBasemapKey) return currentBasemapKey;
    map.removeLayer(basemapLayers[currentBasemapKey]);
    basemapLayers[key].addTo(map);
    // keep the footprint overlay on top
    if (buildingLayer) buildingLayer.bringToFront();
    currentBasemapKey = key;
    return currentBasemapKey;
  }

  function cycleBasemap() {
    const keys = Object.keys(C.MAP.basemaps);
    const next = keys[(keys.indexOf(currentBasemapKey) + 1) % keys.length];
    setBasemap(next);
    return C.MAP.basemaps[next].label;
  }

  function _onEachFeature(feature, layer) {
    const p = feature.properties || {};
    const area = (p.area_m2 != null) ? `${Number(p.area_m2).toLocaleString()} m²` : "—";
    layer.bindPopup(
      `<div class="bldg-popup"><b>Building</b><br>Footprint: ${area}</div>`
    );
    layer.on("mouseover", () => layer.setStyle(C.BUILDING_STYLE_HOVER));
    layer.on("mouseout", () => layer.setStyle(C.BUILDING_STYLE));
  }

  /** Render a WGS84 GeoJSON FeatureCollection of footprints. */
  function loadBuildings(geojson) {
    clearBuildings();
    buildingLayer = L.geoJSON(geojson, {
      style: () => C.BUILDING_STYLE,
      onEachFeature: _onEachFeature,
    }).addTo(map);

    try {
      lastBounds = buildingLayer.getBounds();
      if (lastBounds.isValid()) {
        map.fitBounds(lastBounds, { padding: [40, 40], maxZoom: 19 });
      }
    } catch (_) { lastBounds = null; }
    return buildingLayer;
  }

  function clearBuildings() {
    if (buildingLayer) { map.removeLayer(buildingLayer); buildingLayer = null; }
  }

  function toggleBuildings(show) {
    if (!buildingLayer) return show;
    if (show) { buildingLayer.addTo(map); buildingLayer.bringToFront(); }
    else { map.removeLayer(buildingLayer); }
    return show;
  }

  function fit() {
    if (lastBounds && lastBounds.isValid()) {
      map.fitBounds(lastBounds, { padding: [40, 40], maxZoom: 19 });
    }
  }

  function invalidate() { if (map) setTimeout(() => map.invalidateSize(), 60); }

  return {
    init, setBasemap, cycleBasemap,
    loadBuildings, clearBuildings, toggleBuildings, fit, invalidate,
    currentBasemapLabel: () => C.MAP.basemaps[currentBasemapKey].label,
  };
})();
