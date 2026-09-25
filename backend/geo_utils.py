"""
Geospatial helpers that sit between the inference core and the web layer.

The inference core emits a GeoJSON in the *source* CRS of the uploaded image
(e.g. EPSG:32645 / UTM 45N -- projected metres). Web map basemaps (OSM,
Esri) expect WGS84 lat/lon (EPSG:4326). These helpers:

  * reproject the source GeoJSON to WGS84 for display,
  * compute a per-building footprint area in m^2 from the *projected*
    geometry (accurate because the source CRS is already in metres),
  * derive lat/lon bounds + centre so the frontend can fit the map,
  * read raster metadata,
  * rasterise the RGBA mask to a browser-viewable PNG thumbnail.

Nothing here alters the model output -- it only prepares that output for
transport and display.
"""

import json

import numpy as np
import rasterio
from rasterio.warp import transform_geom


# -------------------------------------------------------------------------
# Area (planar shoelace on projected coordinates -> square metres)
# -------------------------------------------------------------------------

def _ring_area(ring):
    """Absolute planar area of a single linear ring via the shoelace formula."""
    n = len(ring)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def polygon_area_m2(geometry):
    """
    Area in square metres for a Polygon or MultiPolygon whose coordinates are
    in a projected CRS measured in metres. Subtracts interior rings (holes).
    """
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    area = 0.0
    if gtype == "Polygon":
        polys = [coords]
    elif gtype == "MultiPolygon":
        polys = coords
    else:
        return 0.0
    for poly in polys:
        if not poly:
            continue
        area += _ring_area(poly[0])                       # exterior
        for hole in poly[1:]:
            area -= _ring_area(hole)                       # holes
    return max(area, 0.0)


# -------------------------------------------------------------------------
# Reprojection + enrichment
# -------------------------------------------------------------------------

def _iter_coords(coords):
    """Yield every [lon, lat] pair from an arbitrarily nested coordinate list."""
    if not coords:
        return
    if isinstance(coords[0], (int, float)):
        yield coords
    else:
        for c in coords:
            yield from _iter_coords(c)


def reproject_and_enrich(src_geojson, src_epsg):
    """
    Reproject a source-CRS building GeoJSON to WGS84 and attach an `area_m2`
    property to every feature.

    Returns
    -------
    (wgs84_geojson, enriched_source_geojson, stats)
      stats = {
        "count", "total_area_m2",
        "bounds_latlon": [[south, west], [north, east]] or None,
        "center_latlon": [lat, lon] or None,
      }
    """
    src_crs = f"EPSG:{src_epsg}" if isinstance(src_epsg, int) else src_epsg

    wgs_features = []
    enriched_src_features = []
    lons, lats = [], []

    for feat in src_geojson.get("features", []):
        geom = feat.get("geometry")
        if geom is None:
            continue
        area = round(polygon_area_m2(geom), 2)

        # enrich the source (projected) feature
        src_props = dict(feat.get("properties", {}))
        src_props["area_m2"] = area
        enriched_src_features.append({
            "type": "Feature", "properties": src_props, "geometry": geom,
        })

        # reproject to WGS84 for the map
        if isinstance(src_epsg, int):
            wgs_geom = transform_geom(src_crs, "EPSG:4326", geom)
            for lon, lat in _iter_coords(wgs_geom.get("coordinates", [])):
                lons.append(lon)
                lats.append(lat)
        else:
            wgs_geom = geom  # unknown CRS: pass through untouched

        wgs_props = dict(feat.get("properties", {}))
        wgs_props["area_m2"] = area
        wgs_features.append({
            "type": "Feature", "properties": wgs_props, "geometry": wgs_geom,
        })

    total_area = round(sum(f["properties"]["area_m2"] for f in wgs_features), 2)

    bounds = center = None
    if lons and lats:
        south, north = min(lats), max(lats)
        west, east = min(lons), max(lons)
        bounds = [[south, west], [north, east]]
        center = [(south + north) / 2.0, (west + east) / 2.0]

    wgs84_geojson = {
        "type": "FeatureCollection",
        "name": "building_footprints_wgs84",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": wgs_features,
    }
    enriched_source_geojson = {
        "type": "FeatureCollection",
        "name": src_geojson.get("name", "building_footprints"),
        "crs": src_geojson.get("crs"),
        "features": enriched_src_features,
    }

    stats = {
        "count": len(wgs_features),
        "total_area_m2": total_area,
        "bounds_latlon": bounds,
        "center_latlon": center,
    }
    return wgs84_geojson, enriched_source_geojson, stats


# -------------------------------------------------------------------------
# Raster metadata + preview
# -------------------------------------------------------------------------

def raster_metadata(path):
    """Lightweight metadata read for validation / display."""
    with rasterio.open(path) as src:
        epsg = None
        if src.crs is not None:
            try:
                epsg = src.crs.to_epsg()
            except Exception:
                epsg = None
        b = src.bounds
        return {
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": src.dtypes[0],
            "crs_epsg": epsg,
            "crs_raw": str(src.crs) if src.crs is not None else None,
            "bounds": [b.left, b.bottom, b.right, b.top],
        }


def is_readable_raster(path):
    """True if rasterio can open the file as a raster (used for upload validation)."""
    try:
        with rasterio.open(path) as src:
            _ = (src.width, src.height, src.count)
        return True
    except Exception:
        return False


def mask_to_png(tif_path, png_path, max_dim=1400):
    """
    Convert the RGBA GeoTIFF mask to a browser-viewable PNG thumbnail.
    Downsamples so the longest side is <= max_dim to keep the file small.
    """
    from PIL import Image

    with rasterio.open(tif_path) as src:
        h, w = src.height, src.width
        scale = min(1.0, max_dim / float(max(h, w)))
        out_h, out_w = max(1, int(h * scale)), max(1, int(w * scale))
        data = src.read(
            out_shape=(src.count, out_h, out_w),
            resampling=rasterio.enums.Resampling.nearest,
        )

    if data.shape[0] >= 4:
        arr = np.transpose(data[:4], (1, 2, 0))
        mode = "RGBA"
    elif data.shape[0] == 3:
        arr = np.transpose(data[:3], (1, 2, 0))
        mode = "RGB"
    else:
        arr = data[0]
        mode = "L"

    Image.fromarray(arr.astype(np.uint8), mode=mode).save(png_path, optimize=True)
    return png_path
