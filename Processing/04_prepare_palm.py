"""Keep oil-palm pixels from Descals et al. (2024) that fall on New Guinea, as icon points."""

import zipfile

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from shapely.geometry import Point

from figure_layout import load_config, project_path

# About 8 km. One palm icon per cell that contains closed-canopy oil palm.
CELL_DEG = 0.08


def tiles_on_island(zip_path, bounds):
    minx, miny, maxx, maxy = bounds
    selected = []
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".tif")]
    print(f"Scanning {len(names)} oil-palm tiles...")
    for index, name in enumerate(names, start=1):
        if index % 100 == 0:
            print(f"  {index}/{len(names)}")
        uri = f"/vsizip/{zip_path.as_posix()}/{name}"
        with rasterio.open(uri) as source:
            left, bottom, right, top = source.bounds
        if right < minx or left > maxx or top < miny or bottom > maxy:
            continue
        selected.append(name)
    print(f"Tiles on the island: {len(selected)}")
    return selected


def cell_centers(zip_path, name):
    uri = f"/vsizip/{zip_path.as_posix()}/{name}"
    with rasterio.open(uri) as source:
        height = max(1, int(round((source.bounds.top - source.bounds.bottom) / CELL_DEG)))
        width = max(1, int(round((source.bounds.right - source.bounds.left) / CELL_DEG)))
        values = source.read(1, out_shape=(height, width), resampling=Resampling.max)
        transform = source.transform * Affine.scale(source.width / values.shape[1], source.height / values.shape[0])
    palm = values > 0
    rows, cols = np.nonzero(palm)
    if rows.size == 0:
        return []
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    return list(zip(xs, ys))


def main():
    config = load_config()
    aoi = gpd.read_file(project_path(config["files"]["context_gpkg"]), layer="aoi")
    zip_path = project_path(config["files"]["palm_zip"])
    pad = 0.2
    minx, miny, maxx, maxy = aoi.total_bounds
    names = tiles_on_island(zip_path, (minx - pad, miny - pad, maxx + pad, maxy + pad))
    coords = []
    for name in names:
        found = cell_centers(zip_path, name)
        print(f"  {name}: {len(found)} cells")
        coords.extend(found)
    if not coords:
        raise SystemExit("No oil palm cells on the island.")
    points = gpd.GeoDataFrame(
        {"source": "Descals et al. 2024"},
        geometry=[Point(xy) for xy in coords],
        crs="EPSG:4326",
    )
    points["lon_r"] = points.geometry.x.round(2)
    points["lat_r"] = points.geometry.y.round(2)
    points = points.drop_duplicates(["lon_r", "lat_r"])
    points = gpd.clip(points, aoi)
    points = points[["source", "geometry"]]
    destination = project_path(config["files"]["palm_gpkg"])
    if destination.exists():
        destination.unlink()
    points.to_file(destination, layer="palm", driver="GPKG")
    print(f"Wrote {destination} ({len(points)} icons)")


if __name__ == "__main__":
    main()
