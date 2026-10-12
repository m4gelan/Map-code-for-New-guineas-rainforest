"""Clip GEM energy sites and concession polygons to the New Guinea island mask.

Oil palm is not included: the GeoRSPO export is not in Data/07_concessions yet.
HCV probability surfaces are not concession layers and are left unused.
"""

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from figure_layout import load_config, project_path

DROP_STATUS = ("cancel", "retired", "abandon", "decommission", "shelved", "mothbal")


def active(frame, column):
    if column not in frame.columns:
        return frame
    text = frame[column].fillna("").astype(str).str.lower()
    drop = False
    for token in DROP_STATUS:
        drop = drop | text.str.contains(token, regex=False)
    return frame.loc[~drop].copy()


def countries(frame, column):
    text = frame[column].fillna("").astype(str)
    keep = text.str.contains("Papua New Guinea", case=False) | text.str.contains("Indonesia", case=False)
    return frame.loc[keep].copy()


def points_from_latlon(frame, lat_col, lon_col):
    frame = frame.dropna(subset=[lat_col, lon_col]).copy()
    geometry = [Point(xy) for xy in zip(frame[lon_col].astype(float), frame[lat_col].astype(float))]
    return gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326")


def clip_points(frame, region):
    if frame.empty:
        return frame
    return gpd.clip(frame, region)


def extract_zip_member(zip_path, member, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extract(member, destination)
    return destination / member


def extract_nested_gpkg(zip_path, inner_zip, gpkg_name, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        payload = archive.read(inner_zip)
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        inner.extract(gpkg_name, destination)
    return destination / gpkg_name


def slim(frame, columns):
    keep = [name for name in columns if name in frame.columns]
    return frame[keep + ["geometry"]].copy()


def prepare_fields(config, region):
    frame = pd.read_excel(
        project_path(config["files"]["goget_xlsx"]),
        sheet_name="Field-level main data",
    )
    frame = countries(frame, "Country/Area")
    frame = active(frame, "Status")
    points = points_from_latlon(frame, "Latitude", "Longitude")
    points = clip_points(points, region)
    points["lon_r"] = points.geometry.x.round(2)
    points["lat_r"] = points.geometry.y.round(2)
    points = points.drop_duplicates(["lon_r", "lat_r"])
    return slim(points, ["Unit Name", "Fuel type", "Country/Area", "Status", "Location accuracy"])


def prepare_power(config, region):
    frame = pd.read_excel(
        project_path(config["files"]["gogpt_xlsx"]),
        sheet_name="Gas & Oil Units",
    )
    frame = countries(frame, "Country/Area")
    frame = active(frame, "Status")
    points = points_from_latlon(frame, "Latitude", "Longitude")
    points = clip_points(points, region)
    points["_operating"] = points["Status"].fillna("").str.lower().eq("operating").astype(int)
    points = points.sort_values("_operating", ascending=False)
    points = points.drop_duplicates(["Plant name"])
    return slim(points, ["Plant name", "Fuel", "Country/Area", "Status", "Capacity (MW)", "Location accuracy"])


def prepare_lng(config, region):
    cache = project_path("Processing/output/cache")
    gpkg = extract_zip_member(
        project_path(config["files"]["lng_zip"]),
        "GEM-GGIT-LNG-Terminals-2025-09.gpkg",
        cache,
    )
    frame = gpd.read_file(gpkg)
    country_col = next(name for name in frame.columns if "country" in name.lower() or "Country" in name)
    status_col = next((name for name in frame.columns if name.lower() == "status"), None)
    frame = countries(frame, country_col)
    if status_col:
        frame = active(frame, status_col)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    frame = frame.to_crs("EPSG:4326")
    frame = clip_points(frame, region)
    name_col = next((name for name in ("TerminalName", "Name", "ProjectName") if name in frame.columns), None)
    keep = [name for name in (name_col, country_col, status_col) if name]
    return slim(frame, keep)


def prepare_pipelines(config, region):
    cache = project_path("Processing/output/cache")
    gpkg = extract_nested_gpkg(
        project_path(config["files"]["pipelines_zip"]),
        "GEM-GGIT-Gas-Pipelines-2025-11.zip",
        "GEM-GGIT-Gas-Pipelines-2025-11.gpkg",
        cache,
    )
    frame = gpd.read_file(gpkg)
    country_col = next(name for name in frame.columns if "countr" in name.lower())
    status_col = next((name for name in frame.columns if name.lower() == "status"), None)
    frame = countries(frame, country_col)
    if status_col:
        frame = active(frame, status_col)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    frame = frame.to_crs("EPSG:4326")
    frame = gpd.clip(frame, region)
    name_col = next((name for name in ("PipelineName", "Name") if name in frame.columns), None)
    return slim(frame, [name for name in (name_col, country_col, status_col) if name])


def prepare_logging(config, aoi):
    path = project_path(config["files"]["logging_zip"])
    bounds = tuple(aoi.total_bounds)
    frame = gpd.read_file(path, bbox=bounds)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    frame = frame.to_crs("EPSG:4326")
    frame = gpd.clip(frame, aoi)
    frame["geometry"] = frame.geometry.simplify(0.01)
    frame = frame[~frame.geometry.is_empty]
    name_col = next((name for name in frame.columns if name.lower() in {"name", "concession", "company"}), None)
    country_col = next((name for name in frame.columns if "countr" in name.lower()), None)
    return slim(frame, [name for name in (name_col, country_col) if name])


def prepare_mining(config, aoi):
    cache = project_path("Processing/output/cache")
    gpkg = extract_zip_member(
        project_path(config["files"]["mining_zip"]),
        "global_mining_polygons_v1.gpkg",
        cache,
    )
    frame = gpd.read_file(gpkg, bbox=tuple(aoi.to_crs("EPSG:4326").total_bounds))
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    frame = frame.to_crs("EPSG:4326")
    frame = gpd.clip(frame, aoi)
    frame["geometry"] = frame.geometry.simplify(0.005)
    frame = frame[~frame.geometry.is_empty]
    frame["source"] = "Maus et al. 2020"
    return slim(frame, ["source"])


def write_layers(path, layers):
    if path.exists():
        path.unlink()
    first = True
    for name, frame in layers:
        print(f"  {name}: {len(frame)}")
        if frame.empty:
            continue
        frame.to_file(path, layer=name, driver="GPKG", mode="w" if first else "a")
        first = False
    if first:
        raise SystemExit("No pressure layers were written.")


def main():
    config = load_config()
    aoi = gpd.read_file(project_path(config["files"]["context_gpkg"]), layer="aoi")
    region = gpd.GeoDataFrame(geometry=aoi.buffer(0.25), crs=aoi.crs)
    print("Oil and gas fields...")
    fields = prepare_fields(config, region)
    print("Power plants...")
    power = prepare_power(config, region)
    print("LNG terminals...")
    lng = prepare_lng(config, region)
    print("Pipelines...")
    pipelines = prepare_pipelines(config, region)
    print("Logging concessions...")
    logging = prepare_logging(config, aoi)
    print("Mining areas...")
    mining = prepare_mining(config, aoi)
    destination = project_path(config["files"]["pressures_gpkg"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_layers(
        destination,
        [
            ("fields", fields),
            ("power", power),
            ("lng", lng),
            ("pipelines", pipelines),
            ("logging", logging),
            ("mining", mining),
        ],
    )
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
