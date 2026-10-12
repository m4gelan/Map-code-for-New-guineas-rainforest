"""Summarise Hansen tree-cover loss and clip PEATMAP to the New Guinea island mask.

Global Mangrove Watch v4.1.12 in Data/04_mangroves is checked and skipped when the
tiles fall outside the island. CIFOR tropical wetlands are not required: RESOLVE
already maps mangrove and freshwater swamp forest, and PEATMAP supplies peat extent.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine, from_origin

from figure_layout import (
    add_north_arrow,
    add_scale_bar,
    draw_legend,
    load_config,
    map_limits,
    project_path,
    save_figure,
    setup_figure,
)

# Every 25th 30 m pixel. Loss is then counted inside 0.1 degree cells.
SAMPLE_FACTOR = 25
CELL_DEG = 0.1
CANOPY_THRESHOLD = 30
GRID_BOUNDS = (129.5, 155.0, -12.0, 1.5)  # xmin, xmax, ymin, ymax
TILES = ("00N_130E", "00N_140E", "00N_150E", "10S_140E", "10S_150E")
LOSS_BREAKS = (0.01, 0.05, 0.15, 0.30, 1.01)


def tile_paths(folder, suffix):
    loss = folder / f"Hansen_GFC-2024-v1.12_lossyear_{suffix}.tif"
    cover = folder / f"Hansen_GFC-2024-v1.12_treecover2000_{suffix}.tif"
    missing = [path.name for path in (loss, cover) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {folder}")
    return loss, cover


def accumulate_tile(loss_path, cover_path, forest_count, loss_count, xmin, ymax):
    with rasterio.open(loss_path) as loss_src, rasterio.open(cover_path) as cover_src:
        height = loss_src.height // SAMPLE_FACTOR
        width = loss_src.width // SAMPLE_FACTOR
        loss = loss_src.read(1, out_shape=(height, width), resampling=Resampling.nearest)
        cover = cover_src.read(1, out_shape=(height, width), resampling=Resampling.nearest)
        sampled = loss_src.transform * Affine.scale(loss_src.width / width, loss_src.height / height)

    forest = cover >= CANOPY_THRESHOLD
    rows, cols = np.nonzero(forest)
    if rows.size == 0:
        return
    xs, ys = rasterio.transform.xy(sampled, rows, cols, offset="center")
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    col_idx = np.floor((xs - xmin) / CELL_DEG).astype(np.int32)
    row_idx = np.floor((ymax - ys) / CELL_DEG).astype(np.int32)
    inside = (
        (col_idx >= 0)
        & (col_idx < forest_count.shape[1])
        & (row_idx >= 0)
        & (row_idx < forest_count.shape[0])
    )
    np.add.at(forest_count, (row_idx[inside], col_idx[inside]), 1)
    lost = forest & (loss > 0)
    rows, cols = np.nonzero(lost)
    if rows.size == 0:
        return
    xs, ys = rasterio.transform.xy(sampled, rows, cols, offset="center")
    col_idx = np.floor((np.asarray(xs) - xmin) / CELL_DEG).astype(np.int32)
    row_idx = np.floor((ymax - np.asarray(ys)) / CELL_DEG).astype(np.int32)
    inside = (
        (col_idx >= 0)
        & (col_idx < forest_count.shape[1])
        & (row_idx >= 0)
        & (row_idx < forest_count.shape[0])
    )
    np.add.at(loss_count, (row_idx[inside], col_idx[inside]), 1)


def build_loss_grid(config, aoi):
    xmin, xmax, ymin, ymax = GRID_BOUNDS
    cols = int(round((xmax - xmin) / CELL_DEG))
    rows = int(round((ymax - ymin) / CELL_DEG))
    forest_count = np.zeros((rows, cols), dtype=np.int32)
    loss_count = np.zeros((rows, cols), dtype=np.int32)
    folder = project_path(config["files"]["forest_loss_dir"])
    for suffix in TILES:
        loss_path, cover_path = tile_paths(folder, suffix)
        print(f"Sampling {suffix}...")
        accumulate_tile(loss_path, cover_path, forest_count, loss_count, xmin, ymax)

    transform = from_origin(xmin, ymax, CELL_DEG, CELL_DEG)
    island = rasterize(
        [(geom, 1) for geom in aoi.geometry],
        out_shape=(rows, cols),
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    fraction = np.full((rows, cols), np.nan, dtype="float32")
    usable = (island == 1) & (forest_count > 0)
    fraction[usable] = loss_count[usable] / forest_count[usable]

    destination = project_path(config["files"]["forest_loss_tif"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        destination,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
        compress="deflate",
    ) as dataset:
        dataset.write(fraction, 1)
        dataset.update_tags(
            source="Hansen/UMD Global Forest Change v1.12",
            years="2001-2024",
            canopy_threshold=str(CANOPY_THRESHOLD),
            sample=f"nearest pixel every {SAMPLE_FACTOR} native pixels, counted in {CELL_DEG} degree cells",
            trust="observed disturbance, not a legal deforestation finding",
        )
    print(f"Wrote {destination}")
    return fraction, transform


def plot_loss(config, fraction, transform, aoi):
    mask = np.isfinite(fraction) & (fraction >= LOSS_BREAKS[0])
    features = []
    for geom, value in shapes(fraction.astype("float32"), mask=mask, transform=transform):
        features.append(
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {"fraction": float(value)},
            }
        )
    cells = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    cells["class_id"] = np.digitize(cells["fraction"], LOSS_BREAKS[1:-1], right=False)
    cells = cells.to_crs(config["crs"]["map"])

    gpkg = project_path(config["files"]["context_gpkg"])
    crs = config["crs"]["map"]
    land = gpd.read_file(gpkg, layer="land").to_crs(crs)
    coastline = gpd.read_file(gpkg, layer="coastline").to_crs(crs)
    places = gpd.read_file(gpkg, layer="places").to_crs(crs)
    aoi_map = aoi.to_crs(crs)
    colors = config["colors"]
    styles = colors["forest_loss"]

    fig, ax, legend_ax = setup_figure(
        config,
        aoi,
        title=config["preview"]["forest_title"],
        subtitle=config["preview"]["forest_subtitle"],
    )
    land.plot(ax=ax, color=colors["land"], linewidth=0, zorder=0)
    for class_id, style in enumerate(styles):
        subset = cells[cells["class_id"] == class_id]
        if subset.empty:
            continue
        subset.plot(ax=ax, color=style["fill"], linewidth=0, zorder=1)
    coastline.plot(ax=ax, color=colors["coastline"], linewidth=0.35, zorder=2)
    wanted = {spec["name"].lower() for spec in config["places"]["labels"]}
    labeled = places[places["name"].fillna("").str.lower().isin(wanted)]
    labeled.plot(ax=ax, color=colors["city_marker"], markersize=12, zorder=3)
    west, east, south, north = map_limits(aoi_map)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    add_scale_bar(ax, config["preview"]["scale_bar_km"] * 1000, colors["scale_bar"])
    add_north_arrow(ax, colors["scale_bar"])
    handles = [Patch(facecolor=style["fill"], edgecolor="none", label=style["label"]) for style in styles]
    draw_legend(legend_ax, handles, config, title="Tree-cover loss, 2001–2024", ncol=2)
    save_figure(fig, config["files"]["forest_loss_png"])


def read_peat_member(zip_path, member):
    return gpd.read_file(f"zip://{zip_path.as_posix()}!{member}")


def clip_peat(config, aoi):
    pieces = []
    members = [
        (project_path(config["files"]["peat_asia"]), "Asia/SEA_Peatland.shp", "Asia"),
        (project_path(config["files"]["peat_oceania"]), "Oceania/Oceania_Peatland.shp", "Oceania"),
    ]
    for path, member, region in members:
        print(f"Reading PEATMAP {region}...")
        peat = read_peat_member(path, member)
        peat = peat.to_crs(aoi.crs)
        clipped = gpd.clip(peat, aoi)
        if clipped.empty:
            print(f"  no peat inside the island mask in {region}")
            continue
        clipped["region"] = region
        clipped["source"] = "PEATMAP"
        clipped["trust"] = "estimate"
        clipped["vintage"] = "source maps mostly 1999-2010; published 2018"
        pieces.append(clipped)
    if not pieces:
        raise SystemExit("PEATMAP clips are empty.")
    peat = gpd.GeoDataFrame(pd_concat(pieces), crs=aoi.crs)
    keep = [column for column in ("region", "source", "trust", "vintage", "geometry") if column in peat.columns]
    peat = peat[keep]
    destination = project_path(config["files"]["peat_gpkg"])
    if destination.exists():
        destination.unlink()
    peat.to_file(destination, layer="peat", driver="GPKG")
    print(f"Wrote {destination} ({len(peat)} features)")
    return peat


def pd_concat(frames):
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


def mangrove_coverage(config, aoi):
    zip_path = project_path(config["files"]["mangrove_zip"])
    if not zip_path.exists():
        print("No mangrove zip found.")
        return
    import zipfile

    lons = []
    lats = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            stem = Path(name).name
            if "E" not in stem:
                continue
            # GMW_N05E119_v4112_mng_ext.tif
            token = stem.split("_")[1]
            hemisphere_lat = token[0]
            hemisphere_lon = "E" if "E" in token else "W"
            lat_text, lon_text = token[1:].split(hemisphere_lon)
            lat = int(lat_text) * (1 if hemisphere_lat == "N" else -1)
            lon = int(lon_text) * (1 if hemisphere_lon == "E" else -1)
            lats.append(lat)
            lons.append(lon)
    if not lons:
        print(f"No GMW tiles parsed in {zip_path.name}")
        return
    minx, miny, maxx, maxy = aoi.total_bounds
    covers = any(minx <= lon + 1 and lon <= maxx and miny <= lat + 1 and lat <= maxy for lon, lat in zip(lons, lats))
    print(
        f"GMW tiles span {min(lons)}–{max(lons)}°E, {min(lats)}–{max(lats)}° latitude. "
        f"Island is {minx:.1f}–{maxx:.1f}°E, {miny:.1f}–{maxy:.1f}°."
    )
    if not covers:
        print("Skipping mangroves: these tiles do not intersect New Guinea.")
    else:
        print("Mangrove tiles intersect the island. Extent clipping is not in this step.")


def main():
    config = load_config()
    aoi = gpd.read_file(project_path(config["files"]["context_gpkg"]), layer="aoi")
    mangrove_coverage(config, aoi)
    fraction, transform = build_loss_grid(config, aoi)
    plot_loss(config, fraction, transform, aoi)
    clip_peat(config, aoi)


if __name__ == "__main__":
    main()
