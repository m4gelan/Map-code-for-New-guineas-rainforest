"""One policy map: ecoregions, hatched tree-cover loss, and infrastructure icons."""

import geopandas as gpd
import numpy as np
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.features import shapes
from shapely import contains_xy
from shapely.geometry import shape

from icons import palm as palm_marker
from icons import pickaxe as pickaxe_marker
from icons import refinery as refinery_marker
from figure_layout import (
    add_north_arrow,
    add_scale_bar,
    load_config,
    map_limits,
    project_path,
    save_figure,
    setup_figure,
)

CLASS_ORDER = [
    "lowland_rainforest",
    "montane_forest",
    "alpine_grasslands",
    "mangroves",
    "freshwater_swamp_forest",
    "peat_swamp_forest",
    "other",
]


def read_layer(gpkg, layer, crs):
    return gpd.read_file(gpkg, layer=layer).to_crs(crs)


def class_surface(fraction):
    """Smooth class edges without turning the map back into a pixel grid.

    Cells at or above 50 percent are rare. A one-cell growth keeps those
    hotspots visible after the edge is smoothed.
    """
    from scipy.ndimage import binary_dilation, gaussian_filter

    valid = np.isfinite(fraction)
    bands = (
        (1, valid & (fraction >= 0.05) & (fraction < 0.25), 0),
        (2, valid & (fraction >= 0.25) & (fraction < 0.50), 1),
        (3, valid & (fraction >= 0.50), 1),
    )
    classes = np.zeros(fraction.shape, dtype="uint8")
    for class_id, mask, dilate in bands:
        grown = binary_dilation(mask, iterations=dilate) if dilate else mask
        soft = gaussian_filter(grown.astype("float64"), sigma=1.15) > 0.45
        classes[soft] = class_id
    near_forest = binary_dilation(valid, iterations=1)
    classes[~near_forest] = 0
    return classes


def sample_loss(config, aoi_map, crs):
    with rasterio.open(project_path(config["files"]["forest_loss_tif"])) as source:
        fraction = source.read(1)
        transform = source.transform
    classes = class_surface(fraction)
    west, east, south, north = map_limits(aoi_map)
    xs = np.linspace(west, east, 520)
    ys = np.linspace(south, north, 280)
    xx, yy = np.meshgrid(xs, ys)
    points = gpd.GeoSeries(gpd.points_from_xy(xx.ravel(), yy.ravel()), crs=crs).to_crs("EPSG:4326")
    lon = points.x.to_numpy().reshape(xx.shape)
    lat = points.y.to_numpy().reshape(yy.shape)
    columns = (lon - transform.c) / transform.a
    rows = (lat - transform.f) / transform.e
    from scipy.ndimage import map_coordinates

    sampled = map_coordinates(classes.astype("float64"), [rows, columns], order=0, cval=0)
    island = aoi_map.geometry.iloc[0]
    on_island = contains_xy(island, xx, yy)
    return xs, ys, np.where(on_island, sampled, 0)


def draw_loss_hatch(ax, xs, ys, grid, styles, crs):
    classes = np.rint(grid).astype("uint8")
    if not classes.any():
        print("No tree-cover loss above 5 percent.")
        return
    flipped = np.flipud(classes)
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])
    transform = rasterio.transform.from_origin(float(xs[0]) - dx / 2, float(ys[-1]) + dy / 2, dx, dy)
    records = []
    for geom, value in shapes(flipped, mask=flipped > 0, transform=transform):
        records.append({"geometry": shape(geom), "class_id": int(value)})
    zones = gpd.GeoDataFrame(records, crs=crs)
    zones["geometry"] = zones.geometry.make_valid().simplify(3500)
    zones = zones[~zones.geometry.is_empty].dissolve(by="class_id")
    for class_id, style in enumerate(styles, start=1):
        if class_id not in zones.index:
            continue
        part = zones.loc[[class_id]]
        part.plot(ax=ax, color=style["color"], alpha=0.18, linewidth=0, zorder=2)
        part.plot(
            ax=ax,
            facecolor="none",
            edgecolor=style["color"],
            hatch=style["hatch"],
            linewidth=0.8,
            zorder=3,
        )


def optional_layer(path, layer, crs):
    try:
        frame = gpd.read_file(path, layer=layer)
    except Exception as error:
        print(f"Skipping {layer}: {error}")
        return None
    if frame.empty:
        return None
    return frame.to_crs(crs)


def column_legend(host, handles, title, anchor, ncol=1):
    legend = host.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=anchor,
        frameon=False,
        fontsize=6.5,
        title=title,
        title_fontsize=8,
        ncol=ncol,
        borderaxespad=0,
        labelspacing=0.35,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    host.add_artist(legend)
    return legend


def draw_policy_legend(legend_ax, config, present):
    from figure_layout import add_qr_placeholder

    colors = config["colors"]
    icons = config["icons"]
    biomes = []
    for key in CLASS_ORDER:
        if key not in present:
            continue
        style = colors["ecoregions"][key]
        biomes.append(Patch(facecolor=style["fill"], edgecolor="none", label=style["label"]))
    loss = [
        Patch(
            facecolor=style["color"],
            edgecolor=style["color"],
            hatch=style["hatch"],
            alpha=0.45,
            label=style["label"],
        )
        for style in colors["forest_loss"]
    ]
    energy = [
        Line2D(
            [0],
            [0],
            marker=refinery_marker(),
            color="none",
            markerfacecolor=icons["oil_gas_field"]["color"],
            markeredgecolor=icons["oil_gas_field"]["color"],
            markersize=8,
            label="Oil or gas field",
        ),
        Line2D(
            [0],
            [0],
            marker=refinery_marker(),
            color="none",
            markerfacecolor=icons["lng_terminal"]["color"],
            markeredgecolor=icons["lng_terminal"]["color"],
            markersize=8,
            label="LNG terminal",
        ),
        Line2D(
            [0],
            [0],
            color=icons["pipeline"]["color"],
            linestyle="--",
            linewidth=1.2,
            label="Gas pipeline, onshore",
        ),
    ]
    concessions = [
        Line2D([0], [0], color=icons["logging_concession"]["color"], linewidth=1.3, label="Logging, 2020"),
        Line2D(
            [0],
            [0],
            marker=pickaxe_marker(),
            color="none",
            markerfacecolor=icons["mining_lease"]["color"],
            markeredgecolor=icons["mining_lease"]["color"],
            markersize=8,
            label="Mining area, 2020",
        ),
        Line2D(
            [0],
            [0],
            marker=palm_marker(),
            color="none",
            markerfacecolor=icons["oil_palm"]["color"],
            markeredgecolor=icons["oil_palm"]["color"],
            markersize=8,
            label="Oil palm, 2016–2021",
        ),
    ]
    column_legend(legend_ax, biomes, "Biomes", (0.0, 0.98), ncol=2)
    column_legend(legend_ax, loss, "Tree-cover loss", (0.40, 0.98))
    column_legend(legend_ax, energy, "Oil and gas infrastructure", (0.58, 0.98))
    column_legend(legend_ax, concessions, "Concessions", (0.76, 0.98))
    add_qr_placeholder(legend_ax, config)


def main():
    config = load_config()
    context = project_path(config["files"]["context_gpkg"])
    crs = config["crs"]["map"]
    aoi = read_layer(context, "aoi", crs)
    admin1 = read_layer(context, "admin1", crs)
    land = read_layer(context, "land", crs)
    coastline = read_layer(context, "coastline", crs)
    rivers = read_layer(context, "rivers", crs)
    places = read_layer(context, "places", crs)
    ecoregions = read_layer(context, "ecoregions", crs)
    aoi_wgs84 = gpd.read_file(context, layer="aoi")
    colors = config["colors"]
    icons = config["icons"]

    fig, ax, legend_ax = setup_figure(
        config,
        aoi_wgs84,
        title="New Guinea rainforests, loss, and infrastructure",
        subtitle="Ecoregions are 2017 potential types. Hatches show tree-cover loss, 2001–2024. Below 5% is not marked.",
    )

    land.plot(ax=ax, color=colors["land"], linewidth=0, zorder=0)
    for key in CLASS_ORDER:
        subset = ecoregions[ecoregions["map_class"] == key]
        if subset.empty:
            continue
        subset.plot(ax=ax, color=colors["ecoregions"][key]["fill"], linewidth=0, zorder=1)

    xs, ys, grid = sample_loss(config, aoi, crs)
    draw_loss_hatch(ax, xs, ys, grid, colors["forest_loss"], crs)

    pressures = project_path(config["files"]["pressures_gpkg"])
    logging = optional_layer(pressures, "logging", crs)
    mining = optional_layer(pressures, "mining", crs)
    pipelines = optional_layer(pressures, "pipelines", crs)
    if logging is not None:
        logging.boundary.plot(ax=ax, color=icons["logging_concession"]["color"], linewidth=0.7, zorder=3)
    if mining is not None:
        mines = mining.copy()
        mines["geometry"] = mining.geometry.representative_point()
        mines.plot(
            ax=ax,
            marker=pickaxe_marker(),
            color=icons["mining_lease"]["color"],
            markersize=11,
            linewidth=0.8,
            zorder=6,
        )
    if pipelines is not None:
        onshore = gpd.clip(pipelines, aoi.buffer(1200))
        onshore = onshore[~onshore.geometry.is_empty & onshore.geometry.notna()]
        if not onshore.empty:
            onshore.plot(ax=ax, color=icons["pipeline"]["color"], linewidth=1.05, linestyle="--", zorder=4)

    rivers.plot(ax=ax, color=colors["rivers"], linewidth=0.55, zorder=4)
    coastline.plot(ax=ax, color=colors["coastline"], linewidth=0.35, zorder=4)
    admin1.boundary.plot(ax=ax, color=colors["admin_boundary"], linewidth=0.25, zorder=4)
    indonesia = admin1.loc[admin1["admin"] == "Indonesia"].dissolve().geometry.iloc[0]
    png = admin1.loc[admin1["admin"] == "Papua New Guinea"].dissolve().geometry.iloc[0]
    border = indonesia.boundary.intersection(png.boundary.buffer(1500))
    gpd.GeoSeries([border], crs=crs).plot(ax=ax, color=colors["international_border"], linewidth=0.8, zorder=5)

    fields = optional_layer(pressures, "fields", crs)
    lng = optional_layer(pressures, "lng", crs)
    palm = optional_layer(project_path(config["files"]["palm_gpkg"]), "palm", crs)
    if fields is not None:
        fields.plot(
            ax=ax,
            marker=refinery_marker(),
            color=icons["oil_gas_field"]["color"],
            markersize=13,
            linewidth=0.7,
            zorder=6,
        )
    if lng is not None:
        lng.plot(
            ax=ax,
            marker=refinery_marker(),
            color=icons["lng_terminal"]["color"],
            markersize=12,
            linewidth=0.7,
            zorder=6,
        )
    if palm is not None:
        palm.plot(
            ax=ax,
            marker=palm_marker(),
            color=icons["oil_palm"]["color"],
            markersize=9,
            linewidth=0.6,
            zorder=6,
        )

    wanted = {spec["name"].lower() for spec in config["places"]["labels"]}
    labeled = places[places["name"].fillna("").str.lower().isin(wanted)]
    labeled.plot(ax=ax, color=colors["city_marker"], markersize=14, zorder=6)
    lookup = {str(row["name"]).lower(): row.geometry for _, row in labeled.iterrows()}
    for spec in config["places"]["labels"]:
        point = lookup.get(spec["name"].lower())
        if point is None:
            continue
        ax.annotate(
            spec["name"],
            xy=(point.x, point.y),
            xytext=(spec["dx"], spec["dy"]),
            textcoords="offset points",
            ha=spec["ha"],
            va=spec["va"],
            fontsize=7.5,
            color=colors["city_label"],
            zorder=7,
        )

    west, east, south, north = map_limits(aoi)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    add_scale_bar(ax, config["preview"]["scale_bar_km"] * 1000, colors["scale_bar"])
    add_north_arrow(ax, colors["scale_bar"])
    draw_policy_legend(legend_ax, config, set(ecoregions["map_class"].unique()))
    save_figure(fig, config["files"]["policy_png"])


if __name__ == "__main__":
    main()
