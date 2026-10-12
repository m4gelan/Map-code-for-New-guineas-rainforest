"""Ecoregion map of New Guinea on the shared A4 layout."""

import geopandas as gpd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
    frame = gpd.read_file(gpkg, layer=layer)
    return frame.to_crs(crs)


def label_places(ax, places, config):
    icon = config["icons"]["city"]
    wanted = {spec["name"].lower() for spec in config["places"]["labels"]}
    labeled = places[places["name"].fillna("").str.lower().isin(wanted)]
    labeled.plot(
        ax=ax,
        color=config["colors"]["city_marker"],
        markersize=icon["size"],
        marker=icon["marker"],
        edgecolor=icon["edge_color"],
        linewidth=icon["edge_width"],
        zorder=4,
    )
    lookup = {}
    for _, row in labeled.iterrows():
        lookup[str(row["name"]).lower()] = row.geometry
    for spec in config["places"]["labels"]:
        point = lookup.get(spec["name"].lower())
        if point is None:
            print(f"Place not in clipped layer, skipped: {spec['name']}")
            continue
        ax.annotate(
            spec["name"],
            xy=(point.x, point.y),
            xytext=(spec["dx"], spec["dy"]),
            textcoords="offset points",
            ha=spec["ha"],
            va=spec["va"],
            fontsize=8,
            color=config["colors"]["city_label"],
            zorder=6,
        )


def label_rivers(ax, rivers, config):
    if rivers.empty or "name" not in rivers.columns:
        return
    wanted = config["rivers"]["labels"]
    for token in wanted:
        matches = rivers[rivers["name"].fillna("").str.contains(token, case=False, regex=False)]
        if matches.empty:
            print(f"River not in clipped layer, skipped: {token}")
            continue
        longest = max(matches.geometry, key=lambda geom: geom.length)
        point = longest.interpolate(0.55, normalized=True)
        ax.annotate(
            token,
            xy=(point.x, point.y),
            fontsize=7,
            fontstyle="italic",
            color=config["colors"]["river_label"],
            ha="center",
            va="bottom",
            zorder=6,
        )


def legend_handles(config, present_classes):
    colors = config["colors"]
    icons = config["icons"]
    handles = []
    for key in CLASS_ORDER:
        if key not in present_classes:
            continue
        style = colors["ecoregions"][key]
        handles.append(
            Patch(facecolor=style["fill"], edgecolor="#FFFFFF", linewidth=0.4, label=style["label"])
        )
    handles.extend(
        [
            Line2D([0], [0], color=colors["rivers"], linewidth=icons["river"]["width"] * 2, label="Major river"),
            Line2D(
                [0],
                [0],
                color=colors["international_border"],
                linewidth=icons["international_border"]["width"],
                label="International border",
            ),
            Line2D(
                [0],
                [0],
                marker=icons["city"]["marker"],
                color="none",
                markerfacecolor=colors["city_marker"],
                markeredgecolor=icons["city"]["edge_color"],
                markersize=5,
                label="Major city",
            ),
        ]
    )
    return handles


def main():
    config = load_config()
    gpkg = project_path(config["files"]["context_gpkg"])
    if not gpkg.exists():
        raise SystemExit(f"Missing {gpkg}. Run Processing/01_clip_context.py first.")

    crs = config["crs"]["map"]
    aoi = read_layer(gpkg, "aoi", crs)
    admin1 = read_layer(gpkg, "admin1", crs)
    land = read_layer(gpkg, "land", crs)
    coastline = read_layer(gpkg, "coastline", crs)
    rivers = read_layer(gpkg, "rivers", crs)
    lakes = read_layer(gpkg, "lakes", crs)
    places = read_layer(gpkg, "places", crs)
    ecoregions = read_layer(gpkg, "ecoregions", crs)

    colors = config["colors"]
    icons = config["icons"]
    preview = config["preview"]
    aoi_wgs84 = gpd.read_file(gpkg, layer="aoi")

    fig, ax, legend_ax = setup_figure(config, aoi_wgs84)

    land.plot(ax=ax, color=colors["land"], linewidth=0, zorder=0)
    for key in CLASS_ORDER:
        subset = ecoregions[ecoregions["map_class"] == key]
        if subset.empty:
            continue
        subset.plot(ax=ax, color=colors["ecoregions"][key]["fill"], linewidth=0, zorder=1)
    lakes.plot(
        ax=ax,
        color=colors["lakes"],
        edgecolor=icons["lake"]["edge_color"],
        linewidth=icons["lake"]["edge_width"],
        zorder=2,
    )
    rivers.plot(ax=ax, color=colors["rivers"], linewidth=icons["river"]["width"], zorder=3)
    coastline.plot(ax=ax, color=colors["coastline"], linewidth=icons["coastline"]["width"], zorder=3)

    admin1.boundary.plot(ax=ax, color=colors["admin_boundary"], linewidth=icons["admin_boundary"]["width"], zorder=3)
    indonesia = admin1.loc[admin1["admin"] == "Indonesia"].dissolve().geometry.iloc[0]
    png = admin1.loc[admin1["admin"] == "Papua New Guinea"].dissolve().geometry.iloc[0]
    border = indonesia.boundary.intersection(png.boundary.buffer(1500))
    gpd.GeoSeries([border], crs=crs).plot(
        ax=ax,
        color=colors["international_border"],
        linewidth=icons["international_border"]["width"],
        zorder=4,
    )

    label_places(ax, places, config)
    label_rivers(ax, rivers, config)

    west, east, south, north = map_limits(aoi)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    add_scale_bar(ax, preview["scale_bar_km"] * 1000, colors["scale_bar"])
    add_north_arrow(ax, colors["scale_bar"])

    present = set(ecoregions["map_class"].unique())
    draw_legend(legend_ax, legend_handles(config, present), config)
    save_figure(fig, config["files"]["preview_png"])


if __name__ == "__main__":
    main()
