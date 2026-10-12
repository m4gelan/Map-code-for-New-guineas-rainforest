"""Shared A4 layout: 1/8 title and locator, 5/8 map, 2/8 legend and QR placeholder."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Rectangle
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "map_config.yaml"


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(relative):
    return ROOT / relative


def largest_polygon(geometry):
    if geometry.geom_type == "MultiPolygon":
        return max(geometry.geoms, key=lambda part: part.area)
    return geometry


def map_limits(aoi_projected):
    """Limits around the main island, with extra room at the lower left for the scale bar."""
    main = largest_polygon(aoi_projected.geometry.iloc[0])
    minx, miny, maxx, maxy = main.bounds
    width = maxx - minx
    height = maxy - miny
    return (
        minx - width * 0.045,
        maxx + width * 0.012,
        miny - height * 0.055,
        maxy + height * 0.02,
    )


def add_scale_bar(ax, length_m, color):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x = xmin + 0.045 * (xmax - xmin)
    y = ymin + 0.055 * (ymax - ymin)
    tick = 0.012 * (ymax - ymin)
    ax.plot([x, x + length_m], [y, y], color=color, linewidth=2.4, solid_capstyle="butt", zorder=6)
    ax.plot([x, x], [y - tick, y + tick], color=color, linewidth=1.2, zorder=6)
    ax.plot([x + length_m, x + length_m], [y - tick, y + tick], color=color, linewidth=1.2, zorder=6)
    ax.text(
        x + length_m / 2,
        y + tick * 1.4,
        f"{int(length_m / 1000)} km",
        ha="center",
        va="bottom",
        fontsize=8,
        color=color,
        zorder=6,
    )


def add_north_arrow(ax, color):
    ax.annotate(
        "N",
        xy=(0.075, 0.20),
        xytext=(0.075, 0.125),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        color=color,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.2},
        zorder=6,
    )


def add_locator(header, config, aoi_wgs84):
    """Equal Earth world with a red dot on the main island."""
    countries = gpd.read_file(project_path(config["files"]["countries"]))
    countries["geometry"] = countries.geometry.simplify(0.2, preserve_topology=True)
    countries = countries.to_crs(config["preview"]["locator_crs"])
    axis = header.inset_axes([0.78, 0.04, 0.21, 0.92])
    axis.set_facecolor(config["colors"]["inset_ocean"])
    countries.plot(ax=axis, color=config["colors"]["inset_land"], linewidth=0)
    point = largest_polygon(aoi_wgs84.geometry.iloc[0]).representative_point()
    located = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(config["preview"]["locator_crs"])
    located.plot(
        ax=axis,
        color=config["colors"]["inset_dot"],
        markersize=8,
        marker="o",
        zorder=3,
    )
    # Equal Earth, framed on Asia–Australia so the dot sits on the island.
    # Longitude limits are taken on the equator, where this projection is widest.
    frame = gpd.GeoSeries(
        [Point(70, 0), Point(180, 0), Point(140, -48), Point(140, 28)],
        crs="EPSG:4326",
    ).to_crs(config["preview"]["locator_crs"])
    axis.set_xlim(frame.iloc[0].x, frame.iloc[1].x)
    axis.set_ylim(frame.iloc[2].y, frame.iloc[3].y)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(config["colors"]["inset_frame"])
        spine.set_linewidth(0.5)


def add_qr_placeholder(legend_ax, config):
    fig = legend_ax.figure
    position = legend_ax.get_position()
    width_in = position.width * fig.get_figwidth()
    height_in = position.height * fig.get_figheight()
    side_in = height_in * 0.62
    rect_w = side_in / width_in
    rect_h = side_in / height_in
    x = 1.0 - rect_w - 0.015
    y = (1.0 - rect_h) / 2
    legend_ax.add_patch(
        Rectangle(
            (x, y),
            rect_w,
            rect_h,
            fill=False,
            linestyle=(0, (2, 1.5)),
            edgecolor=config["colors"]["city_label"],
            linewidth=0.9,
            transform=legend_ax.transAxes,
            zorder=3,
        )
    )
    legend_ax.text(
        x + rect_w / 2,
        y + rect_h / 2,
        config["preview"]["qr_label"],
        ha="center",
        va="center",
        fontsize=11,
        color=config["colors"]["city_label"],
        transform=legend_ax.transAxes,
        zorder=4,
    )


def setup_figure(config, aoi_wgs84, title=None, subtitle=None):
    preview = config["preview"]
    fig = plt.figure(figsize=tuple(preview["figsize_in"]), dpi=preview["dpi"])
    fig.patch.set_facecolor(config["colors"]["figure_background"])
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=preview["height_ratios"],
        left=0.012,
        right=0.988,
        top=0.985,
        bottom=0.012,
        hspace=0.03,
    )
    header = fig.add_subplot(grid[0])
    map_ax = fig.add_subplot(grid[1])
    legend_ax = fig.add_subplot(grid[2])
    header.set_axis_off()
    legend_ax.set_axis_off()
    map_ax.set_facecolor(config["colors"]["ocean"])
    header.text(
        0.0,
        0.62,
        title or preview["title"],
        fontsize=18,
        color=config["colors"]["city_label"],
        ha="left",
        va="center",
        transform=header.transAxes,
    )
    header.text(
        0.0,
        0.22,
        subtitle or preview["subtitle"],
        fontsize=9,
        color="#3D4654",
        ha="left",
        va="center",
        transform=header.transAxes,
    )
    add_locator(header, config, aoi_wgs84)
    return fig, map_ax, legend_ax


def draw_legend(legend_ax, handles, config, title="Ecoregion class", ncol=3, fontsize=8):
    legend_ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.0, 0.45),
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        title=title,
        title_fontsize=9,
    )
    add_qr_placeholder(legend_ax, config)


def save_figure(fig, relative):
    destination = project_path(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=fig.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {destination}")
