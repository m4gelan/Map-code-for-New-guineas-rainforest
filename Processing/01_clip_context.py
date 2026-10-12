"""Open the Natural Earth and RESOLVE zip files, build the New Guinea island mask, and clip."""

from pathlib import Path

import geopandas as gpd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "map_config.yaml"


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(relative):
    return ROOT / relative


def lower_columns(frame):
    renamed = frame.copy()
    renamed.columns = ["geometry" if name == "geometry" else str(name).lower() for name in renamed.columns]
    return renamed


def read_layer(relative, bbox=None):
    path = project_path(relative)
    if not path.exists():
        raise FileNotFoundError(path)
    if bbox is None:
        frame = gpd.read_file(path)
    else:
        frame = gpd.read_file(path, bbox=tuple(bbox))
    return lower_columns(frame)


def valid_geometry(frame):
    cleaned = frame.copy()
    cleaned["geometry"] = cleaned.geometry.make_valid()
    cleaned = cleaned[~cleaned.geometry.is_empty & cleaned.geometry.notna()]
    return cleaned


def select_admin1(admin1, config):
    names = set(config["aoi"]["indonesia_admin1"]) | set(config["aoi"]["png_admin1"])
    indonesia = admin1["admin"].eq("Indonesia") & admin1["name"].isin(names)
    png = admin1["admin"].eq("Papua New Guinea") & admin1["name"].isin(names)
    selected = admin1.loc[indonesia | png].copy()
    missing = names - set(selected["name"])
    if missing:
        raise SystemExit(f"Admin-1 names not found in Natural Earth: {sorted(missing)}")
    keep = ["name", "name_en", "admin", "iso_3166_2", "geometry"]
    return valid_geometry(selected[keep])


def build_aoi(admin1):
    mask = admin1.dissolve(by="admin", as_index=False)
    mask = mask.dissolve()
    mask = valid_geometry(mask)
    mask["name"] = "New Guinea island mask"
    mask["note"] = (
        "Papua and West Papua plus mainland Papua New Guinea. "
        "Excludes the Bismarck Archipelago and Bougainville."
    )
    return mask[["name", "note", "geometry"]]


def padded_bbox(mask, padding):
    minx, miny, maxx, maxy = mask.total_bounds
    return (minx - padding, miny - padding, maxx + padding, maxy + padding)


def clip_to_mask(frame, mask, columns):
    present = [name for name in columns if name in frame.columns]
    clipped = gpd.clip(frame[present + ["geometry"]], mask)
    return valid_geometry(clipped)


def classify_ecoregion(row):
    name = str(row.get("eco_name") or "").lower()
    biome = str(row.get("biome_name") or "").lower()
    if "mangrove" in name or biome == "mangroves":
        return "mangroves"
    if "peat" in name:
        return "peat_swamp_forest"
    if "swamp" in name:
        return "freshwater_swamp_forest"
    if "alpine" in name or "sub-alpine" in name or "montane grasslands" in biome:
        return "alpine_grasslands"
    if "montane" in name:
        return "montane_forest"
    if any(token in name for token in ("savanna", "dry forest", "grassland")):
        return "other"
    if "moist broadleaf" in biome or "rain forest" in name:
        return "lowland_rainforest"
    return "other"


def filter_places(places, config):
    rank_max = config["places"]["keep_scalerank_max"]
    label_names = {item["name"].lower() for item in config["places"]["labels"]}
    rank = places["scalerank"] if "scalerank" in places.columns else 99
    named = places["name"].str.lower().isin(label_names)
    kept = places.loc[named | (rank <= rank_max)].copy()
    columns = [name for name in ("name", "adm0name", "scalerank", "pop_max") if name in kept.columns]
    return kept[columns + ["geometry"]]


def filter_rivers(rivers, config):
    if "name" not in rivers.columns:
        return rivers
    wanted = [name.lower() for name in config["rivers"]["labels"]]
    river_name = rivers["name"].fillna("").str.lower()
    matched = river_name.apply(lambda value: any(token in value for token in wanted))
    rank = rivers["scalerank"] if "scalerank" in rivers.columns else 99
    kept = rivers.loc[matched | (rank <= 4)].copy()
    columns = [name for name in ("name", "scalerank") if name in kept.columns]
    return kept[columns + ["geometry"]]


def write_layer(frame, path, layer, first):
    frame.to_file(path, layer=layer, driver="GPKG", mode="w" if first else "a")


def main():
    config = load_config()
    output = project_path(config["files"]["context_gpkg"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    print("Reading admin-1 boundaries...")
    admin1 = select_admin1(read_layer(config["files"]["admin1"]), config)
    aoi = build_aoi(admin1)
    padding = config["aoi"]["bbox_padding_degrees"]
    bbox = padded_bbox(aoi, padding)
    print(f"Island mask bounds (degrees): {tuple(round(value, 2) for value in aoi.total_bounds)}")

    print("Clipping basemap...")
    land = clip_to_mask(read_layer(config["files"]["land"], bbox), aoi, ["featurecla", "scalerank"])
    coastline = clip_to_mask(read_layer(config["files"]["coastline"], bbox), aoi, ["featurecla", "scalerank"])
    rivers = filter_rivers(
        clip_to_mask(read_layer(config["files"]["rivers"], bbox), aoi, ["name", "scalerank", "featurecla"]),
        config,
    )
    lakes = clip_to_mask(read_layer(config["files"]["lakes"], bbox), aoi, ["name", "scalerank", "featurecla"])
    places = filter_places(
        clip_to_mask(
            read_layer(config["files"]["places"], bbox),
            aoi,
            ["name", "adm0name", "scalerank", "pop_max", "featurecla"],
        ),
        config,
    )

    print("Clipping ecoregions...")
    ecoregions = clip_to_mask(
        read_layer(config["files"]["ecoregions"], bbox),
        aoi,
        ["eco_name", "biome_name", "biome_num", "eco_id", "realm"],
    )
    ecoregions["map_class"] = ecoregions.apply(classify_ecoregion, axis=1)

    layers = [
        ("aoi", aoi),
        ("admin1", admin1),
        ("land", land),
        ("coastline", coastline),
        ("rivers", rivers),
        ("lakes", lakes),
        ("places", places),
        ("ecoregions", ecoregions),
    ]
    first = True
    for name, frame in layers:
        print(f"  {name}: {len(frame)} features")
        write_layer(frame, output, name, first)
        first = False

    print("Ecoregion classes:")
    print(ecoregions["map_class"].value_counts().to_string())
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
