"""Tools the agent can call. All data in data/regions.csv is SYNTHETIC sample data."""
import csv
from pathlib import Path

DATA = Path(__file__).parent / "data" / "regions.csv"


def compute_ndvi(nir: float, red: float) -> float:
    """NDVI = (NIR - Red) / (NIR + Red), rounded to 4 decimals."""
    if nir + red == 0:
        raise ValueError("nir + red must not be zero")
    return round((nir - red) / (nir + red), 4)


def classify_ndvi(ndvi: float) -> str:
    """Rough land cover class from an NDVI value."""
    if not -1.0 <= ndvi <= 1.0:
        raise ValueError("ndvi must be between -1 and 1")
    if ndvi < 0:
        return "water or non-vegetated surface"
    if ndvi < 0.2:
        return "bare soil or built-up area"
    if ndvi < 0.5:
        return "sparse or moderate vegetation"
    return "dense vegetation"


def get_region_observations(region_id: str) -> list[dict]:
    """All observations (date, red, nir reflectance) for a region."""
    with open(DATA, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["region_id"] == region_id]
    if not rows:
        raise ValueError(f"unknown region_id: {region_id}")
    return [
        {"region_id": r["region_id"], "description": r["description"],
         "date": r["date"], "red": float(r["red"]), "nir": float(r["nir"])}
        for r in rows
    ]


TOOL_FUNCTIONS = {
    "compute_ndvi": compute_ndvi,
    "classify_ndvi": classify_ndvi,
    "get_region_observations": get_region_observations,
}

TOOL_SCHEMAS = [
    {
        "name": "get_region_observations",
        "description": "Get all stored observations (date, red and near-infrared reflectance) for a region id such as forest_a, crop_b, water_c, urban_d.",
        "input_schema": {"type": "object",
                         "properties": {"region_id": {"type": "string"}},
                         "required": ["region_id"]},
    },
    {
        "name": "compute_ndvi",
        "description": "Compute NDVI from near-infrared and red reflectance values.",
        "input_schema": {"type": "object",
                         "properties": {"nir": {"type": "number"}, "red": {"type": "number"}},
                         "required": ["nir", "red"]},
    },
    {
        "name": "classify_ndvi",
        "description": "Classify an NDVI value into a rough land cover class.",
        "input_schema": {"type": "object",
                         "properties": {"ndvi": {"type": "number"}},
                         "required": ["ndvi"]},
    },
]
