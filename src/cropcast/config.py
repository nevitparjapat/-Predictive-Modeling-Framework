"""Shared settings: column names, valid ranges, plain-language labels and pass marks.

Keeping everything in one place means the app, the tests and the report all agree.
"""

TARGET = "yield_t_ha"

# Raw inputs the model needs (besides year, farm and the target)
BASE_FEATURES = [
    "rainfall_mm", "avg_temp_c", "heat_days", "soil_ph", "soil_oc_pct",
    "nitrogen_kg_ha", "irrigation_mm", "ndvi_peak", "prev_yield",
]

# Columns that may have gaps; a "was missing" flag is added for each
FLAG_COLUMNS = ["ndvi_peak", "soil_ph", "rainfall_mm", "nitrogen_kg_ha"]

REQUIRED_COLUMNS = ["year", "farm"] + BASE_FEATURES + [TARGET]

# Validation rules: (lowest allowed, highest allowed)
RULES = {
    "rainfall_mm": (0, 2500),
    "avg_temp_c": (-5, 50),
    "heat_days": (0, 60),
    "soil_ph": (3.5, 9.5),
    "soil_oc_pct": (0, 10),
    "nitrogen_kg_ha": (0, 400),
    "irrigation_mm": (0, 1000),
    "ndvi_peak": (0, 1),
    "yield_t_ha": (0, 15),
}

# Friendly labels, units and help text used by the app
LABELS = {
    "rainfall_mm": ("Rainfall", "mm", "Total rain during the growing season."),
    "avg_temp_c": ("Average temperature", "°C", "Average temperature over the growing season."),
    "heat_days": ("Days above 35 °C", "", "Number of days hotter than 35 °C in the season."),
    "soil_ph": ("Soil pH", "", "How acidic or alkaline the soil is. Around 6.7 is ideal for most crops."),
    "soil_oc_pct": ("Soil organic carbon", "%", "A measure of how rich and healthy the soil is."),
    "nitrogen_kg_ha": ("Nitrogen fertiliser", "kg/ha", "Nitrogen applied per hectare."),
    "irrigation_mm": ("Irrigation", "mm", "Water added by irrigation (not rain)."),
    "ndvi_peak": ("Crop greenness (NDVI)", "0 to 1", "Peak satellite vegetation index. Higher means healthier, denser crop."),
    "prev_yield": ("Last season's yield", "t/ha", "What this same farm produced last season."),
}

# Pass marks chosen before looking at the test results (same as the report)
PASS_MARKS = {
    "rmse_gain_pct": 20.0,   # RMSE must be at least this % lower than the best simple guess
    "mape_max": 12.0,        # percent
    "r2_min": 0.60,
    "cv_spread_max_pct": 25.0,
    "drought_ratio_max": 1.25,
    "coverage_low": 85.0,
    "coverage_high": 95.0,
    "bias_max_pct": 5.0,
}

TRAIN_END_YEAR = 2020     # training seasons: up to and including this year
VALIDATION_YEAR = 2021    # tuning season
# Everything after VALIDATION_YEAR is the locked test set.


def pretty_name(col: str) -> str:
    """Friendly name for any model input, including the ones we build ourselves."""
    if col in LABELS:
        return LABELS[col][0]
    special = {
        "yield_t_ha": "Yield (what we predict)",
        "water_total_mm": "Total water (rain + irrigation)",
        "heat_x_temp": "Heat × temperature",
        "ph_dev_sq": "Soil pH away from ideal",
    }
    if col in special:
        return special[col]
    if col.endswith("_was_missing"):
        return LABELS.get(col[: -len("_was_missing")], (col,))[0] + " was missing"
    return col
