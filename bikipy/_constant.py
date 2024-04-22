from ordered_set import OrderedSet

from bikipy import runtime_settings

INGRESS_MAP_NAME = "ingress"
READER_MAP_NAME = "reader"
EXPERIMENT_MAP_NAME = "experiment"
HABITUATION_TRIAL_MAP_NAME = "habituation"
TRIAL_MAP_NAME = "trial"
ENCLOSURE_MAP_NAME = "enclosure"
PERIMETER_MAP_NAME = "perimeter"
PLUGIN_MAP_NAME = "plugin"
RUNTIME_SETTINGS_MAP_NAME = "runtime_settings"

QUALIA_HEURISTICS_MAP_NAME = "heuristics"
PHYSICAL_OBJECT_MAP_NAME = QUALIA_HEURISTICS_MAP_NAME

PROJECTKIT_CONFIG_KEY_ORDER = OrderedSet(
    (
        "manual",
        INGRESS_MAP_NAME,
        EXPERIMENT_MAP_NAME,
        TRIAL_MAP_NAME,
        ENCLOSURE_MAP_NAME,
        PERIMETER_MAP_NAME,
        PLUGIN_MAP_NAME,
        RUNTIME_SETTINGS_MAP_NAME,
    )
)

AUGMENTED_COORDINATE_CACHED_FILE_LABEL = "augmented"

BIKIPY_ANALYSIS_VIDEO_PREFIX = "bikipy_analysis"

INSPECT_FIG_FILE_FORMAT = ".svgz"
INSPECT_SIMPLE_FIG_FILE_FORMAT = ".jpg"
MINIMUM_FIG_DPI = 300
ANALYSIS_CACHE_STEM_ID = "analysis_cache"

QUIVER_KWARGS = {
    "alpha": runtime_settings.matplotlib_scatter_alpha,
    # "scale_units": "xy",
    "angles": "xy",
    "units": "xy",
}
