from ordered_set import OrderedSet

INGRESS_MAP_NAME = "ingress"
READER_MAP_NAME = "reader"
EXPERIMENT_MAP_NAME = "experiment"
TRIAL_MAP_NAME = "trial"
ENCLOSURE_MAP_NAME = "enclosure"
PERIMETER_MAP_NAME = "perimeter"
PLUGIN_MAP_NAME = "plugin"
RUNTIME_SETTINGS_MAP_NAME = "runtime_settings"

QUALIA_HEURISTICS_MAP_NAME = "qualia_heuristics"
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

INSPECT_FIG_FILE_FORMAT = ".svgz"
ANALYSIS_CACHE_STEM_ID = "analysis_cache"
