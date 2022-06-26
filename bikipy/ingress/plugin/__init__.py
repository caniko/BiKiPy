from typing import Callable

from bikipy.ingress.plugin.center import center_file_path_to_value
from bikipy.ingress.plugin.meters_per_pixel import meters_per_pixel_file_name_to_value
from bikipy.ingress.plugin.perimeter import perimeter_file_path_to_value

PLUGIN_NAME_TO_KEYRING: dict[str, dict[str, str | Callable]] = {
    "meters_per_pixel": {
        "ingress_key": "meters_per_pixel_definition_strategy",
        "code_key": "meters_per_pixel",
        "bikipy_trial_key": "meters_per_pixel",
        "human_readable_index": "MetersPerPixel",
        "file_path_to_value": meters_per_pixel_file_name_to_value,
    },
    "center": {
        "ingress_key": "center_definition_strategy",
        "code_key": "center",
        "bikipy_trial_key": "manual_center_pixels",
        "human_readable_index": "Center",
        "file_path_to_value": center_file_path_to_value,
    },
    "perimeter": {
        "ingress_key": "perimeter_definition_strategy",
        "code_key": "perimeter",
        "bikipy_trial_key": "label_to_perimeter",
        "human_readable_index": "Perimeter",
        "file_path_to_value": perimeter_file_path_to_value,
    },
}

ingress_key_to_plugin_name = {plugin.pop("ingress_key"): plugin for plugin in PLUGIN_NAME_TO_KEYRING.values()}
