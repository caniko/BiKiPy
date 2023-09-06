from typing import TYPE_CHECKING

from bikipy.ingress.plugin.bonsai_timestamp import PluginBonsaiTimestamp
from bikipy.ingress.plugin.center import PluginCenter
from bikipy.ingress.plugin.frame import PluginFrame
from bikipy.ingress.plugin.meters_per_pixel import PluginMeterPerPixel
from bikipy.ingress.plugin.perimeter.change_reference import PluginChangeReference
from bikipy.ingress.plugin.perimeter.enclosure import PluginEnclosure
from bikipy.ingress.plugin.perimeter.radial_maze import PluginRadial
from bikipy.ingress.plugin.perimeter.single import PluginSinglePerimeter
from bikipy.ingress.plugin.video import PluginVideo

if TYPE_CHECKING:
    from bikipy.ingress.plugin.core.base import BasePlugin

ALL_PLUGINS = (
    PluginMeterPerPixel,
    PluginSinglePerimeter,
    PluginEnclosure,
    PluginRadial,
    PluginChangeReference,
    PluginFrame,
    PluginVideo,
    PluginBonsaiTimestamp,
    PluginCenter,
)

plugin_code_key_to_model = {plugin.code_key: plugin for plugin in ALL_PLUGINS}
ingress_key_to_model: dict[str, "BasePlugin"] = {model.ingress_key: model for model in plugin_code_key_to_model.values()}
