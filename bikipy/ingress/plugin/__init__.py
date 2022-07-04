from bikipy.ingress.plugin.base import Plugin
from bikipy.ingress.plugin.center import PluginCenter
from bikipy.ingress.plugin.meters_per_pixel import PluginMeterPerPixel
from bikipy.ingress.plugin.perimeter.perimeter import PluginPerimeter
from bikipy.ingress.plugin.perimeter.radial import PluginRadial
from bikipy.ingress.plugin.perimeter.reference import PluginReference
from bikipy.ingress.plugin.video import PluginVideo


PLUGIN_NAME_TO_MODEL = {
    "meters_per_pixel": PluginMeterPerPixel,
    "perimeter": PluginPerimeter,
    "radial": PluginRadial,
    "reference": PluginReference,
    "video": PluginVideo,
    "center": PluginCenter,
}

ingress_key_to_model: dict[str, Plugin] = {model.ingress_key: model for model in PLUGIN_NAME_TO_MODEL.values()}
