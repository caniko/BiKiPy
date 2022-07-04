"""
Plugins are Pydantic models that read and process data for analysis. These have varying degrees of complexity.
Some are quite simple, and are read and processed similarly no matter the occasion while some have different
definition strategies dependent on their scope.

Scope in this case refers to the stage at which the plugin was defined. We distinguish between "trialwise", "metadata",
and "globally_defined". While globally_defined scopes are always properties, trialwise and metadata are methods.

The beforementioned methods are called in Experiment.trial_id_to_keyword_arguments with the respective trial_id
as the first and only argument. This gives the method the context nescessary to define the correct value for the trial.
"""

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
