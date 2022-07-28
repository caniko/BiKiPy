"""
Plugins are Pydantic models that read and process data for analysis. These have varying degrees of complexity.
Some are quite simple, and are read and processed similarly no matter the occasion while some have different
definition strategies dependent on their scope.

Scope in this case refers to the stage at which the plugin was defined. We distinguish between "trial-wise", "metadata",
and "globally_defined". While globally_defined scopes are always properties, trial-wise and metadata are methods.

The aforementioned methods are called in Experiment.trial_id_to_keyword_arguments with the respective trial_id
as the first and only argument. This gives the method the context necessary to define the correct value for the trial.
"""

from bikipy.ingress.plugin.base import Plugin
from bikipy.ingress.plugin.bonsai_timestamp import PluginBonsaiTimestamp
from bikipy.ingress.plugin.center import PluginCenter
from bikipy.ingress.plugin.meters_per_pixel import PluginMeterPerPixel
from bikipy.ingress.plugin.perimeter.change_reference import PluginChangeReference
from bikipy.ingress.plugin.perimeter.single import PluginSinglePerimeter
from bikipy.ingress.plugin.perimeter.radial import PluginRadial
from bikipy.ingress.plugin.video import PluginVideo


ALL_PLUGINS = (
    PluginMeterPerPixel,
    PluginSinglePerimeter,
    PluginRadial,
    PluginChangeReference,
    PluginVideo,
    PluginBonsaiTimestamp,
    PluginCenter,
)

plugin_code_key_to_model = {plugin.code_key: plugin for plugin in ALL_PLUGINS}
ingress_key_to_model: dict[str, Plugin] = {model.ingress_key: model for model in plugin_code_key_to_model.values()}
