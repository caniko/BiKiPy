from bikipy.ingress.plugin.perimeter.base import AbcPerimeterPlugin


class PluginEnclosure(AbcPerimeterPlugin):
    ingress_key = "enclosure"
    code_key = "enclosure"
    default_trial_argument_key = "label_to_perimeter"
    human_readable_index = "Perimeter"
