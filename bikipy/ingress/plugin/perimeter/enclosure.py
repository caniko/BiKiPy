from bikipy.core.typing import Label
from bikipy.ingress.plugin.perimeter.base import AbcPerimeterPlugin


class PluginEnclosure(AbcPerimeterPlugin):
    ingress_key = "enclosure"
    code_key = "enclosure"
    default_trial_argument_key = "label_to_perimeter"
    human_readable_index = "Perimeter"

    @property
    def globally_defined(self):
        return self.perimeter_mapper(self)

    def trialwise_and_metadata(self, trial_id: Label, naive: bool = False):
        return self.perimeter_mapper(self)
