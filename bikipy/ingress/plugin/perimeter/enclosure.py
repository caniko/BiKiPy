from bikipy.core.typing import Label
from bikipy.ingress.plugin.perimeter.base import AbcPerimeterPlugin


class PluginEnclosure(AbcPerimeterPlugin):
    ingress_key = "enclosure"
    code_key = "enclosure"
    default_trial_argument_key = "label_to_perimeter"
    human_readable_index = "Perimeter"

    @property
    def globally_defined(self):
        self._assert_correct_scope_global()
        return self.perimeter_mapper()

    def trialwise_and_metadata(self, trial_id: Label, naive: bool = False):
        self._assert_correct_scope_trialwise_metadata()
        return self.perimeter_mapper(trial_id)
