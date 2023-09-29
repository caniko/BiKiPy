import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bikipy.ingress.plugin.bonsai_timestamp import PluginBonsaiTimestamp
from bikipy.ingress.plugin.core.plugin_scope import PluginScope


class TestPluginBonsaiTimestamp(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(".").resolve() / "static" / "bonsai_timestamp.csv"
        # Mocking other attributes, which may be necessary for the initialization
        self.ingress_key = "timestamp"
        self.code_key = "timestamp"
        self.default_trial_argument_key = "coordinate_timestamp_index"
        self.human_readable_index = "Timestamp"

        self.plugin = PluginBonsaiTimestamp(
            plugin_scope=PluginScope.TRIALWISE,
            data_path=self.data_path,
            ingress_key=self.ingress_key,
            code_key=self.code_key,
            default_trial_argument_key=self.default_trial_argument_key,
            human_readable_index=self.human_readable_index
        )

    def test_initialization(self):
        # Here we test the initialization of the PluginBonsaiTimestamp class
        self.assertIsInstance(self.plugin, PluginBonsaiTimestamp)

    def test_read(self):
        self.assertTrue(np.all(self.plugin.trialwise_and_metadata("SomeID")[1:]))
