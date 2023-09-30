import unittest
from pathlib import Path

import numpy as np

from bikipy.ingress.plugin.core.plugin_scope import PluginScope
from bikipy.ingress.plugin.meters_per_pixel import PluginMeterPerPixel


# TODO: Make a test for trialwise also

class TestPluginMeterPerPixel(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(".").resolve() / "static" / "meters_per_pixel-diagonal-0.707-721_0.csv"
        # Mocking other attributes, which may be necessary for the initialization
        self.ingress_key = "meters_per_pixel"
        self.code_key = "meters_per_pixel"
        self.default_trial_argument_key = "meters_per_pixel"
        self.human_readable_index = "MetersPerPixel"

        self.plugin = PluginMeterPerPixel(
            plugin_scope=PluginScope.TRIALWISE,
            data_path=self.data_path,
            ingress_key=self.ingress_key,
            code_key=self.code_key,
            default_trial_argument_key=self.default_trial_argument_key,
            human_readable_index=self.human_readable_index
        )

    def test_initialization(self):
        # Here we test the initialization of the PluginMeterPerPixel class
        self.assertIsInstance(self.plugin, PluginMeterPerPixel)

    def test_read(self):
        self.assertTrue(np.all(self.plugin.trialwise_and_metadata("SomeID")))

