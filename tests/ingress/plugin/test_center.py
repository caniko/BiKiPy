import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

from bikipy.ingress.plugin.center import PluginCenter
from bikipy.ingress.plugin.core.plugin_scope import PluginScope


class TestPluginCenter(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(".").resolve() / "static" / "center.csv"
        # Mocking other attributes, which may be necessary for the initialization
        self.ingress_key = "center"
        self.code_key = "center"
        self.default_trial_argument_key = "manual_center_pixels"
        self.human_readable_index = "Center"

        self.plugin = PluginCenter(
            plugin_scope=PluginScope.TRIALWISE,
            data_path=self.data_path,
            ingress_key=self.ingress_key,
            code_key=self.code_key,
            default_trial_argument_key=self.default_trial_argument_key,
            human_readable_index=self.human_readable_index
        )

    def test_initialization(self):
        # Here we test the initialization of the PluginCenter class
        self.assertIsInstance(self.plugin, PluginCenter)

    def test_read(self):
        self.assertTrue(np.all(self.plugin.trialwise_and_metadata("SomeID")))
