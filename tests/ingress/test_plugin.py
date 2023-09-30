import unittest
from pathlib import Path

import numpy as np

from bikipy.ingress.plugin.bonsai_timestamp import PluginBonsaiTimestamp
from bikipy.ingress.plugin.center import PluginCenter
from bikipy.ingress.plugin.core.plugin_scope import PluginScope
from bikipy.ingress.plugin.meters_per_pixel import PluginMeterPerPixel
from bikipy.ingress.plugin.video import PluginVideo


class PluginTestMixin:
    def test_initialization(self):
        # Here we test the initialization of the PluginBonsaiTimestamp class
        self.assertIsInstance(self.plugin, PluginBonsaiTimestamp)


class TestPluginBonsaiTimestamp(unittest.TestCase, PluginTestMixin):
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

    def test_read(self):
        self.assertTrue(np.all(self.plugin.trialwise_and_metadata("SomeID")[1:]))


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


class TestPluginMeterPerPixel(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(".").resolve() / "static" / "meters_per_pixel-diagonal-0.707-721_0.csv"
        # Mocking other attributes, which may be necessary for the initialization
        self.ingress_key = "meters_per_pixel"
        self.code_key = "meters_per_pixel"
        self.default_trial_argument_key = "meters_per_pixel"
        self.human_readable_index = "MetersPerPixel"

        self.plugin = PluginMeterPerPixel(
            plugin_scope=PluginScope.GLOBAL,
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
        self.assertTrue(np.all(self.plugin.globally_defined))


class TestPluginVideo(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(".").resolve() / "static" / "0.video-721_0.mextractor"
        self.ingress_key = "video"
        self.code_key = "video"
        self.default_trial_argument_key = "manual_video"
        self.human_readable_index = "Video"

        self.plugin = PluginVideo(
            plugin_scope=PluginScope.TRIALWISE,
            data_path=self.data_path,
            ingress_key=self.ingress_key,
            code_key=self.code_key,
            default_trial_argument_key=self.default_trial_argument_key,
            human_readable_index=self.human_readable_index
        )

    def test_initialization(self):
        # Here we test the initialization of the PluginVideo class
        self.assertIsInstance(self.plugin, PluginVideo)
