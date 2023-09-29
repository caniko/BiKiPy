from pathlib import Path
from unittest import TestCase

from bikipy.behaviour.core.base import BaseExperiment


class TestExperiment(TestCase):
    def test_create_bare(self):
        self.assertTrue(BaseExperiment(trial_init_error_out_dir=Path(".").resolve().parent))
