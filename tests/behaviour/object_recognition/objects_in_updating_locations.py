from bikipy.behaviour.object_recognition.objects_in_updating_locations import (
    ObjectsInUpdatingLocationsExperiment,
)
from tests.test_data import EXPERIMENT_KWARGS


def test_object_in_updating_locations_experiment():
    experiment = ObjectsInUpdatingLocationsExperiment(**EXPERIMENT_KWARGS)
    assert experiment
