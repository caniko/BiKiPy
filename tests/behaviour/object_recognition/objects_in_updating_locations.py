from bikipy.behaviour.object_recognition.objects_in_updating_locations import ObjectsInUpdatingLocationsExperiment


def test_object_in_updating_locations_experiment():
    experiment = ObjectsInUpdatingLocationsExperiment(
        gaze_start_point_label="center_eye",
        gaze_travel_direction_point_label="nose",
        perimeter_border_normal_metric_magnitude=0.05,
    )
