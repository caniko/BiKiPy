from typing import Union, SupportsFloat, SupportsInt, Sequence, Tuple, List

import numpy as np

from bikipy.feature.angle import counter_clockwise_angel_2d
from bikipy.math.vector import unit_vector, closest_line_to_point
from bikipy.math.point_in_polygon import points_in_parallelogram
from bikipy.border.base import PolygonalBorder


def location_filter(
    nort_object,
    nose: Sequence[Sequence[SupportsFloat]],
    torso: Sequence[Sequence[SupportsFloat]],
) -> Sequence[bool]:
    # Remove nose points that aren't inside the border
    if nort_object.order == 4:
        nose_within_border = points_in_parallelogram(
            nort_object.borders[0],
            nort_object.borders[-1],
            nort_object.borders[1],
            nose,
        )
        torso_outside_polygon = np.logical_not(
            points_in_parallelogram(
                nort_object.sides[0], nort_object.sides[-1], nort_object.sides[1], torso
            )
        )
    else:
        msg = f"Polygon with {nort_object.order} sides is not supported"
        raise NotImplemented(msg)

    # Find states where the nose is within border while the torso is not over object
    return np.logical_and(nose_within_border, torso_outside_polygon)


def gaze_direction_filter(
    nort_object,
    nose: Sequence[Sequence[SupportsFloat]],
    eye_center: Sequence[Sequence[SupportsFloat]],
    max_radians: SupportsFloat,
):
    nose, eye_center = np.asanyarray(nose), np.asanyarray(eye_center)

    eye_to_nose_unit = unit_vector(nose - eye_center)

    closest_side, idx = closest_line_to_point(
        nort_object.side_vectors, nort_object.sides, eye_center
    )

    radians = np.abs(counter_clockwise_angel_2d(closest_side, eye_to_nose_unit) - np.pi)

    return radians <= max_radians


def attention_span_filter(
    valid_indexes: Sequence[bool], fps: SupportsFloat
) -> np.ndarray:
    valid_indexes = np.asanyarray(valid_indexes)
    tolerance = int(round(fps / 4))
    fps = int(round(fps))

    length = valid_indexes.shape[0]
    observation_boolean_indexes = np.full(length, False)

    first_valid_index = None
    i, valid_frames_within_border, true_counter, consecutive_false = 0, 0, 0, 0
    while True:
        if valid_indexes[i]:
            true_counter += 1
            consecutive_false = 0
            if true_counter == fps:  # One second
                first_valid_index = i - fps
        else:
            if first_valid_index:
                if consecutive_false <= tolerance:
                    consecutive_false += 1
                else:
                    observation_boolean_indexes[first_valid_index:i] = True
                    valid_frames_within_border += true_counter

                    consecutive_false, true_counter = 0, 0
                    first_valid_index = None
            else:
                true_counter = 0

        i += 1

        if i == length:
            if true_counter != 0:
                observation_boolean_indexes[first_valid_index:] = True
            break

    if valid_frames_within_border == 0:
        print("Subject didn't observe the nort object")
        return valid_indexes

    assert (
        np.any(observation_boolean_indexes)
        and np.sum(observation_boolean_indexes) >= fps
    ), f"True: {np.sum(observation_boolean_indexes)}; fps: {fps}"

    return observation_boolean_indexes


def nort_observation(
    nort_object: PolygonalBorder,
    eye_center: Sequence[Sequence[SupportsFloat]],
    nose: Sequence[Sequence[SupportsFloat]],
    torso: Sequence[Sequence[SupportsFloat]],
    fps: Union[SupportsInt, SupportsFloat],
    max_radians_gaze_and_object: SupportsFloat = 1 / 4 * np.pi,
) -> np.ndarray:
    """

    Parameters
    ----------
    nort_object
    eye_center
        Points across time defining the position between the eyes of the animal
    nose
        Points across time defining the position of the animal nose
    torso
        Points across time defining the central position of the animal torso
    fps
        Frames per second in the media used for the respective data source
    max_radians_gaze_and_object
        Maximum radians between the gaze vector (eye_centre to nose) and object tangent

    Returns
    -------

    """
    eye_center, nose, torso = (
        np.asanyarray(eye_center),
        np.asanyarray(nose),
        np.asanyarray(torso),
    )
    fps = float(fps)
    max_radians_gaze_and_object = float(max_radians_gaze_and_object)

    proto_observations = np.logical_and(
        location_filter(nort_object, nose, torso),
        gaze_direction_filter(
            nort_object, nose, eye_center, max_radians_gaze_and_object
        ),
    )
    if np.sum(proto_observations) < fps:
        return np.full_like(proto_observations, False)

    return np.array(attention_span_filter(proto_observations, fps))
