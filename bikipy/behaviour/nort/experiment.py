from typing import Union, SupportsFloat, SupportsInt, Sequence

import numpy as np

from bikipy.math.vector import unit_vector, dot_prod_along_axis_1, distance_between_line_and_point
from bikipy.math.point_in_polygon import points_in_parallelogram

from bikipy.behaviour.nort import NortObject


def location_filter(
    nort_object: NortObject,
    nose: Sequence[Sequence[SupportsFloat]],
    torso: Sequence[Sequence[SupportsFloat]]
) -> np.ndarray[bool]:
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
    nort_object: NortObject,
    nose: Sequence[Sequence[SupportsFloat]],
    eye_center: Sequence[Sequence[SupportsFloat]]
):
    
    distance_between_nose_and_nort_object = None

    eye_to_nose_unit = np.apply_along_axis(
        lambda x: unit_vector(x), 1, nose - eye_center
    )
    closest_edges_eye_nose_dot_prod = dot_prod_along_axis_1(
        closest_edges_to_nose * eye_to_nose_unit
    )

    compute_step_radians_between_vectors = (
        closest_edges_eye_nose_dot_prod
        / np.linalg.norm(closest_edges_to_nose, axis=1)
        # * np.apply_along_axis(np.linalg.norm, 1, eye_to_nose_unit)  norm is 1
    )
    radians_between_gaze_and_object = np.abs(
        np.arccos(compute_step_radians_between_vectors)
    )
    valid_radians = (
        radians_between_gaze_and_object <= max_radians_between_gaze_and_object
    )


def attention_span_filter(
    observation_boolean_indexes: Sequence[bool], fps: SupportsInt
) -> np.ndarray[bool]:

    fps = round(fps)
    tolerance = int(round(fps / 4))

    first_valid_index = None
    length = observation_boolean_indexes.shape[0]
    observation_boolean_indexes = np.full(length, False)
    i, valid_frames_within_border, true_counter, consecutive_false = 0, 0, 0, 0
    while i < length:
        if observation_boolean_indexes[i]:
            true_counter += 1
            if true_counter == fps:  # One second
                first_valid_index = i
                valid_frames_within_border += true_counter
            elif true_counter > fps:
                valid_frames_within_border += 1
        else:
            if first_valid_index:
                if consecutive_false < tolerance:
                    consecutive_false += 1
                else:
                    observation_boolean_indexes[first_valid_index:i] = True
                    consecutive_false, true_counter = 0, 0
                    first_valid_index = None
        i += 1

    if valid_frames_within_border == 0:
        print("Subject didn't observe the nort object")
        return False

    assert (
        np.any(observation_boolean_indexes)
        and np.sum(observation_boolean_indexes) >= fps
    )

    return observation_boolean_indexes


def nort(
    nort_object: NortObject,
    eye_center: Sequence[Sequence[SupportsFloat]],
    nose: Sequence[Sequence[SupportsFloat]],
    torso: Sequence[Sequence[SupportsFloat]],
    fps: Union[SupportsInt, SupportsFloat],
    max_radians_between_gaze_and_object: SupportsFloat = 1 / 4 * np.pi,
):
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
    max_distance_from_nose_to_border
        Maximum distance between nose and border
    max_radians_between_gaze_and_object
        Maximum radians between the gaze vector (eye_centre to nose) and object tangent

    Returns
    -------

    """
    eye_center, nose, torso = (
        np.asanyarray(eye_center),
        np.asanyarray(nose),
        np.asanyarray(torso),
    )
    max_radians_between_gaze_and_object = float(max_radians_between_gaze_and_object)

    fps = float(fps)
