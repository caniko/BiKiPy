from typing import SupportsFloat, Sequence

import numpy as np

from bikipy.math.vector import unit_vector, dot_prod_along_axis_1
from bikipy.math.point_in_polygon import points_in_parallelogram
from bikipy.math.statistics import invalidate_array

from bikipy.behaviour.nort import NortObject


def attention(
    nort_object: NortObject,
    eye_center: Sequence[Sequence[SupportsFloat]],
    nose: Sequence[Sequence[SupportsFloat]],
    torso: Sequence[Sequence[SupportsFloat]],
    fps: SupportsFloat,
    max_distance_from_nose_to_border: SupportsFloat,
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
    max_distance_from_nose_to_border = float(max_distance_from_nose_to_border)
    max_radians_between_gaze_and_object = float(max_radians_between_gaze_and_object)
    fps = float(fps)

    # Remove points that are close to the border, but aren't inside
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
    proto_valid_boolean_indexes = np.logical_and(
        nose_within_border, torso_outside_polygon
    )

    # Attention span filter
    # IDEA: Vectorize(?)
    i, valid_frames_within_border, true_counter, consecutive_false = 0, 0, 0, 0
    first_valid_index = None
    tolerance = int(round(fps / 4))
    length = proto_valid_boolean_indexes.shape[0]
    observation_boolean_indexes = np.zeros(length) != 0
    while i < length:
        if proto_valid_boolean_indexes[i]:
            true_counter += 1
            if true_counter == fps:  # One second
                first_valid_index = i
                valid_frames_within_border += true_counter
            elif true_counter > fps:
                valid_frames_within_border += 1
        else:
            if first_valid_index:
                if consecutive_false < tolerance:
                    tolerance += 1
                    i += 1
                    continue
                else:
                    observation_boolean_indexes[first_valid_index:i] = True
                    tolerance = 0

                first_valid_index = None
            true_counter = 0

        i += 1

    if valid_frames_within_border == 0:
        print("Subject didn't observe the nort object")
        return

    proto_valid_boolean_indexes = observation_boolean_indexes
    assert (
        np.any(proto_valid_boolean_indexes)
        and np.sum(proto_valid_boolean_indexes) >= fps
    )

    # Remove points that are far away from the border
    distance_from_nose_to_edges = np.abs(
        np.apply_along_axis(
            lambda nose_coord: np.apply_along_axis(
                lambda side: np.linalg.norm(side - nose_coord), 1, nort_object.sides
            ),
            1,
            nose,
        )
    )
    proto_valid_boolean_indexes = np.any(
        nose_to_sides_distance <= max_distance_from_nose_to_border, axis=1
    )
    eye_center = invalidate_array(eye_center, proto_valid_boolean_indexes)

    # Remove points that have the respective gaze vector higher than the defined max
    two_closest_sides_to_nose = np.apply_along_axis(
        lambda row: np.where(row <= 1)[0],  # Two closest sides are on index 0 and 1
        1,
        np.argsort(nose_to_sides_distance),
    )
    closest_edges_to_nose = np.apply_along_axis(
        lambda x: nort_object.side_pair_to_edge[f"{x[0]}_{x[1]}"],
        1,
        two_closest_sides_to_nose,
    )

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

    exploration_boolean_indexes = np.logical_and(
        valid_radians, proto_valid_boolean_indexes
    )
