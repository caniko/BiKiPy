from typing import Union, Any, AnyStr, SupportsFloat, Sequence

import numpy as np

from bikipy.preferance.generate_border.rem_object import (
    define_object,
    generate_object_borders,
)
from bikipy.utils.point_in_polygon import points_in_rectangle
from bikipy.utils.math import unit_vector, dot_prod_along_axis_1
from bikipy.utils.statistics import invalidate_array
from bikipy.utils.video import get_video_data


class NortPolygon:
    def __init__(
        self,
        sides: Union[Sequence, None] = None,
        guiding_image: Any = None,
        border_distance: Union[SupportsFloat, None] = None,
        feature_scale: Union[Sequence, None] = None,
    ):
        self.border_distance = border_distance
        self.feature_scale = (
            np.asanyarray(feature_scale) if feature_scale is not None else None
        )

        if sides is None:
            if guiding_image is None:
                msg = "'sides' not defined, define 'guiding_image' to define sides with user"
                raise ValueError(msg)

            self.sides = define_object(guiding_image, n=0)
            # TODO: Feature scale
        else:
            self.sides = sides

    @classmethod
    def from_video(
        cls,
        video_path,
        labels: Union[Sequence, None] = None,
        frame_time: AnyStr = "middle",
        *args,
        **kwargs,
    ):
        frame, x_res, y_res = get_video_data(video_path, frame_time)

        if not labels:
            return cls(
                *args, guiding_image=frame, feature_scale=(x_res, y_res), **kwargs
            )
        return {
            label: cls(
                *args, guiding_image=frame, feature_scale=(x_res, y_res), **kwargs
            )
            for label in labels
        }

    @property
    def sides(self):
        return self.__sides

    @sides.setter
    def sides(self, sides: Sequence):
        sides = np.asanyarray(sides)

        self.__sides = sides
        self.number_of_sides = len(sides)
        self.edges = np.array(
            [
                sides[i + 1 if i + 1 != self.number_of_sides else 0] - sides[i]
                for i in range(self.number_of_sides)
            ]
        )
        self.borders = (
            generate_object_borders(sides, self.border_distance)
            if self.border_distance
            else None
        )
        self.side_pair_to_edge = {
            **{
                f"{i}_{i + 1 if i + 1 != self.number_of_sides else 0}": self.edges[i]
                for i in range(self.number_of_sides)
            },
            **{
                f"{i + 1 if i + 1 != self.number_of_sides else 0}_{i}": self.edges[i]
                for i in range(self.number_of_sides)
            },
        }

    @property
    def feat_scaled_sides(self):
        if not self.feature_scale:
            msg = "Feature scale parameters have not been defined in this instance"
            raise AttributeError(msg)
        return self.sides / self.feature_scale

    @property
    def order(self):
        return self.sides.shape[0]

    def attention(
        self,
        eye_center: Sequence,
        nose: Sequence,
        torso: Sequence,
        fps: SupportsFloat,
        max_distance_from_nose_to_border: SupportsFloat,
        max_radians_between_gaze_and_object: SupportsFloat = 1 / 4 * np.pi,
    ):
        """

        Parameters
        ----------
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
        eye_center, nose, torso = np.asanyarray(eye_center), np.asanyarray(nose), np.asanyarray(torso)
        max_distance_from_nose_to_border = float(max_distance_from_nose_to_border)
        max_radians_between_gaze_and_object = float(max_radians_between_gaze_and_object)
        fps = float(fps)

        # Remove points that are far away from the border
        nose_to_sides_distance = np.abs(
            np.linalg.norm(self.sides - np.expand_dims(nose, 1))
        )
        proto_valid_boolean_indexes = np.any(
            nose_to_sides_distance <= max_distance_from_nose_to_border, axis=1
        )
        eye_center = invalidate_array(eye_center, proto_valid_boolean_indexes)

        # Remove points that are close to the border, but aren't inside
        if self.order == 4:
            nose_within_border = points_in_rectangle(
                self.borders[0], self.borders[-1], self.borders[1], nose
            )
            torso_outside_polygon = np.logical_not(points_in_rectangle(
                self.sides[0], self.sides[-1], self.sides[1], nose
            ))
        else:
            msg = f"Polygon with {self.order} sides is not supported"
            raise NotImplemented(msg)

        proto_valid_boolean_indexes = np.logical_and(
            proto_valid_boolean_indexes,
            np.logical_and(nose_within_border, torso_outside_polygon)
        )

        # Remove points that have the respective gaze vector higher than the defined max
        two_closest_sides_to_nose = np.apply_along_axis(
            lambda row: np.where(row <= 1)[0],  # Two closest sides are on index 0 and 1
            1,
            np.argsort(nose_to_sides_distance)
        )
        closest_edges_to_nose = np.apply_along_axis(
            lambda x: self.side_pair_to_edge[f"{x[0]}_{x[1]}"],
            1,
            two_closest_sides_to_nose,
        )

        eye_to_nose_unit = np.apply_along_axis(
            lambda x: unit_vector(x), 1, nose - eye_center
        )
        closest_edges_eye_nose_dot_prod = dot_prod_along_axis_1(closest_edges_to_nose * eye_to_nose_unit)

        compute_step_radians_between_vectors = (
            closest_edges_eye_nose_dot_prod
            / np.apply_along_axis(np.linalg.norm, 1, closest_edges_to_nose)
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

        i, frames_with_attention, true_counter = 0, 0, 0
        length = exploration_boolean_indexes.shape[0]
        while i < length:
            if exploration_boolean_indexes[i]:
                true_counter += 1
                if true_counter == fps:
                    frames_with_attention += true_counter
                elif true_counter > fps:
                    frames_with_attention += 1
            else:
                true_counter = 0

        return frames_with_attention / fps


if __name__ == "__main__":
    from pathlib import Path

    WORKING_DIR = Path("C:/Users/Can/Projects/Neuroscience/Imen/data")
    NORT_DIR = WORKING_DIR / "nort"
    BEFORE_DIR = NORT_DIR / "NORT_02.06.2020"
    AFTER_DIR = NORT_DIR / "NORT_24 08 2020 (after)"
    BORDER_DISTANCE = 3 * 224 / 40

    b_1 = NortPolygon.from_video(
        str(BEFORE_DIR / "Test 47.mp4"), n_borders=2, border_distance=BORDER_DISTANCE
    )
