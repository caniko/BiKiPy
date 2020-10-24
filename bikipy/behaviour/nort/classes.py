from typing import Union, Any, AnyStr, SupportsFloat, Sequence

import numpy as np

from bikipy.math.vector import unit_vector
from bikipy.border.draw.polygon import (
    draw_polygon_corners,
    define_polygon_border,
)
from bikipy.utils.video import get_video_data


class NortObject:
    def __init__(
        self,
        sides: Union[Sequence, None] = None,
        guiding_image: Any = None,
        border_distance: Union[SupportsFloat, None] = None,
        feature_scale: Union[Sequence, None] = None,
        label: Union[AnyStr, None] = None,
    ):
        self.label = label
        self.border_distance = border_distance
        self.feature_scale = (
            np.asanyarray(feature_scale) if feature_scale is not None else None
        )

        if sides is None:
            if guiding_image is None:
                msg = "'sides' not defined, define 'guiding_image' to define sides with user"
                raise ValueError(msg)

            self.sides = draw_polygon_corners(guiding_image, n=0)
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

    @property
    def borders(self):
        if not self.border_distance:
            msg = "border_distance has to be defined as an object attribute"
            raise AttributeError(msg)

        diagonal_unit_2_0 = unit_vector(self.sides[0] - self.sides[2])
        diagonal_unit_3_1 = unit_vector(self.sides[1] - self.sides[3])

        return np.array(
            (
                self.sides[0] + diagonal_unit_2_0 * self.border_distance,
                self.sides[1] + diagonal_unit_3_1 * self.border_distance,
                self.sides[2] - diagonal_unit_2_0 * self.border_distance,
                self.sides[3] - diagonal_unit_3_1 * self.border_distance,
            )
        )
