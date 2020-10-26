from typing import Union, SupportsFloat, Sequence

import numpy as np

from bikipy.border.base import GenericPolygonalBorder
from bikipy.math.vector import unit_vector


class NortObject(GenericPolygonalBorder):
    corners = 0

    def __init__(
        self,
        sides: Sequence[Sequence[SupportsFloat]],
        *args,
        border_distance: Union[SupportsFloat, None] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.sides = sides
        self.border_distance = border_distance

    @property
    def sides(self):
        return self.__sides

    @sides.setter
    def sides(self, sides: Sequence[Sequence[SupportsFloat]]):
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
