from functools import cached_property

import numpy as np
from pydantic import computed_field, field_validator
from pydantic_numpy.typing import NpNDArrayFp64

from bikipy.core.base import BikipyModel
from bikipy.math.geometry import clockwise_sort_points


class Graph(BikipyModel):
    vertices: NpNDArrayFp64 = ...

    @field_validator("vertices")
    def sort_clockwise(cls, value) -> NpNDArrayFp64:
        return clockwise_sort_points(value)

    @computed_field
    @cached_property
    def centroid(self) -> NpNDArrayFp64:
        return np.mean(self.vertices, axis=0)

    @computed_field
    @cached_property
    def linked_vertices(self) -> NpNDArrayFp64:
        # Pair points that are neighbours
        return np.append(self.vertices, np.expand_dims(self.vertices[0], 0), axis=0)

    @computed_field
    @cached_property
    def vertex_pairs(self) -> NpNDArrayFp64:
        return np.array(list(zip(self.linked_vertices, self.linked_vertices[1:])))

    @computed_field
    @cached_property
    def vertex_midpoints(self) -> NpNDArrayFp64:
        return self.linked_vertices[1:] - np.squeeze(np.diff(self.vertex_pairs, axis=1)) / 2.0

    @computed_field
    @cached_property
    def vertex_midpoint_distances_to_centroid(self) -> NpNDArrayFp64:
        return np.linalg.norm(self.centroid - self.vertex_midpoints, axis=1)

    @computed_field
    @cached_property
    def edge_vectors(self) -> NpNDArrayFp64:
        return np.diff(self.linked_vertices, axis=0)

    @computed_field
    @cached_property
    def edge_lengths(self) -> NpNDArrayFp64:
        return np.linalg.norm(self.edge_vectors, axis=1)
