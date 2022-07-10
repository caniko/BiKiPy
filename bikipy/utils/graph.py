from functools import cached_property

import numpy as np
from pydantic import validator

from bikipy.core.base_class import BaseBikipy
from bikipy.core.typing import NDArrayFp64
from bikipy.utils.math.geometry import clockwise_sort_points


class Graph(BaseBikipy):
    vertices: NDArrayFp64 = ...

    @validator("vertices")
    def sort_clockwise(cls, value) -> NDArrayFp64:
        return clockwise_sort_points(value)

    @cached_property
    def centroid(self) -> NDArrayFp64:
        return np.mean(self.vertices, axis=0)

    @cached_property
    def linked_vertices(self) -> NDArrayFp64:
        # Pair points that are neighbours
        return np.append(self.vertices, np.expand_dims(self.vertices[0], 0), axis=0)

    @cached_property
    def vertex_pairs(self) -> NDArrayFp64:
        return np.array(list(zip(self.linked_vertices, self.linked_vertices[1:])))

    @cached_property
    def vertex_midpoints(self) -> NDArrayFp64:
        return self.linked_vertices[1:] - np.squeeze(np.diff(self.vertex_pairs, axis=1)) / 2.0

    @cached_property
    def vertex_midpoint_distances_to_centroid(self) -> NDArrayFp64:
        return np.linalg.norm(self.centroid - self.vertex_midpoints, axis=1)

    @cached_property
    def edge_vectors(self) -> NDArrayFp64:
        return np.diff(self.linked_vertices, axis=0)

    @cached_property
    def edge_lengths(self) -> NDArrayFp64:
        return np.linalg.norm(self.edge_vectors, axis=1)
