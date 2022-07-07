from functools import cached_property

import numpy as np

from bikipy.core.base_class import BaseBikipy
from bikipy.core.typing import NDArrayFp64


class Graph(BaseBikipy):
    vertices: NDArrayFp64 = ...

    @cached_property
    def centroid(self) -> NDArrayFp64:
        return np.mean(self.vertices, axis=0)

    @cached_property
    def linked_vertices(self) -> NDArrayFp64:
        # Pair points that are neighbours
        return np.append(self.vertices, np.expand_dims(self.vertices[0], 0), axis=0)

    @cached_property
    def vertex_pairs(self) -> NDArrayFp64:
        return np.array((self.linked_vertices, self.linked_vertices[1:])).T

    @cached_property
    def vertex_midpoints(self) -> NDArrayFp64:
        return self.linked_vertices[1:] - np.diff(self.linked_vertices, axis=2).transpose(2, 0, 1)[0] / 2.0

    @cached_property
    def vertex_midpoint_distances_to_centroid(self) -> NDArrayFp64:
        return np.linalg.norm(self.centroid - self.vertex_midpoints, axis=1)

    @cached_property
    def edge_lengths(self) -> NDArrayFp64:
        return np.linalg.norm(np.diff(self.linked_vertices, axis=0), axis=1)
