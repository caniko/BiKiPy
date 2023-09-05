from abc import ABC, abstractmethod

from pydantic import BaseModel, computed_field

from bikipy.math.cached import cached_deg2rad, meters2pixels


class ProximityMixin(BaseModel, ABC):
    maximum_distance_meters: float = 0.05

    @property
    @abstractmethod
    def label_to_proximity_boolean(self) -> dict[str, NpNDArrayBool]:
        ...

    @computed_field
    @property
    def maximum_distance_pixels(self) -> float:
        return meters2pixels(self.maximum_distance_meters, self.video.pixels_per_meter)


class RayMixin(BaseModel, ABC):
    maximum_degrees: float = 45.0

    @computed_field
    @property
    def maximum_radians(self) -> float:
        return cached_deg2rad(self.maximum_degrees)

    @property
    @abstractmethod
    def label_to_ray_vector_direction_points(self) -> dict[str, NpNDArrayFp64]:
        ...


class SingleComponentMixin(BaseModel):
    def plot(self) -> None:
        fig, ax = self.video.subplot()
        self.plot_result(ax)
