from functools import cached_property
from typing import Optional

from pydantic import BaseModel

from bikipy.behaviour.base import BaseExperiment, BaseTrial
from bikipy.perimeter.base import Perimeter2D, PerimeterSet
from bikipy.utils.typing import NDArray


class RadialMazeBase(BaseModel):
    corridor_width: float


class BaseRadialMazeExperiment(BaseExperiment, RadialMazeBase):
    pass


class BaseRadialMazeTrial(BaseTrial, RadialMazeBase):
    center: Perimeter2D
    arms: tuple
    reference_point: Optional[NDArray] = None

    def __init__(self, **data):
        if data["reference_point"]:
            data["center"] = data["center"].change_reference(data["reference_point"])
            data["arms"] = data["arms"].change_reference(data["reference_point"])

        data["center"].int_label = 1
        for i, arm_idx in enumerate(range(len(data["arms"])), start=2):
            data["arms"][arm_idx].int_label = i

        super().__init__(**data)

    @cached_property
    def perimeter_set(self):
        return PerimeterSet(perimeters=(self.center, *self.arms))

    @cached_property
    def unit_per_pixel(self):
        return self.center.mean_length / self.corridor_width

    @cached_property
    def _border_presence_data(self):
        return Perimeter.detect_sequential_border_presence(
            self.coordinates_per_frame,
            self.arms,
            inferior_poly_border_instances=[self.center],
        )

    @property
    def alternation_sequence(self):
        return self._border_presence_data[0]

    @property
    def valid_indices(self):
        return self._border_presence_data[1]

    @property
    def valid_boolean_index(self):
        return self._border_presence_data[2]
