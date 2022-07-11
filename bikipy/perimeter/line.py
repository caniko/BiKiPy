from typing import ClassVar, Literal, Optional

import numpy as np
from pydantic import Field, PositiveInt

from bikipy.core.base_class import BaseBikipyHashable

# 0: Use the x coordinate(s) as the perimeter
# 1: Use the y coordinate(s) as the perimeter
from bikipy.core.typing import NDArrayFp64
from bikipy.core.video import VideoMetadataMixin

ORIENTATION_TO_INDEX = {"vertical": 0, "horizontal": 1}
INDEX_TO_ORIENTATION = {0: "vertical", 1: "horizontal"}

LOGIC_ALTERNATIVES = ["<", "<=", ">", ">="]
LOGIC_TO_FUNC = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
}


class LinePerimeter(BaseBikipyHashable, VideoMetadataMixin):
    location: float = Field(description="The location given in pixels")
    orientation: str | PositiveInt = Field(
        description=(
            "A lower and an upper perimeter can be defined.\n"
            "The border_vertices can be oriented both horizontally (horizontal) and vertically (vertical).\n"
            "If vertical: lower -> right; upper -> left. With the use of the position_preference method, "
            "the ratio of timespent in the upper; the lower; the mid portion can be calculated."
            "For orientation to function, video_path or greater_than_borders andless_than_borders has to be defined."
        )
    )
    logic: Literal["<", "<=", ">", ">=", "=="] = Field(description="The logic of the perimeter")

    polygon_order: ClassVar[int] = 1

    @classmethod
    @property
    def exclude_from_settings_schema(cls) -> set[str]:
        """
        Some required fields for a class are sometimes highly specific to its respective object. These fields should
        be recorded in this class-property to be excluded by the settings generator function in the ingress module
        :return:
        """
        return super().exclude_from_settings_schema.union({"location", "orientation", "logic"})

    @property
    def feat_border(self):
        """Feature magnitude perimeter location"""
        return self.location / self.video.metric_resolution[self.orientation]

    @property
    def orientation_label(self):
        """Given name of orientation"""
        return INDEX_TO_ORIENTATION[self.orientation]

    def __mod__(self, other: NDArrayFp64) -> NDArrayFp64:
        """
        Compute values that are true to the perimeter logic

        other: 1 or 2 dimensional coordinates
        :return:
        """
        other = np.asarray(other)

        o_dimensions = len(other.shape)
        if o_dimensions > 2:
            msg = "The following workflow is not designed for R^n when n>2"
            raise ValueError(msg)
        elif o_dimensions == 2:
            other = other.T[self.orientation]

        return LOGIC_TO_FUNC[self.logic](self.location, other)

    def __repr__(self):
        return f"{self.logic}{self.location}; {self.orientation_label}"

    def true_values(self, coordinates: NDArrayFp64) -> NDArrayFp64:
        return np.asarray(coordinates)[self.location % coordinates]
