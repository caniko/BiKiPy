from typing import Union, AnyStr, SupportsFloat, SupportsInt, Sequence

import numpy as np

from bikipy.border.base import Border


# 0: Use the x coordinate(s) as the border
# 1: Use the y coordinate(s) as the border
ORIENTATION_TO_INDEX = {"vertical": 0, "horizontal": 1}
INDEX_TO_ORIENTATION = {0: "vertical", 1: "horizontal"}

LOGIC_ALTERNATIVES = ["<", "<=", ">", ">="]
LOGIC_TO_FUNC = {
    ">": np.greater,
    ">=": np.greater_equal,
    "<": np.less,
    "<=": np.less_equal,
}


class LineBorder(Border):
    def __init__(
        self,
        location: Union[SupportsFloat, SupportsInt],
        orientation: Union[AnyStr, SupportsInt],
        logic: AnyStr,
        resolution: Union[SupportsInt, Sequence, None] = None,
        **kwargs,
    ):
        """
        location: int
            The location given in pixels
        orientation: AnyStr, int
            A lower and an upper border can be defined.
            The borders can be oriented both horizontally (horizontal)
            or vertically (vertical). If vertical: lower -> right; upper -> left.

            With the use of the position_preference method, the ratio of time
            spent in the upper; the lower; the mid portion can be calculated.

            For orientation to function, video_path or greater_than_borders and
            less_than_borders has to be defined.
        logic: {"<", "<=", ">", ">=", "=="}
            The logic of the border
        resolution: Sequence, int; optional
            The respective resolution of the frame. Border orient will
            be used to isolate the correct resolution if both vertical
            and horizontal are provided
        """
        super().__init__(**kwargs)

        if isinstance(orientation, str):
            self.orientation = ORIENTATION_TO_INDEX[orientation]
        elif isinstance(orientation, int):
            self.orientation = orientation
        else:
            msg = f"orientation has to be either string or support integer, and not {type(orientation)}"
            raise TypeError(msg)

        try:
            self.location = float(location)
        except TypeError as e:
            msg = f"location has to support float, and {type(location)} does not support float"
            raise TypeError(msg) from e

        if logic not in LOGIC_ALTERNATIVES:
            msg = f"logic has to be one of the following: {LOGIC_ALTERNATIVES}"
            raise ValueError(msg)
        self.logic = logic

        if resolution:
            try:
                if len(resolution) != 2:
                    msg = "resolution can only contain two elements; horizontal and vertical"
                    raise ValueError(msg)
            except TypeError as e:
                msg = f"resolution has to be a sequence, and not {type(resolution)}"
                raise TypeError(msg) from e

        self.resolution = resolution

    @property
    def feat_border(self):
        """ Feature magnituded border location """
        return self.location / self.resolution[self.orientation]

    @property
    def orientation_label(self):
        """ Given name of orientation """
        return INDEX_TO_ORIENTATION[self.orientation]

    def __mod__(self, other: Sequence) -> np.ndarray:
        """
        Compute values that are true to the border logic

        other: 1 or 2 dimensional coordinates
        :return:
        """
        other = np.asanyarray(other)

        o_dimensions = len(other.shape)
        if o_dimensions > 2:
            msg = "The following workflow is not designed for R^n when n>2"
            raise ValueError(msg)
        elif o_dimensions == 2:
            other = other.T[self.orientation]

        return LOGIC_TO_FUNC[self.logic](self.location, other)

    def __repr__(self):
        return f"{self.logic}{self.location}; {self.orientation_label}"

    def true_values(self, coordinates: Sequence) -> np.ndarray:
        return np.asanyarray(coordinates)[self.location % coordinates]

    def number_true(self, coordinates: Sequence) -> float:
        return np.sum(self % coordinates)
