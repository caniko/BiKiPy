from typing import Union, AnyStr, SupportsFloat, SupportsInt, Any
from collections.abc import Sequence
from warnings import warn

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from bikipy.utils.math import (
    unit_vector,
    orthogonal_vector,
    normal_from_line_to_point,
    find_intersection_between_two_vectors,
)
from bikipy.preferance.generate_border import parallelogram_input

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


class BorderOverlapError(Exception):
    def __init__(self, border_obj, *args, **kwargs):
        message = f"Border {border_obj.label} has data overlap with other borders"
        super().__init__(message, *args, **kwargs)


class Border:
    def __init__(
        self,
        label: Union[AnyStr, None] = None,
        guiding_image: Any = None,
    ):
        """
        Parameters
        ----------
        label: Optional, string
            Label for the border. Useful for manual audition and testing.
        """
        if label:
            try:
                self.label = str(label)
            except TypeError as e:
                msg = f"label has to be a string, not {type(label)}"
                raise TypeError(msg) from e

        self.guiding_image = guiding_image

    def plt_show(self, ax):
        if self.guiding_image:
            ax.imshow(Image.open(self.guiding_image))
        plt.show()


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

    def true_values(self, data) -> np.ndarray:
        return np.asanyarray(data)[self.location % data]

    def number_true(self, data: Sequence) -> float:
        return np.sum(self % data)


class ParallelogramBorder(Border):
    def __init__(
        self,
        base: Union[Sequence, None] = None,
        apex: Union[Sequence, None] = None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        base: Sequence
            The coordinates of the sides of the base of the parallelogram
        apex: Sequence
            The coordinates of the sides of the apex of the parallelogram
        guiding_image: Path to image
            Image used for annotating base and apex; apex and base cannot be defined
            if guiding_image is defined
        """
        super().__init__(**kwargs)

        if not base and not apex:
            if not self.guiding_image:
                msg = "Image cannot be defined when base and apex are defined"
                raise AttributeError(msg)
            self.base, self.apex = parallelogram_input(self.guiding_image)
        else:
            self.base, self.apex = base, apex

    @staticmethod
    def midpoint(close_corner, far_corner) -> np.ndarray:
        close_corner, far_corner = (
            np.asanyarray(close_corner),
            np.asanyarray(far_corner),
        )
        return close_corner + (far_corner - close_corner) / 2

    @staticmethod
    def sort_vectors(vectors: Sequence) -> np.ndarray:
        vectors = np.asanyarray(vectors)

        assert vectors.shape == (2, 2), (
            f"Vector set must define the endpoints of a side,"
            f"the shape must therefore be (2, 2) and not {vectors.shape}"
        )

        vector_norms = np.argsort(np.linalg.norm(vectors, axis=1))
        return vectors[vector_norms]

    @property
    def base(self):
        return self.__base

    @base.setter
    def base(self, base: Sequence):
        base = np.asanyarray(base)
        self.__base = self.sort_vectors(base)

        # self.base_mid = self.base[0] + (self.base[1] - self.base[0]) / 2
        self.base_mid = self.midpoint(*self.__base)
        self.base_vector = base[1] - base[0]

    @property
    def apex(self):
        return self.__apex

    @apex.setter
    def apex(self, apex: Sequence):
        apex = np.asanyarray(apex)
        self.__apex = self.sort_vectors(apex)

        # self.apex_mid = self.apex[0] + (self.apex[1] - self.apex[0]) / 2
        self.apex_mid = self.midpoint(*self.__apex)
        self.apex_vector = apex[1] - apex[0]

    @property
    def midline_vector(self):
        return self.apex_mid - self.base_mid

    @property
    def midline_unit(self):
        return unit_vector(self.midline_vector)

    @property
    def midline_unit_orthogonal(self):
        return orthogonal_vector(self.midline_unit)

    @property
    def midline_magnitude(self):
        return np.linalg.norm(self.midline_vector)

    @property
    def close_feet_vector(self):
        return self.apex[0] - self.base[0]

    @property
    def close_feet_unit(self):
        return unit_vector(self.close_feet_vector)

    @property
    def far_feet_vector(self):
        return self.apex[1] - self.base[1]

    @property
    def far_feet_unit(self):
        return unit_vector(self.far_feet_vector)

    def plot(self, points: Union[Sequence, None] = None, show: bool = True):
        fig, ax = plt.subplots()
        ax.plot(
            # Base
            (self.base[0][0], self.base[1][0]),
            (self.base[0][1], self.base[1][1]),
            "-r",
            # Apex
            (self.apex[0][0], self.apex[1][0]),
            (self.apex[0][1], self.apex[1][1]),
            "-c",
            # Close feet
            (self.base[0][0], self.apex[0][0]),
            (self.base[0][1], self.apex[0][1]),
            "-b",
            # Far feet
            (self.base[1][0], self.apex[1][0]),
            (self.base[1][1], self.apex[1][1]),
            "-g",
            # Midline
            (self.base_mid[0], self.apex_mid[0]),
            (self.base_mid[1], self.apex_mid[1]),
            "-k",
        )
        plt.legend(("Base", "Apex", "Close Feet", "Far Feet", "Midline"))
        if points is not None:
            points = np.asanyarray(points)
            ax.scatter(points.T[0], points.T[1], marker=".")
        if show:
            self.plt_show(ax)
        return fig, ax

    def _line_segment_magnitudes(self, data: Sequence):
        """
        Generate the magnitude of the line segment that goes from origin to the data
        Parameters
        ----------
        data

        Returns
        -------

        """
        data = np.asanyarray(data)
        magnitudes = np.apply_along_axis(
            lambda x: normal_from_line_to_point(self.midline_unit, self.base_mid, x),
            1,
            data,
        )
        return np.hsplit(magnitudes, 2)

    def confined_coordinate_indexes(self, data: Sequence):
        data = np.asanyarray(data)
        (
            base_to_midpoint_apex_magnitudes,
            midpoint_apex_to_coordinate_magnitudes,
        ) = self._line_segment_magnitudes(data)

        valid_data_boolean_indexes = base_to_midpoint_apex_magnitudes <= 0
        east_of_midpoint_booleans = midpoint_apex_to_coordinate_magnitudes <= 0

        # if self.base_mid[0] > self.apex_mid[0]:
        #     valid_data_boolean_indexes = base_to_midpoint_apex_magnitudes >= 0
        #     east_of_midpoint_booleans = midpoint_apex_to_coordinate_magnitudes >= 0
        # else:
        #     valid_data_boolean_indexes = base_to_midpoint_apex_magnitudes <= 0
        #     east_of_midpoint_booleans = midpoint_apex_to_coordinate_magnitudes <= 0

        valid_magnitudes = base_to_midpoint_apex_magnitudes[valid_data_boolean_indexes]
        expanded_valid_magnitudes = np.expand_dims(valid_magnitudes, 0).T

        line_segment_apex = (
            self.base_mid - expanded_valid_magnitudes * self.midline_unit
            if self.base_mid[0] > self.apex_mid[0]
            else self.base_mid + expanded_valid_magnitudes * self.midline_unit
        )

        compute = line_segment_apex.copy()
        east_of_midpoint_booleans = east_of_midpoint_booleans[
            valid_data_boolean_indexes
        ]
        west_of_midpoint_booleans = east_of_midpoint_booleans == False
        midline_to_data_norm = np.linalg.norm(
            data[valid_data_boolean_indexes.T[0]] - line_segment_apex, axis=1
        )

        # if self.base_mid[1] >= self.apex_mid[1]:
        #     compute[east_of_midpoint_booleans] = np.apply_along_axis(
        #         lambda x: find_intersection_between_two_vectors(
        #             self.midline_unit_orthogonal, self.far_feet_unit, x, self.base[1]
        #         ),
        #         1,
        #         compute[east_of_midpoint_booleans],
        #     )
        #     compute[west_of_midpoint_booleans] = np.apply_along_axis(
        #         lambda x: find_intersection_between_two_vectors(
        #             self.midline_unit_orthogonal, self.close_feet_unit, x, self.base[0]
        #         ),
        #         1,
        #         compute[west_of_midpoint_booleans],
        #     )
        #
        #     compute = compute + line_segment_apex
        #     # fig, ax = self.plot(show=False)
        #     # ax.scatter(compute.T[0], compute.T[1], marker="x")
        #     # self.plt_show(ax)
        #
        #     compute = np.linalg.norm(compute, axis=1)
        #     compute[east_of_midpoint_booleans] = compute[east_of_midpoint_booleans] >= midline_to_data_norm[east_of_midpoint_booleans]
        #     compute[west_of_midpoint_booleans] = compute[west_of_midpoint_booleans] <= midline_to_data_norm[west_of_midpoint_booleans]
        #
        # else:
        #     compute[west_of_midpoint_booleans] = np.apply_along_axis(
        #         lambda x: find_intersection_between_two_vectors(
        #             self.midline_unit_orthogonal, self.far_feet_unit, x, self.base[1]
        #         ),
        #         1,
        #         compute[west_of_midpoint_booleans],
        #     )
        #     compute[east_of_midpoint_booleans] = np.apply_along_axis(
        #         lambda x: find_intersection_between_two_vectors(
        #             self.midline_unit_orthogonal, self.close_feet_unit, x, self.base[0]
        #         ),
        #         1,
        #         compute[east_of_midpoint_booleans],
        #     )
        #     compute = np.linalg.norm(line_segment_apex - compute, axis=1)
        #     compute[east_of_midpoint_booleans] = (
        #         compute[east_of_midpoint_booleans]
        #         <= midline_to_data_norm[east_of_midpoint_booleans]
        #     )
        #     compute[west_of_midpoint_booleans] = (
        #         compute[west_of_midpoint_booleans]
        #         >= midline_to_data_norm[west_of_midpoint_booleans]
        #     )

        # Same as data
        # line_to_coord_segment = line_segment_apex + np.expand_dims(
        #     midpoint_apex_to_coordinate_magnitudes[valid_data_boolean_indexes],
        #     0).T * orthogonal_vector(self.midline_unit)

        1

        # fig, ax = self.plot(show=False)
        # ax.scatter(data[valid_data_boolean_indexes.T[0]].T[0], data[valid_data_boolean_indexes.T[0]].T[1], marker="x")
        # self.plt_show(ax)

        return valid_data_boolean_indexes.T[0]

    def confined_coordinates(self, data: Sequence, plot: bool = False):
        data = np.asanyarray(data)
        valid_points = data[self.confined_coordinate_indexes(data)]

        if plot:
            fig, ax = self.plot(show=False)
            ax.scatter(valid_points.T[0], valid_points.T[1], marker="x")
            self.plt_show(ax)

        return valid_points

    @classmethod
    def detect_sequential_border_presence(
        cls, data: Sequence, *instances, overlap: bool = False
    ):
        data = np.asanyarray(data)

        for obj in instances:
            if not isinstance(obj, cls):
                msg = f"The objects passed to this class method has to be instances of {cls.__name__}"
                raise ValueError(msg)

        presence = np.zeros(data.shape[0], dtype="object")

        for border in instances:
            confined_coord_index_booleans = border.confined_coordinate_indexes(data)

            if not overlap:
                if presence[confined_coord_index_booleans].any():
                    presence[
                        np.where(presence[confined_coord_index_booleans] == True)[0]
                    ] = np.nan
                    warn(f"Border {border.label} has data overlap with other borders")
                    # raise BorderOverlapError(border)
                presence[confined_coord_index_booleans] = border.label
            else:
                # IDEA: Add support for overlapping borders
                raise ValueError("Not implemented")
                pass

        return presence

    @classmethod
    def many(
        cls,
        guiding_image: Any,
        n: SupportsInt,
        object_kwargs: Union[Sequence, None] = None,
    ):
        return [
            cls(guiding_image=guiding_image, **object_kwargs[i]) for i in range(int(n))
        ]


class GradientBorder(ParallelogramBorder):
    def __init__(
        self,
        gradient_range: Sequence = (0.5, 1.0),
        **kwargs,
    ):
        """

        Parameters
        ----------
        base: Sequence
            The coordinates of the sides of the base of the parallelogram
        apex: Sequence
            The coordinates of the sides of the apex of the parallelogram
        gradient_range: Sequence
            The range of the gradient; (starting weight, end weight)
        guiding_image: Path to image
            Image used for annotating base and apex; apex and base cannot be defined
            if guiding_image is defined
        """
        super().__init__(**kwargs)

        self.gradient_range = gradient_range

    @property
    def gradient_range(self):
        return self.__gradient_range

    @gradient_range.setter
    def gradient_range(self, gradient_range: Sequence):
        gradient_min, gradient_max = tuple(gradient_range)
        if gradient_min >= gradient_max:
            msg = "gradient_range has to be less than self.gradient_max"
            raise ValueError(msg)

        if not 0.0 <= gradient_min <= 1.0 or not 0.0 <= gradient_max <= 1.0:
            msg = "gradient range must be between 0 and 1.0"
            raise ValueError(msg)

        self.__gradient_range = gradient_range

    @property
    def gradient(self):
        return np.linspace(*self.gradient_range, 100000)

    def weight_coordinates(self, data: Sequence, plot: AnyStr = "scatter"):
        data = np.asanyarray(data)
        (
            base_to_midpoint_apex_magnitudes,
            midpoint_apex_to_coordinate_magnitudes,
        ) = self.line_segment_magnitudes(data)

        valid_data_boolean_indexes = (
            base_to_midpoint_apex_magnitudes < 0
            if self.base_mid[0] > self.apex_mid[0]
            else base_to_midpoint_apex_magnitudes > 0
        )
        weights = np.abs(valid_magnitudes / self.midline_magnitude)

        if plot:
            fig, ax = self.plot(show=False)
            line_segment_apex = (
                self.base_mid - valid_magnitudes * self.midline_unit
                if self.base_mid[0] > self.apex_mid[0]
                else self.base_mid + valid_magnitudes * self.midline_unit
            )

            if plot == "scatter":
                ax.scatter(line_segment_apex.T[0], line_segment_apex.T[1])
            elif plot == "normal":
                ax.plot(
                    (line_segment_apex.T[0], data.T[0]),
                    (line_segment_apex.T[1], data.T[1]),
                )

            self.plt_show(ax)

        return weights


if __name__ == "__main__":
    from pathlib import Path
    from glob import glob
    import re, os

    from bikipy import DeepLabCutReader

    border_img = Path(
        "C:/Users/Can/Projects/Neuroscience/bikipy/examples/data/images/maze.png"
    )
    assert border_img.exists(), f"The image file doesn't exist in {border_img}"
    border_img = str(border_img)

    # borders = ParallelogramBorder.many(
    #     guiding_image=border_img, n=3,
    #     object_kwargs=[{"label": "A"}, {"label": "B"}, {"label": "C"}]
    # )

    ParallelogramBorder(guiding_image=border_img)

    # borders = [
    #     ParallelogramBorder(
    #         base=[[308.81687801, 193.11825], [288.30224458, 234.14751686]],
    #         apex=[[174.33205886, 120.17733115], [151.53802172, 158.92719429]],
    #         guiding_image=border_img,
    #         label="A",
    #     ),
    #     ParallelogramBorder(
    #         base=[[309.95657987, 193.11825], [335.03002072, 234.14751686]],
    #         apex=[[441.02229344, 116.75822557], [461.53692687, 154.36838686]],
    #         guiding_image=border_img,
    #         label="B",
    #     ),
    #     ParallelogramBorder(
    #         base=[[290.58164829, 234.14751686], [335.03002072, 234.14751686]],
    #         apex=[[294.00075387, 394.84547872], [338.44912629, 394.84547872]],
    #         guiding_image=border_img,
    #         label="C",
    #     ),
    # ]
    #
    # filename = "C:/Users/Can/Projects/Neuroscience/Imen/data/y_maze/before/Test 1DLC_resnet50_y_mazeSep13shuffle1_400000.h5"
    # f = DeepLabCutReader.from_hdf(
    #     filename, (640, 480), midpoint_groups=(("left_ear", "right_ear"),)
    # )
    # borders[0].plot()
    #
    #
    # DATA_DIR = Path("C:/Users/Can/Projects/Neuroscience/Imen/data")
    # exp_id_finder = re.compile("\d+")
    # data_dict = {}
    # for subdir in os.listdir(str(DATA_DIR)):
    #     for file_path in glob(os.path.join(str(DATA_DIR / subdir), "**.h5"))[1:]:
    #         filename = "C:/Users/Can/Projects/Neuroscience/Imen/data/y_maze/before/Test 1DLC_resnet50_y_mazeSep13shuffle1_400000.h5"
    #         f = DeepLabCutReader.from_hdf(
    #             file_path, (640, 480), midpoint_groups=(("left_ear", "right_ear"),)
    #         )
    #         # borders[0].plot(f["mid-left_ear-right_ear"])
    #         data_dict[
    #             (subdir, exp_id_finder.findall(Path(file_path).stem)[0])
    #         ] = ParallelogramBorder.detect_sequential_border_presence(
    #             f["mid-left_ear-right_ear"], *borders
    #         )
