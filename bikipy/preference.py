from typing import Union, Sequence
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .readers import DeepLabCutReader


# 0: Use the x coordinate(s) as the border
# 1: Use the y coordinate(s) as the border
ORIENTATION_TO_INDEX = {"vertical": 0, "horizontal": 1}
AXIS_INDEX_TO_NAME = {0: "x", 1: "y"}


class Preference:
    @classmethod
    def from_image(
        cls,
        kin_data: DeepLabCutReader,
        border_orient: str,
        img,
        feature_scale_resolution: Union[Sequence, None] = None,
    ):
        """ Initialize class using data from a sample frame/image

        Parameters
        ----------
        kin_data
            Kinematic data stored in DeepLabCutReader object
        border_orient: str; {"vertical", "horizontal"}
            The orientation of the border used for analysis
        img
            object containing the image that will be used to determine the border location
        feature_scale_resolution: tuple-like; optional
            Resolution of the video that will be used for feature scaling

        Returns
        -------
        __init__ call
        """

        plt.imshow(img)
        plt.title("Lower limit")
        # coordinate for the lower border
        first_border = plt.ginput()[0][ORIENTATION_TO_INDEX[border_orient]]
        plt.title("Upper limit")
        # coordinate for the upper border
        second_border = plt.ginput()[0][ORIENTATION_TO_INDEX[border_orient]]

        return cls(
            kin_data,
            border_orient,
            first_border,
            second_border,
            feature_scale_resolution,
        )

    @classmethod
    def from_video(
        cls, kin_data: DeepLabCutReader, border_orient: str, video_path: str
    ):
        """ Initialize class using data from a sample video file

        Parameters
        ----------
        kin_data
            Kinematic data stored in DeepLabCutReader object
        border_orient: str; {"vertical", "horizontal"}
            The orientation of the border used for analysis
        video_path: str
            The name of the video file in local directory to be used for analysis.
            Required if border_orient == 'lasso' and frame == None.
            None: No action

            str:  The video file matching the string will be selected.
                  File extension must be included.

            True: If there is only one video file, it will be selected.

        Returns
        -------
        __init__ call
        """
        from bikipy.utils.video import get_video_data

        frame, x_res, y_res = get_video_data(video_path)

        return cls.from_image(
            kin_data, border_orient, frame, feature_scale_resolution=(x_res, y_res)
        )

    def __init__(
        self,
        kin_data: pd.DataFrame,
        border_orient: str,
        first_border: tuple,
        second_border: tuple,
        feature_scale_resolution: tuple = None,
    ):
        """
        kin_data: DeepLabCutReader object
            Data container with data to be analysed.
        border_orient: {'horizontal', 'vertical', 'lasso'}, default None
            Optional. A lower and an upper border can be defined.
            The borders can be oriented both horizontally (horizontal)
            or vertically (vertical). If vertical: lower -> right; upper -> left.

            With the use of the position_preference method, the ratio of time
            spent in the upper; the lower; the mid portion can be calculated.

            For border_orient to function, video_path or first_border and
            second_border has to be defined.
        first_border: int
            Variable to define lower border manually.
            See border_orient for context
        second_border: int
            Variable to define upper border manually.
            See border_orient for context
        feature_scale_resolution: bool, default False
            Define if data is normalized
        """
        if not isinstance(kin_data, pd.DataFrame):
            msg = "kin_data has to be a pandas data frame"
            raise ValueError(msg)

        if border_orient != "horizontal" and border_orient != "vertical":
            msg = (
                f"border_orient is either horizontal or vertical, "
                f"and not {border_orient}"
            )
            raise ValueError(msg)

        self.kin_data = kin_data

        if feature_scale_resolution:
            orientation_to_normal_reference = {
                "vertical": feature_scale_resolution[0],
                "horizontal": feature_scale_resolution[1],
            }
            self.first_border /= orientation_to_normal_reference[border_orient]
            self.second_border /= orientation_to_normal_reference[border_orient]
        else:
            self.first_border = first_border
            self.second_border = second_border

        self.feature_scale_resolution = feature_scale_resolution
        self.border_orient = border_orient

    def __repr__(self):
        return (
            f"border orientation: "
            f"first={self.first_border}; second={self.second_border}\n"
            f"DeepLabCutReader:\n{self.kin_data}"
        )

    def area_preference(self, plot=False, border_orient="horizontal"):
        if border_orient == "horizontal":
            or_var = "y"
        else:
            or_var = "x"

        rem_data = self.kin_data
        total_frames = rem_data.shape[0] - 1
        nose = rem_data.nose[or_var].values
        left_ear = rem_data.left_ear[or_var].values
        right_ear = rem_data.right_ear[or_var].values

        # Disregard warnings as they arise from NaN being compared to numbers
        np.warnings.filterwarnings("ignore")

        if border_orient == "vertical":
            nose_test = np.logical_or(
                np.less(nose, self.first_border), np.greater(nose, self.second_border)
            )

            lower_test = np.logical_or(
                np.less(left_ear, self.first_border, where=nose_test),
                np.less(right_ear, self.first_border, where=nose_test),
                where=nose_test,
            )

            lower_result = np.sum(np.extract(lower_test, nose_test))

            upper_test = np.logical_or(
                np.greater(left_ear, self.second_border, where=nose_test),
                np.greater(right_ear, self.second_border, where=nose_test),
                where=nose_test,
            )

            upper_result = np.sum(np.extract(upper_test, nose_test))

        else:
            nose_test = np.logical_or(
                np.greater(nose, self.first_border), np.less(nose, self.second_border)
            )

            lower_test = np.logical_or(
                np.greater(left_ear, self.first_border, where=nose_test),
                np.greater(right_ear, self.first_border, where=nose_test),
                where=nose_test,
            )

            lower_result = np.sum(np.extract(lower_test, nose_test))

            upper_test = np.logical_or(
                np.less(left_ear, self.second_border, where=nose_test),
                np.less(right_ear, self.second_border, where=nose_test),
                where=nose_test,
            )

            upper_result = np.sum(np.extract(upper_test, nose_test))

        percent_lower = (lower_result / total_frames) * 100
        percent_upper = (upper_result / total_frames) * 100

        rest = 100 - percent_lower - percent_upper

        if plot:
            labels = ("Bottom", "Top", "Elsewhere")
            sizes = (percent_lower, percent_upper, rest)

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
            ax1.axis("equal")  # Ensures that pie is drawn as a circle.
            plt.show()
        else:
            return percent_lower, percent_upper, rest


def dynamic_relative_position(
    df: pd.DataFrame, ref_point: Sequence, points: Sequence, axis: int = 1
) -> dict:
    """ Determine the location of an object with respect to either a
    vertical (axis=0 -> x) or horizontal (axis=1 -> y) line

    Parameters
    ----------
    df: pd.DataFrame
        Data to be analysed
    ref_point: tuple-like
        The point that is to be used to determine the location of the border
    points: tuple-like
        Points of that are to be tracked with respect to the border
    axis: int
        The axis of which the border spans

    Returns
    -------
    dict: point -> True if Left (x); Above (y)
    """

    try:
        ref = df.loc[:, (ref_point, AXIS_INDEX_TO_NAME[axis])].values
    except KeyError as e:
        msg = f"axis can be set to either 0 or 1, and not {axis}"
        raise ValueError(msg) from e

    result = {
        point: df.loc[:, (point, AXIS_INDEX_TO_NAME[axis])].values < ref
        for point in points
    }

    return result
