from typing import Union, Any, AnyStr, Sequence, SupportsInt, List
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import cv2

from bikipy.utils.video import get_video_data


class BorderOverlapError(Exception):
    def __init__(self, border_obj, *args, **kwargs):
        message = (
            f"Border {border_obj.label} has coordinates overlap with other borders"
        )
        super().__init__(message, *args, **kwargs)


class Border:
    def __init__(
        self,
        guiding_image: Union[AnyStr, None] = None,
        label: Union[AnyStr, None] = None,
    ):
        """
        Parameters
        ----------
        guiding_image: Optional, string
            Label for the border. Useful for manual audition and testing.
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
            ax.imshow(cv2.imshow(self.guiding_image))
        plt.show()

    def confined_coordinates(self, coordinates: Sequence, plot: bool = False):
        coordinates = np.asanyarray(coordinates)
        valid_points = coordinates[self.confined_coordinate_indexes(coordinates)]

        if plot:
            fig, ax = self.plot(show=False)
            ax.scatter(valid_points.T[0], valid_points.T[1], marker="x")
            self.plt_show(ax)

        return valid_points

    @classmethod
    def detect_sequential_border_presence(
        cls, coordinates: Sequence, instances: Sequence,
        overlap: bool = False, overlap_inferior: Union[Sequence, None] = None
    ):
        coordinates = np.asanyarray(coordinates)
        presence = np.zeros(coordinates.shape[0], dtype="object")
        for border in list(overlap_inferior) + list(instances):
            confined_coord_booleans_index = border.confined_coordinate_indexes(
                coordinates
            )

            if not overlap:
                if presence[confined_coord_booleans_index].any():
                    presence[
                        np.where(presence[confined_coord_booleans_index])[0]
                    ] = np.nan
                    warn(
                        f"Border {border.label} has coordinate overlap with other borders"
                    )
                    # raise BorderOverlapError(border)
                presence[confined_coord_booleans_index] = border.label
            else:
                # IDEA: Add support for overlapping borders
                raise ValueError("Not implemented")
                pass

        return presence


class PolygonalBorder(Border):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_image(cls, guiding_image: Any, corners: SupportsInt = 3, *args, **kwargs):
        """
        Define the corners of a polygon with a guiding image

        Parameters
        ----------
        guiding_image
            Either path to image or PIL.Image object with opened image inside
        corners
            Number of corners on polygon

        Returns
        -------
        List with pixel coordinates of the polygon corners
        """
        if isinstance(guiding_image, np.ndarray):
            img = guiding_image
        else:
            img = cv2.imshow(guiding_image)
        plt.imshow(img)

        sides = plt.ginput(n=corners, timeout=0)
        return cls(*sides, *args, guiding_image=guiding_image, **kwargs)

    @classmethod
    def from_video(
        cls, video_path: Any, frame_time: AnyStr = "middle", *args, **kwargs
    ):
        """
        Initialize class using data from a sample video file

        Parameters
        ----------
        video_path: str
            The name of the video file in local directory to be used for analysis.
            Required if orientation == 'lasso' and frame == None.
            None: No action

            str:  The video file matching the string will be selected.
                  File extension must be included.

            True: If there is only one video file, it will be selected.
        frame_time: str
            Relative location of the frame used for reference in analysis
        corners: int
            Number of corners on polygon

        Returns
        -------
        bikipy.border.polygon.polygon_corners_on_image call with frame from video
        """
        frame, _x_res, _y_res = get_video_data(video_path, frame_time)

        return cls.from_image(frame, *args, **kwargs)
