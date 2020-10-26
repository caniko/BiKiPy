from typing import Union, Any, AnyStr, Sequence, SupportsFloat
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import cv2

from bikipy.utils.video import get_video_data


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


class PolygonalBorder(Border):
    def __init__(
        self,
        feature_scale: Union[Sequence[SupportsFloat], None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.feature_scale = np.asanyarray(feature_scale) if feature_scale else None

    def confined_coordinates(
        self, coordinates: Sequence, plot: bool = False
    ) -> np.ndarray:
        """
        self.confined_coordinate_indexes to fetch and optionally plot the
        confined coordinates within the respective border

        Parameters
        ----------
        coordinates: Sequence
            Coordinates that will have their confinement tested

        plot: bool
            If True, plot the confined coordinates

        Returns
        -------
        np.ndarray, confined coordinates
        """

        coordinates = np.asanyarray(coordinates)
        confined_coordinates = coordinates[
            self.confined_coordinate_indexes(coordinates)
        ]
        if plot:
            fig, ax = self.plot(show=False)
            ax.scatter(confined_coordinates.T[0], confined_coordinates.T[1], marker="x")
            self.plt_show(ax)

        return confined_coordinates

    @classmethod
    def detect_sequential_border_presence(
        cls,
        coordinates: Sequence[Sequence[SupportsFloat]],
        superior_poly_border_instances: Union[Sequence, None],
        inferior_poly_border_instances: Union[Sequence, None] = None,
        clean_outliers: bool = True,
    ) -> np.ndarray:
        """
        Define sequential border confinements of coordinates

        Parameters
        ----------
        coordinates: Sequence
            Coordinates that will have their confinement tested

        superior_poly_border_instances: Sequence
            PolygonalBorder instances that will have the highest priority
            in case of overlap with respect to confinement

        inferior_poly_border_instances: Sequence
            PolygonalBorder instances that will have the lowest priority
            in case of overlap with respect to confinement

        clean_outliers
            Clear elements that aren't confined to any of the given borders
            as a final action before returning the sequential border presence

        Returns
        -------
        np.ndarray that stores the sequential border presence across frames
        """

        border_sequence = (
            list(inferior_poly_border_instances) + list(superior_poly_border_instances)
            if inferior_poly_border_instances
            else superior_poly_border_instances
        )
        coordinates = np.asanyarray(coordinates)
        presence = np.zeros(coordinates.shape[0], dtype=np.object)
        for border in border_sequence:
            confined_coord_booleans_index = border.confined_coordinate_indexes(
                coordinates
            )

            if presence[confined_coord_booleans_index].any():
                presence[np.where(presence[confined_coord_booleans_index])[0]] = np.nan
                warn(f"Border {border.label} has coordinate overlap with other borders")
                # raise BorderOverlapError(border)
            presence[confined_coord_booleans_index] = border.label

        if clean_outliers:
            presence = [e for e in presence[presence != 0] if isinstance(e, str)]

        return presence

    @property
    def feat_scaled_sides(self):
        if not self.feature_scale:
            msg = "Feature scale parameters have not been defined in this instance"
            raise AttributeError(msg)
        return self.sides / self.feature_scale


class GenericPolygonalBorder(PolygonalBorder):
    # Define "corners" (integer) as a class variable for the ginput in from_image(...)
    @classmethod
    def from_image(cls, guiding_image: Any, *args, **kwargs):
        """
        Define the corners of a polygon with a guiding image

        Parameters
        ----------
        guiding_image
            Either path to image or PIL.Image object with opened image inside

        Returns
        -------
        List with pixel coordinates of the polygon corners
        """

        if isinstance(guiding_image, np.ndarray):
            img = guiding_image
        else:
            img = cv2.imshow(guiding_image)
        plt.imshow(img)

        sides = plt.ginput(n=cls.corners, timeout=0)
        return cls(sides, *args, guiding_image=guiding_image, **kwargs)

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

        frame, x_res, y_res = get_video_data(video_path, frame_time)

        return cls.from_image(frame, *args, feature_scale=(x_res, y_res), **kwargs)
