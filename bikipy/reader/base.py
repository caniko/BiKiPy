from typing import Any, Sequence, Union

import pandas as pd

from bikipy.feature.movement import displacement_per_frame


class BaseReader:
    def __init__(
        self,
        df: pd.DataFrame,
        future_scaling: bool = False,
        pixel_resolution: Union[Sequence, None] = None,
        video_path: Any = None,
        data_path: Any = None,
        data_label: Union[str, None] = None,
    ):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            Kinematic data in a pd.DataFrame
        future_scaling : boolean, default False
            Scales the coordinates with respect to their min and max.
            True requires x_max and y_max
        pixel_resolution : Sequence
             The resolution of the videos that are being analyzed
        data_label : String; optional
            Label for the data
        """

        if pixel_resolution:
            self.pixel_resolution = pixel_resolution
            self.resolution = self.pixel_resolution

            self.horizontal_res, self.vertical_res = pixel_resolution
            if not (
                isinstance(self.horizontal_res, (float, int, type(None)))
                and isinstance(self.vertical_res, (float, int, type(None)))
            ):
                msg = f"x and y max are integers; not {self.horizontal_res}; {self.vertical_res}"
                raise AttributeError(msg)

        self.df = df
        if not isinstance(df, pd.DataFrame):
            msg = "df has to be a pandas.DataFrame"
            raise AttributeError(msg)

        self.future_scaling = future_scaling

        self.video_path = video_path
        self.data_path = data_path
        self.data_label = data_label

    def __getitem__(self, item):
        pass

    @property
    def items(self) -> tuple:
        """
        Returns
        -------
        Tuple containing the name of the columns of the DataFrame
        """
        return tuple(self.df.columns.levels[0])

    @property
    def regions_of_interest(self) -> tuple:
        return self.items

    def interpolate_item_displacement(self, item: str):
        return displacement_per_frame(self[str(item)])
