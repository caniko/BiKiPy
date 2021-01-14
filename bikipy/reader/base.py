from typing import AnyStr, Sequence, Union

import numpy as np
import pandas as pd

from bikipy.feature.movement import displacement_per_frame


class BaseReader:
    def __init__(
        self,
        df: pd.DataFrame,
        future_scaling: bool = False,
        pixel_resolution: Union[Sequence, None] = None,
        data_label: Union[AnyStr, None] = None,
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
                isinstance(self.horizontal_res, (int, float, type(None)))
                and isinstance(self.vertical_res, (int, float, type(None)))
            ):
                msg = f"x and y max are integers; not {self.horizontal_res}; {self.vertical_res}"
                raise AttributeError(msg)

        self.df = df
        if not isinstance(df, pd.DataFrame):
            msg = "df has to be a pandas.DataFrame"
            raise AttributeError(msg)

        self.data_label = data_label
        self.future_scaling = future_scaling

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

    def displacement_per_frame(
        self, item, invalid_boolean_indexes: Union[Sequence[bool], None] = None
    ):
        coordinate_sequence = np.asanyarray(self[item])
        invalid_boolean_indexes = invalid_boolean_indexes or np.logical_not(
            coordinate_sequence
        )
        raw_displacement = displacement_per_frame(coordinate_sequence)
        if not invalid_boolean_indexes.any():
            return raw_displacement

        invalid_indexes = np.where(invalid_boolean_indexes)[0]

        i = 0
        while i < invalid_indexes.size:
            start_index = invalid_indexes[i]
            stop_index = invalid_indexes[i] + 1
            z = 1
            while stop_index == invalid_indexes[i + z]:
                stop_index += 1
                z += 1
            assert (
                np.isnan(raw_displacement[start_index:stop_index]).all(),
                np.isnan(raw_displacement[2 - start_index : stop_index + 2]),
            )

            start_point = coordinate_sequence[start_index - 1]
            stop_point = coordinate_sequence[stop_index + 1]
            mean = (start_point - stop_point) / z
