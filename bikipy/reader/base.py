from abc import ABC, abstractmethod
from collections import abc
from functools import cached_property
from logging import getLogger
from typing import ClassVar, Generic, Hashable, Iterable, Optional, Type, TypeVar

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pydantic import Field, FilePath, validate_arguments
from pydantic.generics import GenericModel
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64, NDArrayUint8
from typing_extensions import Literal

from bikipy import runtime_settings
from bikipy._dev_utils.fields import enclosure_field, timestamp_index_field
from bikipy.core.base import BikipyHashable
from bikipy.core.video import VideoMetadataMixin
from bikipy.feature.midpoint import recursive_midpoint
from bikipy.perimeter.base import BasePerimeter
from bikipy.reader.model import model_data
from bikipy.reader.utils import compute_midpoint_label
from bikipy.utils.constants import TO_PARQUET_KWARGS
from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS

BAD_COORDINATE = (np.nan, np.nan, 0.0)  # x, y, likelihood


logger = getLogger(__name__)

Enclosure = TypeVar("Enclosure", bound=BasePerimeter)


class BaseReader(GenericModel, Generic[Enclosure], BikipyHashable, VideoMetadataMixin, ABC):
    df_path: FilePath = Field(..., description="Path to kinematic data, that will be " "converted to pd.DataFrame")
    df_read_kwargs: Optional[dict] = Field(
        default_factory=dict, description="Keyword arguments to pass to the padnas dataframe reader"
    )
    object_tracking_label_for_kinematics: str = Field(
        ..., description="Label of the node that will be used to track general animal movement"
    )
    trial_enclosure: Optional[Enclosure] = enclosure_field
    midpoint_groups: Optional[dict[str, tuple]] = Field(
        description="labels that consist of groups that should have their midpoints computed in the DataFrame"
    )

    model: bool = True
    model_method: Literal["arima", "median", "spline"] = Field(
        "arima", description="Post-hoc filtration method label for improving data accuracy, adapted from DeepLabCut"
    )
    model_kwargs: dict = Field(default_factory=dict)

    required_tail_likelihood: float = Field(
        0.8, description="The pd.DataFrame will be cropped to this combined likelihood score"
    )

    future_scaling: bool = Field(
        None,
        description="Scales the coordinates with respect to their min and max. " "True requires x_max and y_max",
    )

    cache_meters_augmented: bool = True

    timestamp_index: Optional[NDArrayFp64] = timestamp_index_field
    df_is_timestamped: bool = Field(
        False, description="When True, the reader will interpret the DataFrame index as timestamps in seconds"
    )

    x_axis_crop_end_point: float = Field(0.0, description="x component of the raw video crop of video")
    y_axis_crop_end_point: float = Field(0.0, description="y component of the raw video crop of video")
    invert_y_axis: bool = Field(
        runtime_settings.matplotlib_invert_y_axis,
        description=(
            "if True will invert the y-axis. Useful when the user wants to work in "
            "traditional Cartesian coordinate system where the origin is on "
            "the bottom-left"
        ),
    )

    crop_time_seconds: float = 0.0
    crop_from_end: bool = Field(
        False,
        description="Only affective if crop_frames is not 0. " "Will crop from start instead when set to False",
    )
    model_displacement_by_std: Optional[float] = Field(
        2.0,
        description="When set the maximum displacement by frame will have an upper bound defined by the given scale of the STD",
    )

    export_timestamp_data_as_parquet: bool = Field(
        False, description="When timestemp index is defined, export re-indexed df as parquet"
    )

    _using_bikipy_ingress: bool = Field(
        False,
        description="This is a flagg used by the developer to signal the use of bikipy ingress to the class. Currently, it only affects augmented df caching",
    )

    _post_read_midpoints: set = set()
    _time_index_derived_fps: float | None
    _region_of_interest_to_fused_neighbouring_points: dict[str, NDArrayUint8] = Field(default_factory=dict)
    _cached_augmented_df: pd.DataFrame | None

    augmented_coordinate_cached_file_label: ClassVar[str] = "augmented"

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(("df_path", "trial_enclosure", "timestamp_index", "_using_bikipy_ingress"))
        return upstream

    @abstractmethod
    def _isolate_coordinates(self, key: Iterable[Hashable] | Hashable) -> pd.DataFrame:
        ...

    @property
    @abstractmethod
    def region_of_interest_to_boolean_index(self) -> dict[str, NDArrayBool]:
        ...

    def __getitem__(self, query: Iterable[Hashable] | Hashable) -> pd.DataFrame:
        if not isinstance(query, str) and isinstance(query, abc.Iterable):
            return pd.merge([self._isolate_coordinates(item) for item in query], axis=1)
        else:
            if query not in self.all_tracked_labels:
                msg = f"'{query}' is not in object DataFrame (self.summary_frame)"
                raise AttributeError(msg)
            return self._isolate_coordinates(query)

    @property
    def kinematic_coordinates(self) -> pd.DataFrame:
        return self[self.object_tracking_label_for_kinematics]

    @cached_property
    def plot_prepared_kinematic_coordinates(self) -> NDArrayFp64:
        return self.video.prepare_coordinates_for_plotting(self.kinematic_coordinates)

    @cached_property
    def physically_tracked_labels(self) -> set[str]:
        return set(self.raw_df.columns.levels[0])

    @property
    def tracked_midpoint_labels(self) -> set[str]:
        result = self._post_read_midpoints.copy()
        if self.midpoint_groups:
            result.update(self.midpoint_groups.keys())
        return result

    @cached_property
    def all_tracked_labels(self) -> set[str]:
        return self.physically_tracked_labels | self.tracked_midpoint_labels

    @property
    def required_video_metadata_fields(self) -> set:
        base = {"meters_per_pixel", "recording_resolution"}
        if self.crop_time_seconds:
            base.add("fps")
        return base

    @property
    def augmented_file_name(self) -> str:
        stem = self.df_path.stem.replace("coordinates-", f"coordinates-{self.augmented_coordinate_cached_file_label}-")
        return f"{stem}.parquet"

    @property
    def cached_augmented_df_path(self) -> FilePath:
        return self.df_path.with_name(self.augmented_file_name)

    @cached_property
    def augmented(self) -> pd.DataFrame:
        """
        :return: Tracking and augmented data stored in the same frame. The augmented data should
        include midpoints and inner interpolations.
        """
        if self.cached_augmented_df_path.exists():
            return pd.read_parquet(self.cached_augmented_df_path)

        result = self.raw_df.copy()

        # Remove warm up tail with low likelihoods
        tail_likelihood_capped_boolean_idx = np.where(self.combined_raw_likelihood >= self.required_tail_likelihood)[0]
        start_idx, _end_idx = tail_likelihood_capped_boolean_idx[0], tail_likelihood_capped_boolean_idx[-1]
        logger.debug(
            f"Likelihood filtering (>={self.required_tail_likelihood}): " f"Slicing [{start_idx}:] from coordinates"
        )
        result = result.iloc[start_idx:]

        if self.trial_enclosure:
            logger.debug(
                f"Dataset {self.label}: "
                f"Removing coordinates outside the defined trial_enclosure, {self.trial_enclosure.label}"
            )
            for ptl in self.physically_tracked_labels:
                result.loc[:, ptl][
                    ~self.trial_enclosure.compute_confined_coordinate_boolean_index(
                        result.loc[:, pd.IndexSlice[ptl, ("x", "y")]].values
                    )
                ] = BAD_COORDINATE

        if self.model:
            logger.debug(f"Filtering {self.df_path.stem} with the {self.model_method} method")
            result = model_data(result, self.model_method, **self.model_kwargs)

        crop_frames = None
        if self.df_is_timestamped:
            try:
                crop_frames = np.where(self.raw_df.index.values >= self.crop_time_seconds)[0][0]
            except IndexError:
                logger.warning(f"The defined crop seconds, {self.crop_time_seconds}, is out of bounds for DataFrame")
                self.crop_time_seconds = False

        if self.crop_time_seconds:
            crop_frames = crop_frames or round(self.video.fps * self.crop_time_seconds)

            if crop_frames > self.raw_frames:
                logger.warning(
                    f"(cropping number_of_frames: {self.crop_time_seconds} seconds -> {crop_frames} number_of_frames) "
                    f"> total of {self.raw_frames} number_of_frames"
                )
            else:
                result = (
                    result.iloc[self.raw_frames - crop_frames :] if self.crop_from_end else result.iloc[:crop_frames]
                )

        if self.invert_y_axis:
            result.loc[:, pd.IndexSlice[:, "y"]] = self.video.vertical_resolution - result.loc[:, pd.IndexSlice[:, "y"]]

        if self.x_axis_crop_end_point:
            result.loc[:, pd.IndexSlice[:, "x"]] = self.x_axis_crop_end_point + result.loc[:, pd.IndexSlice[:, "x"]]
        if self.y_axis_crop_end_point:
            result.loc[:, pd.IndexSlice[:, "y"]] = result.loc[:, pd.IndexSlice[:, "y"]] - self.y_axis_crop_end_point

        # convert to meters
        if isinstance(self.video.meters_per_pixel, float):
            result.loc[:, pd.IndexSlice[:, ("x", "y")]] = (
                result.loc[:, pd.IndexSlice[:, ("x", "y")]] * self.video.meters_per_pixel
            )
        elif isinstance(self.video.meters_per_pixel, np.ndarray):
            result.loc[:, pd.IndexSlice[:, "x"]] = result.loc[:, pd.IndexSlice[:, "x"]] * self.video.meters_per_pixel[0]
            result.loc[:, pd.IndexSlice[:, "y"]] = result.loc[:, pd.IndexSlice[:, "y"]] * self.video.meters_per_pixel[1]
        else:
            raise RuntimeError(f"Could not match video.meters_per_pixel type: {type(self.video.meters_per_pixel)}")

        if self.midpoint_groups:
            generated_midpoints = set()
            midpoint_loop_iterator = list(self.midpoint_groups.items())
            for name, group in midpoint_loop_iterator:
                if (self.physically_tracked_labels | generated_midpoints).issuperset(group):
                    result = pd.concat(
                        (result, self._compute_midpoint(result, group, name)),
                        axis=1,
                    )
                    generated_midpoints.add(name)
                elif (self.tracked_midpoint_labels | self.physically_tracked_labels).issuperset(group):
                    # This midpoint depends on another midpoint, which has not been generated yet.
                    # Putting it at the end of the loop
                    midpoint_loop_iterator.append((name, group))
                else:
                    msg = (
                        f"{self.df_path}: Midpoint {name}, cannot be derived as its components are "
                        f"not defined in the tracked dataset nor in midpoint_groups.\n"
                        f"The following are tracked: {self.physically_tracked_labels}"
                    )
                    raise ValueError(msg)

        if self.cache_meters_augmented:
            self._cache_augmented(result)

        return result

    @property
    def df(self) -> pd.DataFrame:
        return self.augmented

    @property
    def raw_frames(self) -> int:
        return len(self.raw_df)

    @property
    def number_of_frames(self) -> int:
        return len(self.df)

    @property
    def duration_seconds(self) -> float:
        return self.number_of_frames / self.video.fps if self.timestamp_index is None else self.timestamp_index[-1]

    @property
    def info(self) -> pd.Series:
        return pd.Series(
            [self.raw_frames, self.number_of_frames, self.duration_seconds],
            index=[("Reader", "RawFrames"), ("Reader", "AugmentedFrames"), ("Reader", "DurationSeconds")],
        )

    def _read_hdf(self, path: FilePath) -> pd.DataFrame:
        return pd.read_hdf(path)

    def _read_parquet(self, path: FilePath) -> pd.DataFrame:
        return pd.read_parquet(path)

    @cached_property
    def raw_df(self) -> pd.DataFrame:
        match self.df_path.suffix:
            case ".h5" | ".hdf":
                df = self._read_hdf(self.df_path, **self.df_read_kwargs)
            case ".parquet":
                df = self._read_parquet(self.df_path, **self.df_read_kwargs)
            case _:
                msg = f"{self.df_path.suffix}, is not natively supported by DeepLabCut."
                raise ValueError(msg)

        if "timestamped" in self.df_path.stem:
            self.df_is_timestamped = True

        if self.timestamp_index is not None:
            df.set_index(self.timestamp_index, inplace=True)
            self.df_is_timestamped = True

        if isinstance(df.index, (np.timedelta64, pd.TimedeltaIndex)):
            df.index = df.index.values.astype(float) / 10.0**9.0
            self.df_is_timestamped = True

        return df

    @property
    def combined_raw_likelihood(self) -> NDArrayFp64:
        return np.multiply.reduce(self.raw_df.loc[:, pd.IndexSlice[:, "likelihood"]], axis=1)

    @cached_property
    def trial_length_seconds(self) -> float:
        if self.df_is_timestamped:
            return self.raw_df.index.values[-1]
        return len(self.raw_df) * self.fps

    @validate_arguments(config={"arbitrary_types_allowed": True})
    def plot_boolean_index(
        self, boolean_index: NDArrayBool, ax: Axes, manual_kinematic_coordinates: Optional[str] = None
    ) -> None:
        region_label = manual_kinematic_coordinates or self.object_tracking_label_for_kinematics
        ax.scatter(
            *self[region_label][boolean_index].T,
            marker="x",
            alpha=runtime_settings.matplotlib_scatter_alpha,
            label="Valid",
        )
        ax.scatter(
            *self[region_label][~boolean_index].T,
            marker="x",
            alpha=runtime_settings.matplotlib_scatter_alpha,
            label="Invalid",
        )
        plt.legend(**BOTTOM_LEGEND_KWARGS)

    def _compute_midpoint(
        self, df: pd.DataFrame, midpoint_group: Iterable[str], manual_midpoint_label: Optional[Hashable] = None
    ) -> pd.DataFrame:
        midpoint_label = compute_midpoint_label(midpoint_group, manual_midpoint_label)
        return pd.DataFrame(
            recursive_midpoint(
                *(df.loc[:, pd.IndexSlice[component_name, ("x", "y")]].values for component_name in midpoint_group)
            ),
            columns=[(midpoint_label, "x"), (midpoint_label, "y")],
            index=df.index,
        )

    def _cache_augmented(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.cached_augmented_df_path, **TO_PARQUET_KWARGS)

    def flush_reads(self) -> None:
        try:
            del self.raw_df
            del self.augmented
        except AttributeError as e:
            logger.error(str(e))


ReaderCLS = Type[BaseReader]
Reader = TypeVar("Reader", bound=BaseReader)
