from abc import ABC, abstractmethod
from collections import abc, defaultdict
from functools import cached_property
from logging import getLogger
from typing import Hashable, Iterable, Optional

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from pydantic import ConfigDict, Field, FilePath, computed_field, validate_call
from pydantic_numpy.typing import Np1DArrayBool, NpNDArrayFp64
from typing_extensions import Literal

from bikipy import runtime_settings
from bikipy._constant import AUGMENTED_COORDINATE_CACHED_FILE_LABEL
from bikipy._dev_utils.fields import enclosure_field, timestamp_index_field
from bikipy.core.base import BikipyHashable
from bikipy.core.typing import ConfinementSequence
from bikipy.core.video import VideoMetadataMixin
from bikipy.feature.midpoint import recursive_midpoint
from bikipy.math.high_velocity import high_velocity_removal
from bikipy.math.shortcut import seconds_to_frames
from bikipy.perimeter.base import BasePerimeter
from bikipy.reader.compute import compute_midpoint_label, trial_video_frame_slice
from bikipy.reader.model import model_data
from bikipy.utils.constants import TO_PARQUET_KWARGS
from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS
from bikipy.utils.plot.color import make_color_map
from bikipy.utils.plot.generic import ax_plot_coordinate_with_boolean_index

BAD_COORDINATE = (np.nan, np.nan, 0.0)  # x, y, likelihood


logger = getLogger(__name__)


class BaseReader(BikipyHashable, VideoMetadataMixin, ABC):
    model_config = ConfigDict(extra="allow")

    df_path: FilePath = Field(description="Path to kinematic data, that will be " "converted to pd.DataFrame")
    df_read_kwargs: Optional[dict] = Field(
        default_factory=dict, description="Keyword arguments to pass to the padnas dataframe reader"
    )
    object_tracking_label_for_kinematics: str = Field(
        description="Label of the node that will be used to track general animal movement"
    )
    trial_enclosure: Optional[BasePerimeter] = enclosure_field
    midpoint_groups: Optional[dict[str, tuple]] = Field(
        description="labels that consist of groups that should have their midpoints computed in the DataFrame"
    )

    max_meters_per_second: Optional[float] = 0.5

    stat_model: bool = True
    stat_model_method: Literal["arima", "median", "spline"] = Field(
        "arima", description="Post-hoc filtration method label for improving data accuracy, adapted from DeepLabCut"
    )
    stat_model_kwargs: dict = Field(default_factory=dict)
    stat_model_displacement_by_std: Optional[float] = Field(
        2.0,
        description="When set the maximum displacement by frame will have an upper "
        "bound defined by the given scale of the STD",
    )

    required_tail_likelihood: float = Field(
        0.8, description="The pd.DataFrame will be cropped to this combined likelihood score"
    )

    skeleton_edges: tuple[tuple[str, str], ...] = Field(default_factory=tuple)

    cache_meters_augmented: bool = True

    manual_timestamp_index: Optional[NpNDArrayFp64] = timestamp_index_field

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

    seconds_to_try_to_crop_from_start: float = 0.0
    seconds_to_try_to_crop_from_end: float = 0.0

    crop_target_trial_length_seconds: Optional[float] = Field(
        description="Target seconds of the trial, achieved by cropping from start, "
        "unless crop_target_from_end is True. When seconds_to_try_to_crop_from_start or seconds_to_try_to_crop_from_end is defined, "
        "they are considered as minimums, the target cropper may change these values."
    )
    crop_target_from_end: bool = Field(
        False,
        description="Evaluated when crop_target_trial_length_seconds is not 0. "
        "Will crop from start instead when set to False",
    )

    export_timestamp_data_as_parquet: bool = Field(
        False, description="When timestamp index is defined, export re-indexed df as parquet"
    )

    # TODO: Convert back to private
    using_bikipy_ingress: bool = Field(
        False,
        description="This is a flagg used by the developer to signal the use of bikipy ingress to the class. "
        "Currently, it only affects augmented df caching",
    )
    # TODO: Convert back to private
    post_read_midpoints: set = Field(default_factory=set)

    def __getitem__(self, query: Iterable[Hashable] | Hashable) -> pd.DataFrame:
        if not isinstance(query, str) and isinstance(query, abc.Iterable):
            return pd.merge([self._isolate_coordinates(item) for item in query], axis=1)
        else:
            if query not in self.all_tracked_labels:
                msg = f"'{query}' is not in object DataFrame (self.summary_frame): {self.all_tracked_labels}"
                raise AttributeError(msg)
            return self._isolate_coordinates(query)

    @classmethod
    @property
    def fields_to_exclude_from_single_schema(cls) -> set[str]:
        result = super().fields_to_exclude_from_single_schema
        result.update(("df_path", "trial_enclosure", "manual_timestamp_index", "using_bikipy_ingress"))
        return result

    @property
    @abstractmethod
    def region_of_interest_to_boolean_index(self) -> dict[str, Np1DArrayBool]: ...

    @staticmethod
    @abstractmethod
    def isolate_coordinates_from_native_df(df: pd.DataFrame, key: Iterable[str] | str) -> NpNDArrayFp64: ...

    @computed_field  # type: ignore[misc]
    @cached_property
    def max_pixels_per_frame(self) -> float | None:
        if self.max_meters_per_second:
            return self.video.pixels_per_meter * self.max_meters_per_second / self.video.fps

    @computed_field  # type: ignore[misc]
    @property
    def kinematic_coordinates(self) -> pd.DataFrame:
        return self[self.object_tracking_label_for_kinematics]

    @computed_field  # type: ignore[misc]
    @cached_property
    def kinematic_coordinates_prepared_for_plotting(self) -> NpNDArrayFp64:
        return self.video_for_computation().prepare_coordinates_for_plotting(self.kinematic_coordinates)

    @computed_field  # type: ignore[misc]
    @cached_property
    def physically_tracked_labels(self) -> set[str]:
        return set(self.raw_df.columns.levels[0])

    @computed_field  # type: ignore[misc]
    @cached_property
    def tracked_midpoint_labels(self) -> set[str]:
        result = self.post_read_midpoints.copy()
        if self.midpoint_groups:
            result.update(self.midpoint_groups.keys())
        return result

    @computed_field  # type: ignore[misc]
    @cached_property
    def all_tracked_labels(self) -> set[str]:
        return self.physically_tracked_labels | self.tracked_midpoint_labels

    @computed_field  # type: ignore[misc]
    @cached_property
    def label_to_plot_color(self) -> dict[str, matplotlib.colors.ListedColormap]:
        return {
            label: color for label, color in zip(self.all_tracked_labels, make_color_map(len(self.all_tracked_labels)))
        }

    @computed_field  # type: ignore[misc]
    @property
    def required_video_metadata_fields(self) -> set:
        base = {"meters_per_pixel", "resolution"}
        if (
            self.seconds_to_try_to_crop_from_start
            or self.seconds_to_try_to_crop_from_end
            or self.crop_target_trial_length_seconds
        ):
            base.add("fps")
        return base

    @computed_field  # type: ignore[misc]
    @property
    def likelihood_columns(self) -> NpNDArrayFp64:
        return self.raw_df.loc[:, pd.IndexSlice[:, "likelihood"]]

    _df_is_timestamped: bool = False

    @computed_field  # type: ignore[misc]
    @cached_property
    def raw_df(self) -> pd.DataFrame:
        match self.df_path.suffix:
            case ".h5" | ".hdf":
                df = pd.read_hdf(self.df_path, **self.df_read_kwargs)
            case ".parquet":
                df = pd.read_parquet(self.df_path, **self.df_read_kwargs)
            case _:
                msg = f"{self.df_path.suffix}, is not natively supported by DeepLabCut."
                raise ValueError(msg)

        assert not self.raw_df.empty

        if "timestamped" in self.df_path.stem:
            self._df_is_timestamped = True

        if self.manual_timestamp_index is not None:
            df.set_index(self.manual_timestamp_index, inplace=True)
            self._df_is_timestamped = True

        if isinstance(df.index, (np.timedelta64, pd.TimedeltaIndex)):
            # Converting the timestamps to nanoseconds and then to seconds.
            df.index = df.index.values.astype(float) / 10.0**9.0
            self._df_is_timestamped = True

        return df

    @computed_field  # type: ignore[misc]
    @cached_property
    def df_is_timestamped(self) -> bool:
        """
        When True, the reader will interpret the DataFrame index as timestamps in seconds

        :return:
        """
        assert not self.raw_df.empty
        return self._df_is_timestamped

    @computed_field  # type: ignore[misc]
    @property
    def cached_augmented_df_path(self) -> FilePath:
        stem = self.df_path.stem.replace("coordinates-", f"coordinates-{AUGMENTED_COORDINATE_CACHED_FILE_LABEL}-")
        return self.df_path.with_name(f"{stem}.parquet")

    @computed_field  # type: ignore[misc]
    @cached_property
    def crop_frames_slice(self) -> slice:
        fps = self.video_for_computation().fps

        return trial_video_frame_slice(
            self.likelihood_columns,
            self.required_tail_likelihood,
            seconds_to_frames(self.crop_target_trial_length_seconds, fps),
            seconds_to_frames(self.seconds_to_try_to_crop_from_start, fps),
            seconds_to_frames(self.seconds_to_try_to_crop_from_end, fps),
            self.crop_target_from_end,
        )

    @computed_field  # type: ignore[misc]
    @cached_property
    def augmented(self) -> pd.DataFrame:
        """
        :return: Tracking and augmented data stored in the same frame. The augmented data should
        include midpoints and inner interpolations.
        """
        if self.cached_augmented_df_path.exists():
            return pd.read_parquet(self.cached_augmented_df_path)

        result = self.raw_df.copy()

        result = result.iloc[self.crop_frames_slice]

        if self.trial_enclosure:
            logger.debug(
                f"Dataset {self.label}: "
                f"Removing coordinates outside the defined trial_enclosure, {self.trial_enclosure.label}"
            )
            for ptl in self.physically_tracked_labels:
                result.loc[:, ptl][
                    ~self.trial_enclosure.compute_confinement_boolean_index(
                        result.loc[:, pd.IndexSlice[ptl, ("x", "y")]].values, potential_label="trial_enclosure"
                    )
                ] = BAD_COORDINATE

        if self.max_meters_per_second:
            for ptl in self.physically_tracked_labels:
                result.loc[:, pd.IndexSlice[ptl, ["x", "y"]]] = high_velocity_removal(
                    result.loc[:, pd.IndexSlice[ptl, ["x", "y"]]].values, self.max_pixels_per_frame
                )

        if self.stat_model:
            logger.debug(f"Filtering {self.df_path.stem} with the {self.stat_model_method} method")
            result = model_data(result, self.stat_model_method, **self.stat_model_kwargs)

        if self.invert_y_axis:
            result.loc[:, pd.IndexSlice[:, "y"]] = (
                self.video_for_computation().vertical_resolution - result.loc[:, pd.IndexSlice[:, "y"]]
            )
            self.y_axis_crop_end_point = -self.y_axis_crop_end_point

        if self.x_axis_crop_end_point:
            result.loc[:, pd.IndexSlice[:, "x"]] = self.x_axis_crop_end_point + result.loc[:, pd.IndexSlice[:, "x"]]
        if self.y_axis_crop_end_point:
            result.loc[:, pd.IndexSlice[:, "y"]] = result.loc[:, pd.IndexSlice[:, "y"]] + self.y_axis_crop_end_point

        # convert to meters
        if isinstance(self.video_for_computation().meters_per_pixel, float):
            result.loc[:, pd.IndexSlice[:, ("x", "y")]] = (
                result.loc[:, pd.IndexSlice[:, ("x", "y")]] * self.video_for_computation().meters_per_pixel
            )
        elif isinstance(self.video_for_computation().meters_per_pixel, np.ndarray):
            result.loc[:, pd.IndexSlice[:, "x"]] = (
                result.loc[:, pd.IndexSlice[:, "x"]] * self.video_for_computation().meters_per_pixel[0]
            )
            result.loc[:, pd.IndexSlice[:, "y"]] = (
                result.loc[:, pd.IndexSlice[:, "y"]] * self.video_for_computation().meters_per_pixel[1]
            )
        else:
            raise TypeError(
                f"Could not match video.meters_per_pixel type: {type(self.video_for_computation().meters_per_pixel)}"
            )

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

        if self.skeleton_edges:
            for node_name_a, node_name_b in self.skeleton_edges:
                distances = self.isolate_coordinates_from_native_df(
                    result, node_name_a
                ) - self.isolate_coordinates_from_native_df(result, node_name_b)
                np.diff(distances)

        if self.cache_meters_augmented:
            self._cache_augmented(result)

        return result

    @computed_field  # type: ignore[misc]
    @property
    def df(self) -> pd.DataFrame:
        return self.augmented

    @computed_field  # type: ignore[misc]
    @property
    def frames(self) -> int:
        return len(self.df)

    @computed_field  # type: ignore[misc]
    @cached_property
    def timestamp_index(self) -> NpNDArrayFp64 | None:
        if not self.df_is_timestamped:
            return

        result = self.raw_df.index.values

        return result[self.crop_frames_slice] if self.crop_frames_slice else result

    @computed_field  # type: ignore[misc]
    @cached_property
    def fps_from_timestamped_index(self) -> float | None:
        if self.timestamp_index is None:
            return None

        # We convert timestamps to time difference, i.e. delta(seconds)
        time_deltas = np.diff(self.timestamp_index)

        per_second_counts = []
        count = 0
        cum_sum = 0.0
        for time_delta in time_deltas:
            cum_sum += time_delta
            count += 1
            if cum_sum >= 1.0:
                per_second_counts.append(count)
                count = 1
                cum_sum = cum_sum - 1.0

        return float(np.mean(per_second_counts))

    @computed_field  # type: ignore[misc]
    @property
    def trial_start_seconds(self) -> float:
        return self.crop_frames_slice.start / self.fps

    @computed_field  # type: ignore[misc]
    @property
    def trial_end_seconds(self) -> float:
        return self.crop_frames_slice.stop / self.fps

    @computed_field  # type: ignore[misc]
    @property
    def trial_length_seconds(self) -> float:
        return self.trial_end_seconds - self.trial_start_seconds

    label_to_plot_prepped_coordinates: dict[str, NpNDArrayFp64] | None = Field(default_factory=dict)
    label_to_plot_without_resized_coordinates: dict[str, NpNDArrayFp64] | None = Field(default_factory=dict)

    def coordinates_for_plot(self, label_to_plot: str, with_resize: bool = True) -> NpNDArrayFp64:
        if with_resize:
            try:
                return self.label_to_plot_prepped_coordinates[label_to_plot]
            except KeyError:
                result = self.video_for_computation().prepare_coordinates_for_plotting(self[label_to_plot])
                self.label_to_plot_prepped_coordinates[label_to_plot] = result
                return result
        try:
            return self.label_to_plot_without_resized_coordinates[label_to_plot]
        except KeyError:
            result = self.video_for_computation().prepare_coordinates_for_plotting(
                self[label_to_plot], with_resize=False
            )
            self.label_to_plot_without_resized_coordinates[label_to_plot] = result
            return result

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def plot_boolean_index(self, boolean_index: Np1DArrayBool, ax: Axes, label_to_plot: Optional[str] = None) -> None:
        coordinates_for_plot = self.coordinates_for_plot(label_to_plot or self.object_tracking_label_for_kinematics)
        ax_plot_coordinate_with_boolean_index(
            ax, boolean_index, coordinates_for_plot, plot_line=True, plot_non_confinement=False
        )
        plt.legend(**BOTTOM_LEGEND_KWARGS)

    def plot_skeleton_in_frame(
        self, frame_index: int, ax: Axes, labels_to_exclude: Optional[Iterable[str]] = None, revert_crop: bool = False
    ) -> None:
        for label in self.all_tracked_labels:
            if labels_to_exclude and label in labels_to_exclude:
                continue

            coordinates = self.coordinates_for_plot(label, with_resize=False)[frame_index]
            if revert_crop:
                coordinates = coordinates - np.array([self.x_axis_crop_end_point, self.y_axis_crop_end_point])

            ax.scatter(*coordinates.T, c=self.label_to_plot_color[label], label=label)

        ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")

    def confinement_index_defaultdict(self) -> defaultdict[str, Np1DArrayBool]:
        return defaultdict(lambda: np.zeros(len(self.augmented), dtype=bool))

    def confinement_sequence_defaultdict(self, more_than_254: bool = False) -> defaultdict[str, ConfinementSequence]:
        data_type = np.uint16 if more_than_254 else np.uint8
        return defaultdict(lambda: np.zeros(len(self.augmented), dtype=data_type))

    def coordinate_sequence_defaultdict(self) -> defaultdict[str, ConfinementSequence]:
        return defaultdict(lambda: np.zeros((len(self.augmented), 2), dtype=np.float64))

    def flush_reads(self) -> None:
        try:
            del self.raw_df
            del self.augmented
        except AttributeError as e:
            logger.error(str(e))

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

    def _isolate_coordinates(self, key: Iterable[str] | str) -> NpNDArrayFp64:
        return self.isolate_coordinates_from_native_df(self.df, key)

    def _cache_augmented(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.cached_augmented_df_path, **TO_PARQUET_KWARGS)


ReaderCLS = type[BaseReader]
