import os
import re
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from functools import cached_property
from logging import getLogger
from operator import attrgetter
from pathlib import Path, PurePath
from typing import Any, ClassVar, Iterable, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pydantic import DirectoryPath, Field, FilePath

from bikipy._base_class import BikipyBase, VideoMetaDataMixin
from bikipy.feature.motion import Motion, motion_2d_multi_indexer
from bikipy.reader.deeplabcut import DeepLabCutReader
from bikipy.utils.store import RangeDict
from bikipy.utils.typing import NDArray

logger = getLogger(__name__)


LABEL_VS_DATA_READER = {"deeplabcut": DeepLabCutReader}


class Behaviour(BikipyBase, VideoMetaDataMixin):
    metric_resolution: Union[NDArray, float, None] = None
    manual_units_per_pixel: Optional[float] = None
    data_import_kwargs: Optional[dict] = None
    data_format_label: Literal["deeplabcut"] = "deeplabcut"

    _live: bool = False

    @property
    def units_per_pixel(self):
        return self.manual_units_per_pixel or self.computed_units_per_pixel

    @cached_property
    def computed_units_per_pixel(self):
        if not np.any(self.metric_resolution):
            msg = "metric_resolution attribute needs to be defined to compute units_per_pixel"
            raise AttributeError(msg)
        if isinstance(self.metric_resolution, (float, int)):
            return self.metric_resolution / np.mean(self.recording_resolution)
        else:
            return np.array(self.metric_resolution) / self.recording_resolution


class BaseExperiment(Behaviour):
    point_label_for_motion_features: str
    trial_class: Any = None
    trial_id_vs_trial_class: Optional[dict] = None
    trial_id_vs_keyword_arguments: Optional[dict] = None
    trial_id_range_vs_keyword_arguments: Optional[RangeDict] = None
    common_trial_keyword_arguments: dict = Field(default_factory=dict)
    inspection_figure_save: Union[DirectoryPath, bool] = False

    # Computational settings
    _enable_process_pooling = True

    # Formatting settings
    _deeplabcut_trial_id_finder = re.compile(r"\d+")

    def __getitem__(self, item: int):
        return self.trial_id_vs_trial_object[item]

    def trial_keyword_arguments(self, trial_id: int) -> dict:
        """
        Function useful for customizing initiation parameters for trial objects

        :param trial_id: Respective trial ID
        :return:
        """
        result = {
            **self.common_trial_keyword_arguments,
            "int_id": trial_id,
            "metric_resolution": self.metric_resolution,
            "point_label_for_motion_features": self.point_label_for_motion_features,
            "data_format_label": self.data_format_label,
        }

        if self.trial_id_vs_keyword_arguments:
            result.update(self.trial_id_vs_keyword_arguments[trial_id])
        if self.trial_id_range_vs_keyword_arguments:
            result.update(self.trial_id_range_vs_keyword_arguments[trial_id])
        if self.data_import_kwargs:
            result["data_import_kwargs"] = self.data_import_kwargs

        assert result["coordinate_data_path"]

        return result

    @cached_property
    def trial_objects(self) -> list:
        if self.trial_class:
            return [
                self.trial_class(**self.trial_keyword_arguments(trial_id))
                for trial_id in self._trial_id_key_view
            ]
        elif self.trial_id_vs_trial_class:
            return [
                trial_class(**self.trial_keyword_arguments(trial_id))
                for trial_id, trial_class in self.trial_id_vs_trial_class.items()
            ]
        else:
            msg = (
                "Either trial_class or trial_id_vs_trial_class have to be "
                "exclusively defined"
            )
            raise AttributeError(msg)

    @cached_property
    def _trial_class_vs_trial_ids(self) -> dict:
        if not self.trial_id_vs_trial_class:
            msg = (
                "This experiment object has no trial_id_vs_trial_class, "
                "this attribute is reserved for experiments with "
                "several trial classes"
            )
            raise AttributeError(msg)

        result = {}
        for trial_id, trial_class in self.trial_id_vs_trial_class.items():
            if trial_class in result:
                result[trial_class].append(trial_id)
            else:
                result[trial_class] = [trial_id]

        return dict(
            sorted(result.items(), key=lambda trial_c: trial_c[0].trial_sequence_index)
        )

    @cached_property
    def _trial_class_vs_trial_objects(self):
        return {
            trial_class: [
                self.trial_id_vs_trial_object[trial_id] for trial_id in trial_ids
            ]
            for trial_class, trial_ids in self._trial_class_vs_trial_ids.items()
        }

    @cached_property
    def _trial_classes(self) -> tuple:
        return tuple(
            sorted(self._trial_class_vs_trial_ids, key=lambda x: x.trial_sequence_index)
        )

    @cached_property
    def trial_class_name_vs_trial_ids(self):
        return {
            trial_class.__name__: trial_ids
            for trial_class, trial_ids in self._trial_class_vs_trial_ids.items()
        }

    @cached_property
    def trial_id_vs_trial_object(self) -> dict:
        return {trial.int_id: trial for trial in self.trial_objects}

    @cached_property
    def trial_class_name_vs_trial_objects(self):
        result = {}
        for trial_class_name, trial_ids in self._trial_class_vs_trial_ids.items():
            result[trial_class_name] = [
                self.trial_id_vs_trial_object[trial_id] for trial_id in trial_ids
            ]
        return result

    @cached_property
    def animal_id_vs_trial_objects(self) -> dict:
        result = {}
        for trial in self.trial_objects:
            if trial.animal_id in result:
                result[trial.animal_id].append(trial)
            else:
                result[trial.animal_id] = [trial]
        for trials in result.values():
            trials.sort(key=lambda t: t.int_id)
        return dict(sorted(result.items()))

    @cached_property
    def trial_classes_without_features(self):
        return np.where(
            [
                trial_object.trial_has_feature_frame
                for trial_object in self.animal_id_vs_trial_objects.values()
            ]
        )[0]

    @property
    def inspect(self):
        if isinstance(self.inspection_figure_save, bool):
            return self.inspection_figure_save
        elif isinstance(self.inspection_figure_save, PurePath):
            if not self.inspection_figure_save.exists():
                os.makedirs(self.inspection_figure_save)
            return self.inspection_figure_save / f"experiment_{self.timestamp}_inspect"
        else:
            raise AttributeError()

    @property
    def _trial_id_key_view(self):
        if self.trial_id_vs_keyword_arguments:
            return self.trial_id_vs_keyword_arguments.keys()
        if self.trial_id_range_vs_keyword_arguments:
            return self.trial_id_range_vs_keyword_arguments.keys()

    @property
    def trial_id_tuple(self) -> tuple:
        return tuple(self._trial_id_key_view)

    @cached_property
    def number_of_trials(self):
        return len(self._trial_id_key_view)

    # DataFrame methods

    @cached_property
    def animal_summary_frame(self) -> pd.DataFrame:
        df = self.feature_summary_frame
        df.columns = self._feature_frame_columns(
            levels=self.animal_id_indexed_motion_summary_frame.columns.nlevels
        )
        return df.join(self.animal_id_indexed_motion_summary_frame, how="inner")

    @cached_property
    def feature_summary_frame(self) -> pd.DataFrame:
        data_dict = {}
        if self._enable_process_pooling:
            with ProcessPoolExecutor() as executor:
                for animal_id, trial_objects in self.animal_id_vs_trial_objects.items():
                    trial_objects = [
                        trial_object
                        for trial_object in copy(trial_objects)
                        if trial_object.trial_has_feature_frame
                    ]
                    data_dict[animal_id] = sum(
                        list(
                            executor.map(
                                attrgetter("feature_summary_row"), trial_objects
                            )
                        ),
                        [],
                    )
        else:
            for animal_id, trial_objects in self.animal_id_vs_trial_objects.items():
                data_dict[animal_id] = sum(
                    (
                        trial_object.feature_summary_row
                        for trial_object in trial_objects
                        if trial_object.trial_has_feature_frame
                    ),
                    [],
                )

        result = pd.DataFrame.from_dict(
            data_dict, orient="index", columns=self._feature_frame_columns()
        )
        result.index.set_names("Animal ID")

        return result

    def _feature_frame_columns(self, levels: Optional[int] = None) -> pd.MultiIndex:
        if self.trial_class:
            columns = self.trial_class.feature_summary_column
        elif self.trial_id_vs_trial_class:
            columns = sum(
                (
                    trial_object.feature_summary_column
                    for trial_object in self._trial_classes
                    if trial_object.trial_has_feature_frame
                ),
                [],
            )
        else:
            raise ValueError

        if levels:
            column_array = np.array(columns)
            if levels > (native_nlevel := column_array.shape[1]):
                return pd.MultiIndex.from_arrays(
                    np.concatenate(
                        (
                            column_array,
                            [["" for _ in range(levels - native_nlevel)]]
                            * len(column_array),
                        ),
                        axis=1,
                    )
                )
            elif levels < native_nlevel:
                msg = (
                    "Can not reduce the number of levels that are natively defined"
                    "in index"
                )
                raise ValueError(msg)

        return pd.MultiIndex.from_tuples(columns)

    @cached_property
    def motion_summary_frame(self) -> pd.DataFrame:
        if self._enable_process_pooling:
            with ProcessPoolExecutor() as executor:
                rows = executor.map(attrgetter("motion_features"), self.trial_objects)
        else:
            rows = (trial_object.motion_features for trial_object in self.trial_objects)

        result = pd.DataFrame(
            rows,
            columns=self._motion_summary_columns,
            index=self._frame_index,
        )

        return pd.concat((self._trial_id_vs_animal_id_frame, result), axis=1)

    @cached_property
    def animal_id_indexed_motion_summary_frame(self) -> pd.DataFrame:
        motion = self.motion_summary_frame.reset_index().sort_values(
            by=[("All", "Animal ID"), ("Test ID", "")]
        )

        series = {}
        for i, animal_id_df in motion.copy().groupby(("All", "Animal ID")):
            new_series = animal_id_df.unstack().unstack(2)
            new_series.columns = self._class_labels

            new_series = (
                new_series.stack().reorder_levels((2, 0, 1)).sort_index(level=0)
            )

            series[i] = new_series.drop(
                [
                    new_series.index[index]
                    for index in np.where(
                        new_series.index.get_level_values(level=2) == "Animal ID"
                    )[0]
                ],
            )
        return pd.DataFrame.from_dict(series, orient="index")

    @staticmethod
    def _feature_2d_multi_indexer(feature: str, category) -> tuple:
        return tuple([(feature, category) for category in category])

    @cached_property
    def _frame_index(self) -> pd.Series:
        return pd.Series(self._trial_id_key_view, name="Test ID", dtype=np.int16)

    @cached_property
    def _trial_class_name_vs_frame_index(self) -> dict:
        return {
            class_name: pd.Series(trial_ids, name="Test ID", dtype=np.int16)
            for class_name, trial_ids in self.trial_class_name_vs_trial_ids.items()
        }

    @cached_property
    def _trial_id_vs_animal_id_frame(self) -> pd.DataFrame:
        try:
            return pd.DataFrame(
                (trial.animal_id for trial in self.trial_objects),
                columns=(("All", "Animal ID"),),
                index=self._frame_index,
            )
        except AttributeError as e:
            msg = (
                "animal_id needs to be defined for each trial instance to use "
                "this export method"
            )
            raise AttributeError(msg) from e

    @cached_property
    def _class_labels(self):
        return tuple(trial_class.trial_label for trial_class in self._trial_classes)

    @property
    def _motion_summary_columns(self) -> list:
        return motion_2d_multi_indexer("All")


class BaseTrial(Behaviour):
    coordinate_data_path: FilePath = Field(
        description="Path to file storing coordinate data"
    )
    animal_id: int = Field(None, description="The ID of the animal in the trial")
    point_label_for_motion_features: Optional[str] = Field(
        description="Label of the node that will be used to track general animal movement"
    )
    rigid_nodes_freezing: Optional[Sequence[Union[str, int]]] = Field(
        None,
        description="Nodes that should remain during freeze/immobility, most often due to fear.",
    )
    inspection_figure_save: Union[DirectoryPath, bool] = Field(
        False, description="Path to save figures for inspection of results"
    )
    inspect_image: Optional[FilePath] = Field(
        None, description="Image used for inspection"
    )
    # Variables for trials with zones, see doc for more info.
    perimeters: Optional[Sequence] = None
    trial_start_perimeter: Optional[str] = None

    category: ClassVar[Optional[str]] = "trial"

    trial_sequence_index: ClassVar[Optional[int]] = None
    trial_label: ClassVar[str] = ""

    second_tolerance: ClassVar[float] = 0.35

    trial_has_feature_frame: ClassVar[bool] = False
    feature_summary_column: ClassVar[list] = []

    @property
    def feature_summary_row(self) -> list:
        raise NotImplementedError

    @staticmethod
    def _get_reader(coordinate_data_format):
        try:
            return LABEL_VS_DATA_READER[coordinate_data_format]
        except KeyError as e:
            msg = (
                f"{coordinate_data_format} as a format for data ingestion has "
                f"no implementation. Choose from: {LABEL_VS_DATA_READER.keys()}"
            )
            raise NotImplemented(msg) from e

    @cached_property
    def reader(self):
        return self._get_reader(self.data_format_label)(
            df_path=self.coordinate_data_path,
            **self._video_metadata_dict_manual_format,
            **self.data_import_kwargs,
        )

    @property
    def coordinates_per_frame(self) -> np.ndarray:
        return self.reader[self.point_label_for_motion_features]

    @cached_property
    def number_of_frames(self) -> int:
        return len(self.coordinates_per_frame)

    @cached_property
    def experiment_seconds(self) -> int:
        return self.coordinates_per_frame.shape[0] / self.fps

    @property
    def inspect_image_path(self) -> Union[DirectoryPath, bool]:
        if isinstance(self.inspection_figure_save, str) or isinstance(
            self.inspection_figure_save, PurePath
        ):
            return Path(self.inspection_figure_save) / self.best_id
        return self.inspection_figure_save  # return the bool in any case

    @cached_property
    def recording_center_pixel(self) -> np.ndarray:
        return self.recording_resolution / 2.0

    @cached_property
    def motion(self) -> Motion:
        return Motion(self.coordinates_per_frame, self.units_per_pixel, self.fps)

    # Perimeter

    def detect_confined_perimeter(self, coordinate: np.ndarray) -> np.ndarray:
        """
        This function is used to determine current location of subject.

        :param coordinate:
        :return:
        """

        coordinate = np.expand_dims(coordinate, 0)
        for label, perimeter in self.int_id_vs_perimeter.items():
            if perimeter.coordinate_confinement_boolean_index(coordinate):
                logger.info(f"Location: {label}, {coordinate}")
                return label
        logger.debug(f"Location could not be determined, {coordinate}")

    @cached_property
    def int_id_vs_perimeter(self) -> dict:
        self._validate_perimeters_object()
        return {
            i: perimeter
            for i, perimeter in enumerate(self.perimeters.values(), start=1)
        }

    def _validate_perimeters_object(self) -> None:
        if not self.perimeters:
            msg = "perimeters is not defined as an object variable, which is required for int_id_vs_perimeter"
            raise AttributeError(msg)

    @cached_property
    def _perimeter_label_vs_int_id(self) -> dict:
        self._validate_perimeters_object()
        return {label: i for i, label in enumerate(self.perimeters, start=1)}

    @cached_property
    def _int_id_vs_perimeter_label(self) -> dict:
        self._validate_perimeters_object()
        return {i: label for i, label in enumerate(self.perimeters, start=1)}

    @property
    def _start_int_id(self) -> int:
        return self._perimeter_label_vs_int_id[self.trial_start_perimeter]

    def _perimeter_label_sequence_to_int_id(self, label_sequence: Iterable) -> tuple:
        return tuple(self._perimeter_label_vs_int_id[label] for label in label_sequence)

    # Miscellaneous

    @cached_property
    def _uint_zeros_based_on_frame_length(self) -> np.ndarray:
        return np.zeros(self.number_of_frames, dtype=np.uint8)

    @cached_property
    def _frame_tolerance(self) -> int:
        return round(self.second_tolerance * self.fps)

    @property
    def motion_features(self) -> list:
        return self.motion.to_list
