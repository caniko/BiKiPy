import os
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from functools import cached_property
from logging import getLogger
from numbers import Integral
from operator import attrgetter
from pathlib import Path
from typing import Any, ClassVar, Hashable, Iterable, Literal, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydantic
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydantic import DirectoryPath, Field, FilePath, validator
from pydantic_numpy import NDArray

from bikipy import ENABLE_PROCESS_POOLING
from bikipy.core.base_class import BikipyBaseHashable
from bikipy.core.mixin import VideoMetadataMixin
from bikipy.core.typing import NDArrayFp64, NDArrayInt16
from bikipy.feature.motion import Motion, motion_multi_indexer
from bikipy.reader.deeplabcut import DeepLabCutReader
from bikipy.utils.misc import rise_to_n_levels
from bikipy.utils.ranged_dict import RangeDict
from bikipy.utils.render_video import VideoWriter

logger = getLogger(__name__)

LABEL_to_DATA_READER = {"deeplabcut": DeepLabCutReader}


class Behaviour(BikipyBaseHashable, VideoMetadataMixin):
    center: Optional[NDArrayInt16] = None

    data_import_kwargs: Optional[dict] = None
    data_format_label: Literal["deeplabcut"] = "deeplabcut"

    _live: ClassVar[bool] = False

    @cached_property
    def recording_center_pixel(self) -> NDArrayInt16:
        return self.recording_resolution / 2.0

    @cached_property
    def center_translation(self):
        return self.recording_center_pixel - self.center if self.center is not None else None


class BaseExperiment(Behaviour):
    manual_trial_ids: Optional[tuple] = None
    trial_id_to_trial_class_name: Optional[dict] = None
    trial_id_to_keyword_arguments: Optional[dict] = None
    trial_id_range_to_keyword_arguments: Optional[RangeDict] = None
    common_trial_keyword_arguments: Optional[dict] = None
    stage: Optional[str] = Field(
        description="Experiment stage label, if experiment object is in a sequence of experiment objects"
    )
    inspection_dir: Optional[DirectoryPath] = Field(description="Path to save figures for inspection of results")

    trial_classes: ClassVar[tuple[Any]] = Field(..., description="Trial classes designed for this experiment class")

    @validator("trial_id_to_trial_class_name")
    def sort_trial_id_to_trial_class_name_ascending(cls, value):
        return dict(sorted(value.items()))

    @validator("trial_id_to_trial_class_name")
    def sort_trial_id_to_keyword_arguments_ascending(cls, value):
        return dict(sorted(value.items()))

    def __getitem__(self, item: int):
        return self.trial_id_to_trial_object[item]

    @classmethod
    @property
    def is_trial_sequence(cls):
        return len(cls.trial_classes) != 1

    @classmethod
    @property
    def trial_class_names(cls):
        return tuple(trial_class.__name__ for trial_class in cls.trial_classes)

    @classmethod
    @property
    def trial_class(cls):
        if not cls.is_trial_sequence:
            msg = f"{cls.__name__}: trial_class attribute can only be utilized when there is only one Trial class"
            raise AttributeError(msg)
        return cls.trial_classes[0]

    @classmethod
    @property
    def stage_index_to_trial_class(cls):
        if not cls.is_trial_sequence:
            msg = f"{cls.__name__}: stage_index_to_trial_class is undefined in non-sequential experiment classes"
            raise AttributeError(msg)

        try:
            return {trial_class.experiment_sequence_index: trial_class for trial_class in cls.trial_classes}
        except AttributeError:
            msg = "experiment_sequence_index must be defined for each trial class when working with a sequence of trial classes"
            raise AttributeError(msg)

    @classmethod
    @property
    def trial_class_name_to_trial_class(cls):
        if not cls.is_trial_sequence:
            msg = (
                f"{cls.__name__}: trial_class_name_to_trial_class attribute can only be utilized when "
                f"there are many Trial classes"
            )
            raise AttributeError(msg)

        try:
            return {trial_class.__name__: trial_class for trial_class in cls.trial_classes}
        except AttributeError:
            msg = "experiment_sequence_index must be defined for each trial class when working with a sequence of trial classes"
            raise AttributeError(msg)

    def trial_keyword_arguments(self, trial_id: Hashable) -> dict:
        """
        Function useful for customizing initiation parameters for trial objects
        """
        result = {"data_format_label": self.data_format_label}

        if self.common_trial_keyword_arguments:
            result.update(self.common_trial_keyword_arguments)
        if self.trial_id_to_keyword_arguments:
            result.update(self.trial_id_to_keyword_arguments[trial_id])
        if self.trial_id_range_to_keyword_arguments:
            if not isinstance(trial_id, int):
                msg = "Trial IDs must be integers when trial_id_range_to_keyword_arguments is used"
                raise AttributeError(msg)
            result.update(self.trial_id_range_to_keyword_arguments[trial_id])

        assert result["coordinate_data_path"]

        if hasattr(self, "metric_resolution"):
            result["metric_resolution"] = self.metric_resolution
        if self.data_import_kwargs:
            result["data_import_kwargs"] = self.data_import_kwargs

        if "animal_id" not in result:
            result["animal_id"] = trial_id

        if self.inspection_dir:
            if "stage" in result:
                result["inspection_dir"] = self.inspection_dir / result["stage"]
                if not result["inspection_dir"].exists():
                    os.mkdir(result["inspection_dir"])
            else:
                result["inspection_dir"] = self.inspection_dir

        return result

    @cached_property
    def trial_objects(self) -> list:
        result, bad_trial_ids_to_error_msg = [], {}
        for trial_id in self.trial_ids:
            trial_class = (
                self.trial_class_name_to_trial_class[self.trial_id_to_trial_class_name[trial_id]]
                if self.is_trial_sequence
                else self.trial_class
            )
            try:
                result.append(trial_class(**self.trial_keyword_arguments(trial_id)))
            except pydantic.error_wrappers.ValidationError as e:
                bad_trial_ids_to_error_msg[trial_id] = str(e)
                continue
        if bad_trial_ids_to_error_msg:
            msg = f"Some trial IDs yielded pydantic validation errors:"
            for trial_id, msg in bad_trial_ids_to_error_msg.items():
                msg += f"\n{trial_id}:\n{msg}\n"
            raise ValueError(msg)
        return result

    @cached_property
    def trial_class_name_to_trial_ids(self):
        return {trial_class.__name__: trial_ids for trial_class, trial_ids in self._trial_class_vstrial_ids.items()}

    @cached_property
    def trial_id_to_trial_object(self) -> dict:
        return {trial.int_id: trial for trial in self.trial_objects}

    @cached_property
    def trial_class_name_to_trial_objects(self):
        result = {}
        for trial_class_name, trial_ids in self._trial_class_vstrial_ids.items():
            result[trial_class_name] = [self.trial_id_to_trial_object[trial_id] for trial_id in trial_ids]
        return result

    @cached_property
    def animal_id_to_trial_objects(self) -> dict:
        result = {}
        for trial in self.trial_objects:
            if trial.animal_id in result:
                result[trial.animal_id].append(trial)
            else:
                result[trial.animal_id] = [trial]
        for trials in result.values():
            trials.sort(key=lambda t: t.best_id)
        return dict(sorted(result.items()))

    @cached_property
    def animal_id_vstrial_ids(self) -> dict:
        return {
            animal_id: (trial_object.int_id for trial_object in trial_objects)
            for animal_id, trial_objects in self.animal_id_to_trial_objects.items()
        }

    @cached_property
    def trial_id_to_animal_id(self) -> dict:
        result = {}
        for animal_id, trial_objects in self.animal_id_to_trial_objects.items():
            for trial_object in trial_objects:
                result[trial_object.int_id] = animal_id
        return dict(sorted(result.items()))

    @cached_property
    def stage_index_to_trial_objects(self) -> dict:
        result = {}
        for trial in self.trial_objects:
            if trial.stage in result:
                result[trial.stage].append(trial)
            else:
                result[trial.stage] = [trial]
        return result

    @cached_property
    def trial_ids(self) -> NDArray:
        if self.manual_trial_ids:
            result = self.manual_trial_ids
        elif self.trial_class:
            result = tuple(self.trial_id_to_keyword_arguments)
        elif self.trial_id_to_trial_class_name:
            result = tuple(self.trial_id_to_trial_class_name)
        else:
            self._neither_singular_trial_class_or_trial_id_to_trial_class_name()

        first_trial_id = result[0]
        first_type = type(first_trial_id)
        if not all(isinstance(trial_id, first_type) for trial_id in result):
            msg = "The Trial IDs must have the same type"
            raise AttributeError(msg)

        # TODO: Infer dtype

        return result

    @cached_property
    def number_of_trials(self):
        return len(self.trial_ids)

    # DataFrame methods

    @cached_property
    def animal_id_indexed_feature_frame(self) -> pd.DataFrame:
        dataset = (
            self.animal_id_indexed_experiment_specific_feature_frame,
            self.animal_id_indexed_motion_summary_frame,
        )
        return pd.concat(
            dataset,
            axis=1,
            keys=["Stage"] if self.stage else None,
            # Prepend experiment stage to column MultiIndex:
            # https://stackoverflow.com/a/42094658/9793651
            names=self.column_multi_index_names,
        )

    @cached_property
    def animal_id_indexed_experiment_specific_feature_frame(self) -> pd.DataFrame:
        assert self._at_least_one_trial_class_has_features

        data_dict = {}
        if ENABLE_PROCESS_POOLING:
            with ProcessPoolExecutor() as executor:
                for animal_id, trial_objects in self.animal_id_to_trial_objects.items():
                    trial_objects = [
                        trial_object for trial_object in copy(trial_objects) if trial_object.trial_has_defined_features
                    ]
                    data_dict[animal_id] = sum(
                        list(executor.map(attrgetter("feature_summary_row"), trial_objects)),
                        [],
                    )
        else:
            for animal_id, trial_objects in self.animal_id_to_trial_objects.items():
                data_dict[animal_id] = sum(
                    (
                        trial_object.feature_summary_row
                        for trial_object in trial_objects
                        if trial_object.trial_has_defined_features
                    ),
                    [],
                )

        result = pd.DataFrame.from_dict(data_dict, orient="index", columns=self._feature_frame_columns())
        result.index.name = "Animal ID"
        result.columns.names = ["Feature", "Location_Category"]

        return result

    @cached_property
    def animal_id_indexed_motion_summary_frame(self) -> pd.DataFrame:
        return (
            pd.merge(
                self._trial_id_animal_id(self.motion_summary_frame.columns.nlevels),
                self.motion_summary_frame,
                on="Trial ID",
            )
            .drop("Trial ID", axis=1)
            .set_index("Animal ID")
            .sort_index()
        )

    @cached_property
    def motion_summary_frame(self) -> pd.DataFrame:
        if ENABLE_PROCESS_POOLING:
            with ProcessPoolExecutor() as executor:
                rows = executor.map(attrgetter("motion_features"), self.trial_objects)
        else:
            rows = [trial_object.motion_features for trial_object in self.trial_objects]

        return pd.DataFrame(
            rows,
            columns=self._motion_summary_column_index,
            index=self._trial_id_series,
        )

    @cached_property
    def motion_summary_columns(self) -> list:
        return motion_multi_indexer("All", self._pandas_multi_index_level)

    @cached_property
    def column_multi_index_names(self):
        return ["Feature", "Location_Category"] if self.stage is None else ["Stage", "Feature", "Location_Category"]

    # Private methods

    @cached_property
    def _at_least_one_trial_class_has_features(self):
        return any(not trial_class.feature_summary_column.empty for trial_class in self.trial_classes)

    def _feature_frame_columns(self, levels: Optional[int] = None) -> pd.MultiIndex:
        if not self.is_trial_sequence:
            columns = self.trial_class.feature_summary_column
        elif self.trial_id_to_trial_class_name:
            columns = sum(
                (
                    list(trial_object.feature_summary_column)
                    for trial_object in self.trial_classes
                    if trial_object.trial_has_defined_features
                ),
                [],
            )
        else:
            self._neither_singular_trial_class_or_trial_id_to_trial_class_name()

        if levels:
            columns = rise_to_n_levels(columns, levels)

        return pd.MultiIndex.from_tuples(columns)

    @staticmethod
    def _feature_2d_multi_indexer(feature: str, category) -> tuple:
        return tuple([(feature, category) for category in category])

    @cached_property
    def _trial_id_series(self) -> pd.Series:
        return pd.Series(self.trial_ids, name="Trial ID")

    @cached_property
    def _trial_id_indexed_animal_ids(self) -> pd.Series:
        return pd.Series(
            self.trial_id_to_animal_id.values(),
            index=self._trial_id_series,  # derived from self.trial_id_to_animal_id
            name="Animal ID",
        ).sort_index()

    def _trial_id_animal_id(self, levels: Optional[int] = None) -> pd.DataFrame:
        result = self._trial_id_indexed_animal_ids.reset_index()
        if levels:
            columns = rise_to_n_levels(result.columns, levels)
            result.columns = columns
        return result

    @cached_property
    def _trial_class_vstrial_ids(self) -> dict:
        if not self.trial_id_to_trial_class_name:
            msg = (
                "This experiment object has no trial_id_to_trial_class_name, "
                "this attribute is reserved for experiments with "
                "several trial classes"
            )
            raise AttributeError(msg)

        result = {}
        for trial_id, trial_class in self.trial_id_to_trial_class_name.items():
            if trial_class in result:
                result[trial_class].append(trial_id)
            else:
                result[trial_class] = [trial_id]

        return dict(sorted(result.items(), key=lambda trial_c: trial_c[0].experiment_sequence_index))

    @cached_property
    def _trial_class_to_trial_objects(self):
        return {
            trial_class: [self.trial_id_to_trial_object[trial_id] for trial_id in trial_ids]
            for trial_class, trial_ids in self._trial_class_vstrial_ids.items()
        }

    @cached_property
    def _class_labels(self):
        return tuple(trial_class.trial_label for trial_class in self.trial_classes)

    def _difference_warning(self, other, attribute: str):
        if (self_attr := getattr(self, attribute)) == (other_attr := getattr(other, attribute)):
            return
        logger.warning(f"Joining experiments {self.best_id} & {other.best_id}: " f"{self_attr} != {other_attr}")

    @cached_property
    def _trial_id_column_index(self):
        return (
            "Trial ID",
            *["" for _ in range(self._motion_summary_column_depth - 1)],
        )

    @cached_property
    def _animal_id_column_index(self):
        return (
            "Animal ID",
            *["" for _ in range(self._motion_summary_column_depth - 1)],
        )

    @cached_property
    def _animal_id_key_view(self):
        return self.animal_id_to_trial_objects.keys()

    @cached_property
    def _motion_summary_column_index(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_tuples(self.motion_summary_columns, names=self.column_multi_index_names)

    @cached_property
    def _motion_summary_column_depth(self):
        motion_summary_column_index_list = list(self._motion_summary_column_index)
        result = len(motion_summary_column_index_list[0])
        assert all(result == len(column) for column in motion_summary_column_index_list[1:])
        return result

    @cached_property
    def _pandas_multi_index_level(self) -> int:
        return len(self.column_multi_index_names)

    def _make_categorical_inspection_dir(self, trial_root_dir: Path):
        pass

    def _neither_singular_trial_class_or_trial_id_to_trial_class_name(self):
        msg = (
            "Either trial_class has to be singularly defined, "
            "or trial_id_to_trial_class_name have to be exclusively defined"
        )
        raise AttributeError(msg)


class BaseTrial(Behaviour):
    coordinate_data_path: FilePath = Field(..., description="Path to file storing coordinate data")
    animal_id: int = Field(..., description="The ID of the animal in the trial")
    object_tracking_label_for_kinematics: Optional[str] = Field(
        ..., description="Label of the node that will be used to track general animal movement"
    )
    rigid_nodes_freezing: Optional[Sequence[str | int]] = Field(
        description="Nodes that should remain during freeze/immobility, most often due to fear.",
    )
    stage: Optional[str] = Field(description="The semantic stage of the experiment")
    inspection_dir: Optional[DirectoryPath] = Field(description="Path to save figures for inspection of results")
    inspect_image: Optional[FilePath] = Field(
        description="Image to use as background in the plots for visualising the analysis data",
    )
    # Variables for trials with zones, see doc for more info.
    perimeters: Optional[Sequence] = None
    trial_start_perimeter: Optional[str] = None

    # Class variables
    category: ClassVar[Optional[str]] = "trial"

    experiment_sequence_index: ClassVar[Optional[int]] = None
    trial_label: ClassVar[Optional[str]] = None

    second_tolerance: ClassVar[float] = 0.15

    trial_has_video_space_for_analysis: ClassVar[bool] = False

    @classmethod
    @property
    def feature_headers(cls):
        return []

    @classmethod
    @property
    def feature_summary_column(cls):
        if not cls.trial_label:
            return cls.feature_headers
        return pd.MultiIndex.from_product([[cls.trial_label], cls.feature_headers])

    @classmethod
    @property
    def trial_has_defined_features(cls) -> bool:
        return bool(cls.feature_headers)

    @property
    def motion_features(self) -> list:
        return self.motion.to_list

    @cached_property
    def reader(self):
        try:
            reader_init_func = LABEL_to_DATA_READER[self.data_format_label]
        except KeyError as e:
            msg = (
                f"{self.data_format_label} as a format for data ingestion has "
                f"no implementation. Choose from: {LABEL_to_DATA_READER.keys()}"
            )
            raise NotImplemented(msg) from e

        return reader_init_func(
            df_path=self.coordinate_data_path,
            **self._reader_init_kwargs,
        )

    @property
    def framewise_confined_coordinates(self) -> NDArrayFp64:
        return self.reader[self.object_tracking_label_for_kinematics]

    @cached_property
    def number_of_frames(self) -> int:
        return len(self.framewise_confined_coordinates)

    @cached_property
    def experiment_seconds(self) -> int:
        return self.framewise_confined_coordinates.shape[0] / self.fps

    @cached_property
    def motion(self) -> Motion:
        return Motion(
            coordinate_sequence=self.framewise_confined_coordinates,
            meters_per_pixel=self.meters_per_pixel,
            fps=self.fps,
        )

    # PolygonPerimeter

    def detect_confined_perimeter(self, coordinate: NDArrayFp64) -> NDArrayFp64:
        """
        This function is used to determine current location of subject.

        :param coordinate:
        :return:
        """

        coordinate = np.expand_dims(coordinate, 0)
        for label, perimeter in self._int_id_to_perimeter.items():
            if perimeter.coordinate_confinement_boolean_index(coordinate):
                logger.info(f"Location: {label}, {coordinate}")
                return label
        logger.debug(f"Location could not be determined, {coordinate}")

    def render_analytical_video(self):
        if not self.video_path:
            msg = "video_path needs to be defined to render analytical video"
            raise AttributeError(msg)

        cap = cv2.VideoCapture(str(self.video_path))
        writer = VideoWriter(
            filename=self.video_path.with_name(f"{self.video_path.stem}_analysis.mp4"),
            fps=round(self.fps * 0.75),
        )
        success, frame = cap.read()
        assert success

        i = 0
        frames = []
        if ENABLE_PROCESS_POOLING:
            with ProcessPoolExecutor() as executor:
                while success:
                    frames.append(executor.submit(self._process_frame, frame, i))
                    success, frame = cap.read()
                    i += 1

                for frame in frames:
                    writer.add(frame.result())
        else:
            logger.debug("Process pooling is disabled, will create video with one core")
            while success:
                frame = self._process_frame(frame, i)
                writer.add(frame)
                success, frame = cap.read()
                i += 1

        writer.close()

    def _process_frame(self, frame: NDArrayFp64, frame_index: int) -> NDArrayFp64:
        if self.trial_has_video_space_for_analysis:
            return cv2.hconcat(
                (
                    self._overlay_video_frame(frame, frame_index),
                    self._create_analysis_frame(frame_index),
                )
            )
        return self._overlay_video_frame(frame, frame_index)

    def _overlay_video_frame(self, frame: NDArrayFp64, frame_index: int) -> NDArrayFp64:
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        ax.imshow(frame)
        ax.scatter(*self.framewise_confined_coordinates[frame_index])
        ax.axis("off")

        canvas.draw()

        return np.frombuffer(canvas.tostring_rgb(), dtype="uint8")

    def _create_analysis_frame(self, frame_index: int) -> NDArrayFp64:
        raise NotImplemented("The analysis space is a work in progress")

    @cached_property
    def _reader_init_kwargs(self):
        return self.data_import_kwargs

    @cached_property
    def _int_id_to_perimeter(self) -> dict:
        self._validate_perimeters_object()
        return {perimeter.int_id: perimeter for perimeter in self.perimeters}

    def _validate_perimeters_object(self) -> None:
        if not self.perimeters:
            msg = "perimeters is not defined as an object variable, " "which is required for _int_id_to_perimeter"
            raise AttributeError(msg)

    @cached_property
    def _perimeter_label_to_int_id(self) -> dict:
        self._validate_perimeters_object()
        return {label: i for i, label in enumerate(self.perimeters, start=1)}

    @cached_property
    def _int_id_to_perimeter_label(self) -> dict:
        self._validate_perimeters_object()
        return {i: label for i, label in enumerate(self.perimeters, start=1)}

    @property
    def _start_int_id(self) -> int:
        return self._perimeter_label_to_int_id[self.trial_start_perimeter]

    def _perimeter_label_sequence_to_int_id(self, label_sequence: Iterable) -> tuple:
        return tuple(self._perimeter_label_to_int_id[label] for label in label_sequence)

    # Miscellaneous

    @cached_property
    def _inspection_image_name(self):
        return f"trial_{self.best_id}.jpg"

    @cached_property
    def _uint_zeros_based_on_frame_length(self) -> NDArrayFp64:
        return np.zeros(self.number_of_frames, dtype=np.uint8)

    @cached_property
    def _frame_tolerance(self) -> int:
        return round(self.second_tolerance * self.fps)
