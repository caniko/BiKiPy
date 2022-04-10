import os
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from functools import cached_property
from logging import getLogger
from operator import attrgetter
from pathlib import Path, PurePath
from typing import Any, ClassVar, Iterable, Literal, Optional, Sequence, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydantic import DirectoryPath, Field, FilePath, validator

from bikipy.core.base_class import BikipyBaseHashable
from bikipy.core.mixin import VideoMetadataMixin
from bikipy.feature.motion import Motion, motion_2d_multi_indexer
from bikipy.reader.deeplabcut import DeepLabCutReader
from bikipy.utils.misc import to_tuple, rise_to_n_levels
from bikipy.utils.render_video import VideoWriter
from bikipy.utils.store import RangeDict

logger = getLogger(__name__)

LABEL_VS_DATA_READER = {"deeplabcut": DeepLabCutReader}


class Behaviour(BikipyBaseHashable, VideoMetadataMixin):
    data_import_kwargs: Optional[dict] = None
    data_format_label: Literal["deeplabcut"] = "deeplabcut"

    _live: ClassVar[bool] = False

    # Computational settings
    enable_process_pooling: ClassVar[bool] = True


class BaseExperiment(Behaviour):
    stage: str
    point_label_for_motion_features: str
    trial_id_vs_trial_class: Optional[dict] = None
    trial_id_vs_keyword_arguments: Optional[dict] = None
    trial_id_range_vs_keyword_arguments: Optional[RangeDict] = None
    common_trial_keyword_arguments: Optional[dict] = None
    inspection_dir: Optional[DirectoryPath] = Field(
        None, description="Path to save figures for inspection of results"
    )

    trial_class: ClassVar[Any] = None

    # Computational settings
    enable_process_pooling: ClassVar[bool] = True

    @validator("trial_id_vs_trial_class")
    def sort_trial_id_vs_trial_class_ascending(cls, value):
        return dict(sorted(value.items()))

    @validator("trial_id_vs_trial_class")
    def sort_trial_id_vs_keyword_arguments_ascending(cls, value):
        return dict(sorted(value.items()))

    def __getitem__(self, item: int):
        return self.trial_id_vs_trial_object[item]

    def trial_keyword_arguments(self, trial_id: int) -> dict:
        """
        Function useful for customizing initiation parameters for trial objects

        :param trial_id: Respective trial ID
        :return:
        """
        result = {
            "int_id": trial_id,
            "point_label_for_motion_features": self.point_label_for_motion_features,
            "data_format_label": self.data_format_label,
        }

        if self.common_trial_keyword_arguments:
            result.update(self.common_trial_keyword_arguments)
        if self.trial_id_vs_keyword_arguments:
            result.update(self.trial_id_vs_keyword_arguments[trial_id])
        if self.trial_id_range_vs_keyword_arguments:
            result.update(self.trial_id_range_vs_keyword_arguments[trial_id])

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
            self._neither_singular_trial_class_or_trial_id_vs_trial_class()

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
    def animal_id_vs_trial_ids(self) -> dict:
        return {
            animal_id: (trial_object.int_id for trial_object in trial_objects)
            for animal_id, trial_objects in self.animal_id_vs_trial_objects.items()
        }

    @cached_property
    def trial_id_vs_animal_id(self) -> dict:
        result = {}
        for animal_id, trial_objects in self.animal_id_vs_trial_objects.items():
            for trial_object in trial_objects:
                result[trial_object.int_id] = animal_id
        return dict(sorted(result.items()))

    @cached_property
    def stage_vs_trial_objects(self) -> dict:
        result = {}
        for trial in self.trial_objects:
            if trial.stage in result:
                result[trial.stage].append(trial)
            else:
                result[trial.stage] = [trial]
        return result

    @cached_property
    def stages(self):
        return tuple(self.stage_vs_trial_objects.keys())

    @property
    def trial_id_tuple(self) -> tuple:
        return tuple(self._trial_id_key_view)

    @cached_property
    def number_of_trials(self):
        return len(self._trial_id_key_view)

    # DataFrame methods

    @cached_property
    def animal_id_indexed_feature_frame(self) -> pd.DataFrame:
        return pd.concat(
            (
                self.animal_id_indexed_experiment_specific_feature_frame,
                self.animal_id_indexed_motion_summary_frame,
            ),
            axis=1,
            # Prepend experiment stage to column MultiIndex:
            # https://stackoverflow.com/a/42094658/9793651
            keys=[self.stage],
            names=["Stage", "Feature", "Category"],
        )

    @cached_property
    def animal_id_indexed_experiment_specific_feature_frame(self) -> pd.DataFrame:
        assert self._at_least_one_trial_class_has_features

        data_dict = {}
        if self.enable_process_pooling:
            with ProcessPoolExecutor() as executor:
                for animal_id, trial_objects in self.animal_id_vs_trial_objects.items():
                    trial_objects = [
                        trial_object
                        for trial_object in copy(trial_objects)
                        if trial_object._trial_has_feature_frame
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
                        if trial_object._trial_has_feature_frame
                    ),
                    [],
                )

        result = pd.DataFrame.from_dict(
            data_dict, orient="index", columns=self._feature_frame_columns()
        )
        result.index.name = "Animal ID"
        result.columns.names = ["Feature", "Category"]

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
        if self.enable_process_pooling:
            with ProcessPoolExecutor() as executor:
                rows = executor.map(attrgetter("motion_features"), self.trial_objects)
        else:
            rows = [trial_object.motion_features for trial_object in self.trial_objects]

        return pd.DataFrame(
            rows,
            columns=self._motion_summary_column_index,
            index=self._trial_id_series,
        )

    # Private methods

    @cached_property
    def _at_least_one_trial_class_has_features(self):
        return any(
            trial_class.feature_summary_column for trial_class in self._trial_classes
        )

    def _feature_frame_columns(self, levels: Optional[int] = None) -> pd.MultiIndex:
        if self.trial_class:
            columns = self.trial_class.feature_summary_column
        elif self.trial_id_vs_trial_class:
            columns = sum(
                (
                    trial_object.feature_summary_column
                    for trial_object in self._trial_classes
                    if trial_object._trial_has_feature_frame
                ),
                [],
            )
        else:
            self._neither_singular_trial_class_or_trial_id_vs_trial_class()

        if levels:
            columns = rise_to_n_levels(columns, levels)

        return pd.MultiIndex.from_tuples(columns)

    @staticmethod
    def _feature_2d_multi_indexer(feature: str, category) -> tuple:
        return tuple([(feature, category) for category in category])

    @cached_property
    def _trial_id_series(self) -> pd.Series:
        return pd.Series(
            self._trial_id_key_view,
            name="Trial ID",
            dtype=np.uint16,
        )

    @cached_property
    def _trial_id_indexed_animal_ids(self) -> pd.Series:
        return pd.Series(
            self.trial_id_vs_animal_id.values(),
            index=self._trial_id_series,  # derived from self.trial_id_vs_animal_id
            name="Animal ID",
            dtype=np.uint16,
        ).sort_index()

    def _trial_id_animal_id(self, levels: Optional[int] = None) -> pd.DataFrame:
        result = self._trial_id_indexed_animal_ids.reset_index()
        if levels:
            columns = rise_to_n_levels(result.columns, levels)
            result.columns = columns
        return result

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

    @property
    def _trial_id_key_view(self):
        return self.trial_id_vs_animal_id.keys()

    @cached_property
    def _trial_classes(self) -> tuple:
        if self.trial_class:
            return (self.trial_class,)
        return tuple(
            sorted(self._trial_class_vs_trial_ids, key=lambda x: x.trial_sequence_index)
        )

    @cached_property
    def _class_labels(self):
        return tuple(trial_class.trial_label for trial_class in self._trial_classes)

    def _difference_warning(self, other, attribute: str):
        if (self_attr := getattr(self, attribute)) == (
            other_attr := getattr(other, attribute)
        ):
            return
        logger.warning(
            f"Joining experiments {self.best_id} & {other.best_id}: "
            f"{self_attr} != {other_attr}"
        )

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
        return self.animal_id_vs_trial_objects.keys()

    @cached_property
    def _motion_summary_column_index(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_tuples(
            self.motion_summary_columns, names=["Feature", "Category"]
        )

    @cached_property
    def _motion_summary_column_depth(self):
        motion_summary_column_index_list = list(self._motion_summary_column_index)
        result = len(motion_summary_column_index_list[0])
        assert all(
            result == len(column) for column in motion_summary_column_index_list[1:]
        )
        return result

    @cached_property
    def motion_summary_columns(self) -> list:
        return motion_2d_multi_indexer("All")

    def _make_categorical_inspection_dir(self, trial_root_dir: Path):
        pass

    def _neither_singular_trial_class_or_trial_id_vs_trial_class(self):
        msg = (
            "Either trial_class has to be singularly defined, "
            "or trial_id_vs_trial_class have to be exclusively defined"
        )
        raise AttributeError(msg)


class BaseTrial(Behaviour):
    coordinate_data_path: FilePath = Field(
        description="Path to file storing coordinate data"
    )
    animal_id: int = Field(description="The ID of the animal in the trial")
    point_label_for_motion_features: Optional[str] = Field(
        description="Label of the node that will be used to track general animal movement"
    )
    rigid_nodes_freezing: Optional[Sequence[Union[str, int]]] = Field(
        None,
        description="Nodes that should remain during freeze/immobility, most often due to fear.",
    )
    stage: Optional[str] = Field(
        None, description="The semantic stage of the experiment"
    )
    inspection_dir: Optional[DirectoryPath] = Field(
        None, description="Path to save figures for inspection of results"
    )
    inspect_image: Optional[FilePath] = Field(
        None,
        description="Image to use as background in the plots for visualising the analysis data",
    )
    # Variables for trials with zones, see doc for more info.
    perimeters: Optional[Sequence] = None
    trial_start_perimeter: Optional[str] = None

    # Class variables
    category: ClassVar[Optional[str]] = "trial"

    trial_sequence_index: ClassVar[Optional[int]] = None
    trial_label: ClassVar[str] = ""

    second_tolerance: ClassVar[float] = 0.15

    feature_headers: ClassVar[Optional[list[str]]] = None
    trial_has_video_space_for_analysis: ClassVar[bool] = False

    @cached_property
    def feature_summary_column(self):
        if not self.trial_label:
            return self.feature_headers
        return pd.MultiIndex.from_product([self.trial_label], self.feature_headers)

    @property
    def motion_features(self) -> list:
        return self.motion.to_list

    @cached_property
    def reader(self):
        try:
            reader_init_func = LABEL_VS_DATA_READER[self.data_format_label]
        except KeyError as e:
            msg = (
                f"{self.data_format_label} as a format for data ingestion has "
                f"no implementation. Choose from: {LABEL_VS_DATA_READER.keys()}"
            )
            raise NotImplemented(msg) from e

        return reader_init_func(
            df_path=self.coordinate_data_path,
            **self._reader_init_kwargs,
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

    @cached_property
    def recording_center_pixel(self) -> np.ndarray:
        return self.recording_resolution / 2.0

    @cached_property
    def motion(self) -> Motion:
        return Motion(self.coordinates_per_frame, self.meters_per_pixel, self.fps)

    # PolygonPerimeter

    def detect_confined_perimeter(self, coordinate: np.ndarray) -> np.ndarray:
        """
        This function is used to determine current location of subject.

        :param coordinate:
        :return:
        """

        coordinate = np.expand_dims(coordinate, 0)
        for label, perimeter in self._int_id_vs_perimeter.items():
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
        if self.enable_process_pooling:
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

    def _process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        if self.trial_has_video_space_for_analysis:
            return cv2.hconcat(
                (
                    self._overlay_video_frame(frame, frame_index),
                    self._create_analysis_frame(frame_index),
                )
            )
        return self._overlay_video_frame(frame, frame_index)

    def _overlay_video_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)

        ax.imshow(frame)
        ax.scatter(*self.coordinates_per_frame[frame_index])
        ax.axis("off")

        canvas.draw()

        return np.frombuffer(canvas.tostring_rgb(), dtype="uint8")

    def _create_analysis_frame(self, frame_index: int) -> np.ndarray:
        raise NotImplemented("The analysis space is a work in progress")

    @cached_property
    def _reader_init_kwargs(self):
        return self.data_import_kwargs

    @cached_property
    def _int_id_vs_perimeter(self) -> dict:
        self._validate_perimeters_object()
        return {perimeter.int_id: perimeter for perimeter in self.perimeters}

    def _validate_perimeters_object(self) -> None:
        if not self.perimeters:
            msg = (
                "perimeters is not defined as an object variable, "
                "which is required for _int_id_vs_perimeter"
            )
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
    def _inspection_image_name(self):
        return f"trial_{self.best_id}.jpg"

    @cached_property
    def _uint_zeros_based_on_frame_length(self) -> np.ndarray:
        return np.zeros(self.number_of_frames, dtype=np.uint8)

    @cached_property
    def _frame_tolerance(self) -> int:
        return round(self.second_tolerance * self.fps)

    @classmethod
    @property
    def _trial_has_feature_frame(cls) -> bool:
        return cls.feature_summary_column is not None
