from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from functools import cached_property, reduce
from logging import getLogger
from operator import attrgetter
from typing import ClassVar, Hashable, Iterable, Literal, Optional, Sequence, TypeVar

import cv2
import numpy as np
import pandas as pd
from pydantic import Field, FilePath, ValidationError, validator, PositiveInt
from pydantic.fields import FieldInfo
from pydantic_numpy import NDArray
from tqdm import tqdm
from yaspin import yaspin
from yaspin.spinners import Spinners

from bikipy import ENABLE_PROCESS_POOLING
from bikipy.core.base_class import BaseBikipyHashable, BaseBikipyInspectMixin, BaseBikipy
from bikipy.core.typing import NDArrayFp64, NDArrayInt16, NDArrayUint8, TrialId
from bikipy.core.video import VideoMetadata, VideoMetadataMixin, incongruity_permissive_video_join
from bikipy.feature.motion import Motion, motion_multi_indexer
from bikipy.ingress.plugin import PluginChangeReference, PluginRadial
from bikipy.perimeter.base import SinglePerimeter, PerimeterSet, BaseSinglePerimeter
from bikipy.reader.deeplabcut import DeepLabCutReader
from bikipy.utils.collection_utils import (
    add_filler_to_sequence,
    chain_iterables_to_multi_index,
    chain_lists_to_tuple,
    copycat_assumes_levels_of_icon,
    flatten_multi_index,
    max_len_in_iterable,
)
from bikipy.utils.ranged_dict import RangeDict

LABEL_to_DATA_READER = {"deeplabcut": DeepLabCutReader}

logger = getLogger(__name__)


class Behaviour(BaseBikipyHashable, BaseBikipyInspectMixin, VideoMetadataMixin):
    data_format_label: Literal["deeplabcut"] = "deeplabcut"

    _live: ClassVar[bool] = False

    @staticmethod
    def multi_index_names(index_content: Iterable):
        match max_level := max_len_in_iterable(index_content):
            case 2:
                return ["Feature", "Location/Category"]
            case 3:
                return ["Stage", "Feature", "Location/Category"]
            case _:
                msg = f"The highest level in the feature_column_index is too high, {max_level}:\n{', '.join(index_content)}"
                raise AttributeError(msg)


class BaseTrial(Behaviour):
    coordinate_data_path: FilePath = Field(..., description="Path to file storing coordinate data")
    reader_kwargs: dict = Field(..., description="Keyword arguments that will be passed on the reader objects on init")
    animal_id: str | PositiveInt = Field(..., description="The ID of the animal in the trial")
    object_tracking_label_for_kinematics: Optional[str] = Field(
        ..., description="Label of the node that will be used to track general animal movement"
    )
    manual_center_pixels: Optional[NDArrayInt16]
    rigid_nodes_freezing: Optional[Sequence[str | PositiveInt]] = Field(
        description="Nodes that should remain during freeze/immobility, most often due to fear.",
    )
    inspect_image: Optional[NDArray] = Field(
        description="Image to use as background in the plots for visualising the analysis data",
    )
    crop_time_seconds: float = 0.0
    crop_from_end: bool = Field(
        True,
        description="Only affective if crop_time_seconds is not 0.0. " "Will crop from start instead when set to False",
    )

    required_video_metadata_fields = {"meters_per_pixel", "recording_resolution", "fps"}

    # Variables for trials with zones, see doc for more info.
    trial_start_perimeter: Optional[str]

    # Class variables
    category = "trial"

    experiment_class_name: ClassVar[str] = ...
    trial_label: ClassVar[str] = ...

    second_tolerance: ClassVar[float] = 0.15

    @classmethod
    @property
    def experiment_class(cls) -> "Experiment":
        from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS

        return EXPERIMENT_NAME_TO_CLASS[cls.experiment_class_name]

    @classmethod
    @property
    def trial_has_defined_features(cls) -> bool:
        try:
            return bool(cls.feature_headers)
        except AttributeError:
            return False

    @classmethod
    @property
    def experiment_stage_index(cls) -> int:
        return cls.experiment_class.trial_class_name_to_stage_index[cls.__name__]

    @classmethod
    @property
    def column_index_levels(cls):
        return 3 if cls.experiment_stage_index else 2

    @cached_property
    def manual_center_meters(self) -> NDArrayFp64 | None:
        if self.manual_center_pixels is not None:
            return self.manual_center_pixels * self.video.meters_per_pixel

    @cached_property
    def center_meter_translation(self) -> NDArrayFp64 | None:
        if self.manual_center_meters is not None:
            return self.manual_center_meters - self.video.center_meters

    @property
    def motion_features(self) -> tuple:
        return self.motion.as_tuple

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
            label=self.coordinate_data_path.stem,
            manual_video=self.video,
            crop_time_seconds=self.crop_time_seconds,
            # crop_from_end=self.crop_from_end,
            **self.reader_kwargs,
        )

    @property
    def framewise_confined_coordinates(self) -> NDArrayFp64:
        return self.reader[self.object_tracking_label_for_kinematics]

    @cached_property
    def number_of_frames(self) -> int:
        return len(self.framewise_confined_coordinates)

    @cached_property
    def experiment_seconds(self) -> int:
        return self.framewise_confined_coordinates.shape[0] / self.video.fps

    @cached_property
    def motion(self) -> Motion:
        return Motion(
            coordinate_sequence=self.framewise_confined_coordinates,
            fps=self.video.fps,
        )

    @property
    def perimeters(self) -> list[SinglePerimeter]:
        return []

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
    def trial_id_all_features_df_row(self):
        if self.trial_has_defined_features:
            return *self.feature_df_rows, *self.motion_features
        return self.motion_features

    @property
    def _video(self):
        video = super()._video
        if self.perimeters:
            perimeter_video = reduce(
                incongruity_permissive_video_join, (perimeter.video for perimeter in self.perimeters)
            )
            # Manually passed video parameters should override any
            new_video = VideoMetadata.join(video, perimeter_video, ignore_incongruity=True)

            # The resolution on perimeters should be more correct than whatever
            # provided by the user, hence it being master
            final_video = VideoMetadata.join(new_video, video, ignore_incongruity=True)
            for perimeter in self.perimeters:
                perimeter.manual_video = final_video
            return final_video
        return video

    @cached_property
    def _uint_zeros_based_on_frame_length(self) -> NDArrayUint8:
        return np.zeros(self.number_of_frames, dtype=np.uint8)

    @cached_property
    def _frame_tolerance(self) -> int:
        return round(self.second_tolerance * self.video.fps)


Trial = TypeVar("Trial", bound=BaseTrial)


class BaseExperiment(Behaviour):
    manual_trial_ids: Optional[tuple]
    trial_id_to_trial_class_name: Optional[dict] = Field(default_factory=dict)
    trial_id_to_keyword_arguments: Optional[dict] = Field(default_factory=dict)
    trial_class_name_to_keyword_arguments: Optional[dict] = Field(default_factory=dict)
    trial_id_range_to_keyword_arguments: Optional[RangeDict]
    common_trial_keyword_arguments: Optional[dict] = Field(default_factory=dict)
    stage: Optional[str] = Field(
        description="Experiment stage label, if experiment object is in a sequence of experiment objects"
    )
    label: Optional[str]
    inspect_image_path: Optional[FilePath] = Field(description="Used globally")
    compute_only_one_df_row: bool = Field(False, description="Used to rapidly generate combo df during debugging")

    habituation_trial_class: ClassVar[Trial] = Field(
        ..., description="The trial class that will be used in case first_trial_is_habituation is called"
    )
    trial_classes: ClassVar[tuple[Trial]] = Field(..., description="Trial classes designed for this experiment class")

    @validator("trial_id_to_trial_class_name")
    def sort_trial_id_to_trial_class_name_ascending(cls, value):
        return dict(sorted(value.items()))

    @validator("trial_id_to_trial_class_name")
    def sort_trial_id_to_keyword_arguments_ascending(cls, value):
        return dict(sorted(value.items()))

    def __getitem__(self, item: int):
        return self.trial_id_to_trial_object[item]

    def save(self):
        self.trial_label_to_df
        super().save()

    @classmethod
    def first_trial_is_habituation(cls) -> "Experiment":
        cls.habituation_trial_class.experiment_class_name = cls.__name__
        cls.trial_classes = (cls.habituation_trial_class, *cls.trial_classes)
        return cls

    @classmethod
    @property
    def trial_class_name_to_stage_index(cls) -> dict[str, int]:
        return {trial_class_name: i for trial_class_name, i in enumerate(cls.trial_classes)}

    @classmethod
    @property
    def trial_sequence_length(cls) -> int:
        return len(cls.trial_classes)

    @classmethod
    @property
    def has_stages(cls) -> bool:
        return cls.trial_sequence_length != 1

    @classmethod
    @property
    def column_index_levels(cls) -> int:
        return 3 if cls.has_stages else 2

    @classmethod
    @property
    def trial_classes_with_feature_headers(cls) -> int:
        return sum(1 for _trial_class in cls.trial_classes if _trial_class.trial_has_defined_features)

    @classmethod
    @property
    def trial_class_names(cls) -> tuple[str, ...]:
        return tuple(trial_class.__name__ for trial_class in cls.trial_classes)

    @classmethod
    @property
    def trial_class_labels(cls) -> tuple[str, ...]:
        return tuple(trial_class.trial_label for trial_class in cls.trial_classes)

    @classmethod
    @property
    def trial_class_name_to_label(cls) -> dict:
        return dict(zip(cls.trial_class_names, cls.trial_class_labels))
        # return {name: label for name, label in zip(cls.trial_class_names, cls.trial_class_labels)}

    @classmethod
    @property
    def trial_class(cls) -> Trial:
        if cls.has_stages:
            msg = f"{cls.__name__}: trial_class attribute can only be utilized when there is only one Trial class"
            raise AttributeError(msg)
        return cls.trial_classes[0]

    @classmethod
    @property
    def stage_index_to_trial_class(cls) -> dict[int, Trial]:
        if not cls.has_stages:
            msg = f"{cls.__name__}: stage_index_to_trial_class is undefined in non-sequential experiment classes"
            raise AttributeError(msg)

        try:
            return {i: trial_class for i, trial_class in enumerate(cls.trial_classes)}
        except AttributeError:
            msg = "experiment_stage_index must be defined for each trial class when working with a sequence of trial classes"
            raise AttributeError(msg)

    @classmethod
    @property
    def stage_index_to_trial_class_name(cls):
        if isinstance(tuple(cls.stage_index_to_trial_class.values())[-1], FieldInfo):
            msg = (
                f"To the developers: Setting the trial_classes class variable is required; "
                f"please do so for {cls.__name__}"
            )
            raise AttributeError(msg)
        return {i: trial_class.__name__ for i, trial_class in cls.stage_index_to_trial_class.items()}

    @classmethod
    @property
    def trial_class_name_to_trial_class(cls) -> dict[str, Trial]:
        if not cls.has_stages:
            msg = (
                f"{cls.__name__}: trial_class_name_to_trial_class attribute can only be utilized when "
                f"there are many Trial classes"
            )
            raise AttributeError(msg)

        try:
            return {trial_class.__name__: trial_class for trial_class in cls.trial_classes}
        except AttributeError:
            msg = "experiment_stage_index must be defined for each trial class when working with a sequence of trial classes"
            raise AttributeError(msg)

    def trial_keyword_arguments(self, trial_id: TrialId) -> dict:
        """
        Function useful for customizing initiation parameters for trial objects
        """
        result = {**self.video.manual_video_metadata, "data_format_label": self.data_format_label}

        if self.common_trial_keyword_arguments:
            result.update(self.common_trial_keyword_arguments)

        if (
            self.has_stages
            and (trial_class_name := self.trial_id_to_trial_class_name[trial_id])
            in self.trial_class_name_to_keyword_arguments
        ):
            result.update(self.trial_class_name_to_keyword_arguments[trial_class_name])

        if self.trial_id_to_keyword_arguments:
            result.update(self.trial_id_to_keyword_arguments[trial_id])

        if self.trial_id_range_to_keyword_arguments:
            if not isinstance(trial_id, int):
                msg = "Trial IDs must be integers when trial_id_range_to_keyword_arguments is used"
                raise AttributeError(msg)
            result.update(self.trial_id_range_to_keyword_arguments[trial_id])

        assert result["coordinate_data_path"]

        if "animal_id" not in result:
            result["animal_id"] = trial_id

        result["inspect_directory"] = self.inspect_directory
        if "inspect_image" not in result:
            result["inspect_image"] = self._initialized_inspect_image

        if "label_to_perimeter" in result:
            result.update(result.pop("label_to_perimeter"))

        if "perimeter_set" in result:
            perimeter_set: PerimeterSet = result.pop("perimeter_set")
            result.update(perimeter_set.label_to_perimeter)

        if PluginRadial.bikipy_trial_key in result:
            perimeter_set_group: dict = result.pop(PluginRadial.bikipy_trial_key)
            result.update(perimeter_set_group)

        if PluginChangeReference.bikipy_trial_key in result:
            val = result.pop(PluginChangeReference.bikipy_trial_key)
            if isinstance(val, PerimeterSet):
                result.update(val.label_to_perimeter)
            elif isinstance(val, BaseSinglePerimeter):
                result[val.label] = val
            elif isinstance(val, dict):
                result.update(val)
            else:
                msg = f"Unsupported type for PluginChangeReference: {type(val)}"
                raise AttributeError(msg)

        return result

    @cached_property
    def trial_objects(self) -> list[Trial]:
        result, bad_trial_ids_to_error_msg = [], {}
        for trial_id in self.trial_ids:
            trial_class = (
                self.trial_class_name_to_trial_class[self.trial_id_to_trial_class_name[trial_id]]
                if self.has_stages
                else self.trial_class
            )
            try:
                result.append(trial_class(**self.trial_keyword_arguments(trial_id)))
            except ValidationError as e:
                bad_trial_ids_to_error_msg[trial_id] = str(e)
                continue
        if bad_trial_ids_to_error_msg:
            msg = f"Some trial IDs yielded pydantic validation errors:"
            for trial_id, msg in bad_trial_ids_to_error_msg.items():
                msg += f"\n{trial_id}:\n{msg}\n"
            raise ValueError(f"{msg}\n{len(bad_trial_ids_to_error_msg)} errors out of {len(self.trial_ids)} Trials")
        return result

    @cached_property
    def trial_class_name_to_trial_ids(self) -> dict[str, Trial]:
        return {
            trial_class.__name__: trial_ids for trial_class, trial_ids in self._trial_class_name_to_trial_ids.items()
        }

    @cached_property
    def trial_id_to_trial_object(self) -> dict[Hashable, Trial]:
        return {trial.label: trial for trial in self.trial_objects}

    @cached_property
    def trial_class_name_to_trial_objects(self) -> dict[str, Trial]:
        return {
            trial_class_name: [self.trial_id_to_trial_object[trial_id] for trial_id in trial_ids]
            for trial_class_name, trial_ids in self._trial_class_name_to_trial_ids.items()
        }

    @cached_property
    def animal_id_to_trial_objects(self) -> dict[Hashable, Trial]:
        result = {}
        for trial in self.trial_objects:
            if trial.animal_id in result:
                result[trial.animal_id].append(trial)
            else:
                result[trial.animal_id] = [trial]
        for trials in result.values():
            trials.sort(key=lambda t: t.label)
        return dict(sorted(result.items()))

    @cached_property
    def animal_id_to_trial_ids(self) -> dict:
        return {
            animal_id: (trial_object.int_id for trial_object in trial_objects)
            for animal_id, trial_objects in self.animal_id_to_trial_objects.items()
        }

    @cached_property
    def trial_id_to_animal_id(self) -> dict:
        result = {trial_id: kwargs["animal_id"] for trial_id, kwargs in self.trial_id_to_keyword_arguments.items()}
        return dict(sorted(result.items(), key=lambda item: item[1]))

    @cached_property
    def stage_index_to_trial_objects(self) -> dict[int, Trial]:
        result = {}
        for trial in self.trial_objects:
            if trial.experiment_stage_index in result:
                result[trial.experiment_stage_index].append(trial)
            else:
                result[trial.experiment_stage_index] = [trial]
        return result

    @cached_property
    def trial_ids(self) -> tuple:
        if self.manual_trial_ids:
            result = self.manual_trial_ids
        elif not self.has_stages and self.trial_class:
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
    def number_of_trials(self) -> int:
        return len(self.trial_ids)

    # DataFrame methods =========================================

    @cached_property
    def trial_label_to_df(self) -> dict[str | PositiveInt, pd.DataFrame]:
        data_dicts = defaultdict(dict)
        if ENABLE_PROCESS_POOLING:
            with yaspin(Spinners.pong, text="Computing experiment features..."):
                with ProcessPoolExecutor() as executor:
                    for trial_class_label, trial_objects in self._trial_class_label_to_trial_objects.items():
                        data_dicts[trial_class_label] = {
                            trial_object.label: features
                            for trial_object, features in zip(
                                trial_objects, executor.map(attrgetter("trial_id_all_features_df_row"), trial_objects)
                            )
                        }
        else:
            for trial_class_label, trial_objects in tqdm(
                self._trial_class_label_to_trial_objects.items(), desc="Computing experiment features"
            ):
                if self.compute_only_one_df_row:
                    data_dicts[trial_class_label] = {
                        trial_objects[0].label: trial_objects[0].trial_id_all_features_df_row
                    }
                    break
                data_dicts[trial_class_label] = {
                    trial_object.label: trial_object.trial_id_all_features_df_row for trial_object in trial_objects
                }

        result = {}
        for trial_class_label, data_dict in data_dicts.items():
            result[trial_class_label] = pd.DataFrame.from_dict(
                data_dict,
                orient="index",
                columns=self.trial_class_label_to_trial_id_indexed_column_index[trial_class_label],
            )
            result[trial_class_label].index.name = "Trial ID"

        return result

    @cached_property
    def combined_feature_motion_df(self) -> pd.DataFrame:
        return pd.concat(
            (
                self.feature_df_fit_to_motion_index,
                self.motion_df_fit_to_feature_index,
            ),
            axis=1,
        )

    @cached_property
    def animal_id_indexed_feature_df(self) -> pd.DataFrame:
        assert self.experiment_has_features

        data_dict = {}
        if ENABLE_PROCESS_POOLING:
            with yaspin(Spinners.pong, text="Computing experiment features..."):
                with ProcessPoolExecutor() as executor:
                    for animal_id, trial_objects in self.animal_id_to_trial_objects.items():
                        trial_objects = [
                            trial_object
                            for trial_object in copy(trial_objects)
                            if trial_object.trial_has_defined_features
                        ]
                        data_dict[animal_id] = chain_lists_to_tuple(
                            executor.map(attrgetter("feature_df_rows"), trial_objects)
                        )
        else:
            for animal_id, trial_objects in tqdm(
                self.animal_id_to_trial_objects.items(), desc="Computing experiment features"
            ):
                data_dict[animal_id] = chain_lists_to_tuple(
                    (
                        trial_object.feature_df_rows
                        for trial_object in trial_objects
                        if trial_object.trial_has_defined_features
                    )
                )
                if self.compute_only_one_df_row:
                    break

        result = pd.DataFrame.from_dict(data_dict, orient="index", columns=self.feature_column_index)
        result.index.name = "Animal"

        return result

    @cached_property
    def animal_id_indexed_motion_df(self) -> pd.DataFrame:
        data_dict = {}
        if ENABLE_PROCESS_POOLING:
            with yaspin(Spinners.pong, text="Computing motion features..."):
                with ProcessPoolExecutor() as executor:
                    for animal_id, trial_objects in self.animal_id_to_trial_objects.items():
                        data_dict[animal_id] = chain_lists_to_tuple(
                            executor.map(attrgetter("motion_features"), trial_objects),
                        )
        else:
            for animal_id, trial_objects in tqdm(
                self.animal_id_to_trial_objects.items(), desc="Computing motion features"
            ):
                data_dict[animal_id] = chain_lists_to_tuple(
                    (trial_object.motion_features for trial_object in trial_objects)
                )
                if self.compute_only_one_df_row:
                    break

        result = pd.DataFrame.from_dict(data_dict, orient="index", columns=self.animal_motion_column_index)
        result.index.name = "Animal"

        return result

    # Motion <-> Feature fitting ===================================

    @cached_property
    def feature_df_fit_to_motion_index(self) -> pd.DataFrame:
        if self.feature_column_index.nlevels >= self.animal_motion_column_index.nlevels:
            return self.animal_id_indexed_feature_df
        return copycat_assumes_levels_of_icon(self.animal_id_indexed_feature_df, self.animal_id_indexed_motion_df)

    @cached_property
    def motion_df_fit_to_feature_index(self) -> pd.DataFrame:
        if self.animal_motion_column_index.nlevels >= self.feature_column_index.nlevels:
            return self.animal_id_indexed_motion_df
        return copycat_assumes_levels_of_icon(self.animal_id_indexed_motion_df, self.animal_id_indexed_feature_df)

    # DataFrame helper methods =====================================

    @classmethod
    @property
    def feature_column_index(cls) -> pd.MultiIndex:
        all_feature_headers = chain_lists_to_tuple(
            (
                add_filler_to_sequence(trial_class.feature_headers, trial_class.trial_label)
                if cls.trial_classes_with_feature_headers > 1
                else trial_class.feature_headers
                for trial_class in cls.trial_classes
                if trial_class.trial_has_defined_features
            )
        )
        if not all_feature_headers:
            msg = f"{cls.__name__} does not have any features, yet feature column index was called"
            raise AttributeError(msg)

        return pd.MultiIndex.from_tuples(all_feature_headers, names=cls.multi_index_names(all_feature_headers))

    @classmethod
    @property
    def experiment_has_features(cls) -> bool:
        return cls.feature_column_index is not None

    @classmethod
    @property
    def motion_column_headers(cls) -> list[tuple[str, ...], ...]:
        return motion_multi_indexer("All", 2)

    @classmethod
    @property
    def motion_column_index(cls) -> pd.MultiIndex:
        return pd.MultiIndex.from_tuples(
            cls.motion_column_headers, names=cls.multi_index_names(cls.motion_column_headers)
        )

    @classmethod
    @property
    def animal_motion_column_index(cls) -> pd.MultiIndex:
        return chain_iterables_to_multi_index(
            add_filler_to_sequence(cls.motion_column_index, trial_label) for trial_label in cls.trial_class_labels
        )

    @classmethod
    @property
    def trial_class_label_to_trial_id_indexed_column_index(cls):
        return {
            trial_class.trial_label: flatten_multi_index(
                chain_lists_to_tuple([trial_class.feature_headers, cls.motion_column_headers])
            )
            for trial_class in cls.trial_classes
        }

    # Helper methods =====================================

    @cached_property
    def _trial_id_series(self) -> pd.Series:
        return pd.Series(self.trial_ids, name="Trial ID")

    @cached_property
    def _trial_id_indexed_animal_ids(self) -> pd.Series:
        return pd.Series(
            self.trial_id_to_animal_id.values(),
            index=self._trial_id_series,  # derived from self.trial_id_to_animal_id
            name="Animal",
        ).sort_index()

    @cached_property
    def _trial_class_name_to_trial_ids(self) -> dict:
        if not self.trial_id_to_trial_class_name:
            msg = (
                "This experiment object has no trial_id_to_trial_class_name, "
                "this attribute is reserved for experiments with "
                "several trial classes"
            )
            raise AttributeError(msg)

        result = defaultdict(list)
        for trial_id, trial_class_name in self.trial_id_to_trial_class_name.items():
            result[trial_class_name].append(trial_id)

        return result

    @cached_property
    def _trial_class_label_to_trial_objects(self) -> dict:
        if not self.has_stages:
            return {self.trial_class.trial_label: self.trial_objects}

        return {
            self.trial_class_name_to_label[trial_class_name]: trial_objects
            for trial_class_name, trial_objects in self.trial_class_name_to_trial_objects.items()
        }

    @cached_property
    def _class_labels(self):
        return tuple(trial_class.trial_label for trial_class in self.trial_classes)

    @cached_property
    def _initialized_inspect_image(self) -> NDArray | None:
        if not self.inspect_image_path:
            return None
        return cv2.imread(str(self.inspect_image_path))

    @staticmethod
    def _neither_singular_trial_class_or_trial_id_to_trial_class_name(self):
        msg = (
            "Either trial_class has to be singularly defined, "
            "or trial_id_to_trial_class_name have to be exclusively defined"
        )
        raise AttributeError(msg)


Experiment = TypeVar("Experiment", bound=BaseExperiment)


class HabituationTrialMixin(BaseBikipy):
    trial_label = "Habituation"
