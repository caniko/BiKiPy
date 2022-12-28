from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property, lru_cache, reduce
from logging import getLogger
from operator import attrgetter
from typing import ClassVar, Hashable, Iterable, Literal, Optional, Sequence, TypeVar

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    PositiveInt,
    ValidationError,
    validator,
)
from pydantic.fields import FieldInfo
from pydantic_numpy import NDArray
from pydantic_numpy.dtype import NDArrayFp64, NDArrayInt16, NDArrayUint8
from tqdm import tqdm
from yaspin import yaspin
from yaspin.spinners import Spinners

from bikipy import runtime_settings
from bikipy.core.base_class import BaseBikipyHashable, BaseBikipyInspectMixin
from bikipy.core.typing import TrialId
from bikipy.core.video import (
    VideoMetadata,
    VideoMetadataMixin,
    incongruity_permissive_video_join,
)
from bikipy.feature.motion import Motion, motion_multi_indexer
from bikipy.ingress.plugin import PluginChangeReference, PluginRadial
from bikipy.ingress.workflow.base import FIRST_TRIAL_IS_HABITUATION_INGRESS_FIELD
from bikipy.perimeter.base import (
    BaseSinglePerimeter,
    Perimeter,
    PerimeterSet,
    SinglePerimeter,
)
from bikipy.reader.data_with_likelihood import DeepLabCutReader
from bikipy.utils.collection_utils import max_len_in_iterable
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


class BaseTrial(Behaviour, VideoMetadataMixin):
    framewise_coordinates_path: FilePath = Field(..., description="Path to file storing coordinate data")
    manual_reader_kwargs: dict = Field(
        ..., description="Keyword arguments that will be passed on the reader objects on init"
    )
    animal_id: str | PositiveInt = Field(..., description="The ID of the animal in the trial")
    object_tracking_label_for_kinematics: Optional[str] = Field(
        ..., description="Label of the node that will be used to track general animal movement"
    )
    coordinate_timestamp_set: Optional[NDArray]
    manual_center_pixels: Optional[NDArrayInt16]
    rigid_nodes_freezing: Optional[Sequence[str | PositiveInt]] = Field(
        description="Nodes that should remain during freeze/immobility, most often due to fear.",
    )
    crop_time_seconds: float = 0.0
    crop_from_end: bool = Field(
        True,
        description="Only affective if crop_time_seconds is not 0.0. Will crop from start instead when set to False",
    )

    # Derive meters per pixel from perimeter
    meters_per_pixel_from_perimeter: bool = Field(
        False,
        description="Derive meters per pixel from perimeter dimensions. The ratio is derived from source defined in "
        "meters_per_pixel_from_perimeter_source",
    )
    meters_per_pixel_from_perimeter_source: Literal["side", "diagonal", "diameter", "radius", None] = Field(
        None,
        description="The perimeter attribute that will be used to derive meters_per_pixel. Supported sources with "
        "respect to SinglePerimeter type:\n"
        "Polygon: To be decided\n"
        "Regular polygon (every side has equal length): side\n"
        "Rectangle: diagonal\n"
        "Circle: diameter, radius\n",
    )
    length_meters_of_meters_per_pixel_source: Optional[float]
    manual_perimeter_to_derive_meters_per_pixel: Optional[str]

    # Class variables
    category = "trial"

    perimeter_labels: ClassVar[set[str]] = set()
    _label_to_perimeter: ClassVar[dict[str, SinglePerimeter] | None]

    # Variables for trials with zones, see doc for more info.
    trial_start_perimeter: Optional[str]

    required_video_metadata_fields = {"meters_per_pixel", "recording_resolution", "fps"}

    experiment_class_name: ClassVar[str] = ...
    trial_label: ClassVar[str] = ...
    constant_feature_headers: ClassVar[tuple[tuple[str, ...]] | None] = motion_multi_indexer("All", 2)

    second_tolerance: ClassVar[float] = 0.15

    @classmethod
    @property
    def exclude_from_settings_schema(cls) -> set[str]:
        """
        Some required fields for a class are sometimes highly specific to its respective object. These fields should
        be recorded in this class-property to be excluded by the settings generator function in the ingress module
        :return:
        """
        return {
            "framewise_coordinates_path",
            "coordinate_timestamp_set_path",
            "manual_reader_kwargs",
            "animal_id",
            *cls.perimeter_physical_object_labels,
            *super().exclude_from_settings_schema,
        }

    @classmethod
    @property
    def perimeter_physical_object_labels(cls) -> set[str]:
        result = cls.perimeter_labels
        if hasattr(cls, "physical_object_labels"):
            result = result.union(cls.physical_object_labels)
        return result

    @classmethod
    @property
    def experiment_class(cls) -> "Experiment":
        from bikipy.behaviour.mapping import experiment_name_to_class

        return experiment_name_to_class[cls.experiment_class_name]

    @property
    def perimeter_to_derive_meters_per_pixel(self) -> Perimeter:
        if self.manual_perimeter_to_derive_meters_per_pixel:
            try:
                return self._label_to_perimeter[self.manual_perimeter_to_derive_meters_per_pixel]
            except TypeError:
                # self._label_to_perimeter is None -> TypeError
                msg = f"The class, {self.__class__.__name__}, does not define _label_to_perimeter, which makes the mapping of manual_perimeter_to_derive_meters_per_pixel to a Perimeter object impossible"
                raise AttributeError(msg)

    @cached_property
    def manual_center_meters(self) -> NDArrayFp64 | None:
        if self.manual_center_pixels is not None:
            return self.manual_center_pixels * self.video.meters_per_pixel

    @cached_property
    def center_meter_translation(self) -> NDArrayFp64 | None:
        if self.manual_center_meters is not None:
            return self.manual_center_meters - self.video.center_meters

    @property
    def _reader_kwargs(self) -> dict:
        return {
            "df_path": self.framewise_coordinates_path,
            "timestamp_index": self.coordinate_timestamp_set,
            "label": self.framewise_coordinates_path.stem,
            "manual_video": self.video,
            "crop_time_seconds": self.crop_time_seconds,
            **self.manual_reader_kwargs,
        }

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

        return reader_init_func(**self._reader_kwargs)

    @property
    def kinematic_coordinates(self) -> NDArrayFp64:
        return self.reader[self.object_tracking_label_for_kinematics]

    @cached_property
    def number_of_frames(self) -> int:
        return len(self.kinematic_coordinates)

    @cached_property
    def experiment_seconds(self) -> int:
        return self.kinematic_coordinates.shape[0] / self.video.fps

    @cached_property
    def motion(self) -> Motion:
        return Motion(
            coordinate_sequence=self.kinematic_coordinates,
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

    # Miscellaneous

    @cached_property
    def trial_feature_series(self) -> pd.Series:
        # Concatenate and reverse the order
        result = pd.concat(self._trial_feature_series_list[::-1], axis=0)
        self._post_analysis_flush()
        return result

    @property
    def _trial_feature_series_list(self) -> list[pd.Series]:
        return [self.reader.info, pd.Series(self.motion.as_tuple, index=self.constant_feature_headers)]

    @property
    def _video(self) -> VideoMetadata:
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

            if self.meters_per_pixel_from_perimeter:
                logger.debug("meters_per_pixel_from_perimeter -> True: Deriving meters_per_pixel from perimeter")
                if not self.perimeter_to_derive_meters_per_pixel:
                    msg = f"perimeter_to_derive_meters_per_pixel is not defined for class, {self.__class__.__name__}"
                    raise AttributeError(msg)

                final_video.meters_per_pixel = (
                    self.perimeter_to_derive_meters_per_pixel.derived_meters_per_pixel.derived_meters_per_pixel
                )

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

    def _post_analysis_flush(self) -> None:
        self.reader.flush_reads()
        self.video.flush()


Trial = TypeVar("Trial", bound=BaseTrial)


class HabituationTrialMixin(BaseModel):
    trial_label = "Habituation"


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
    trial_init_error_out_dir: Optional[DirectoryPath] = Field(description="Directory to write Trial class init errors")

    compute_first_two_feature_series_only: bool = Field(
        False, description="Used to rapidly generate combo df during debugging"
    )
    skip_habituation: bool = Field(
        False, description="Skip the habituation class during analysis, practically skipping the the habituation class"
    )

    experiment_labels: ClassVar[set[str]] = ...

    habituation_trial_class: ClassVar[Optional[Trial]] = Field(
        description="The trial class that will be used in case set_first_trial_to_habituation is called"
    )
    _first_trial_is_habituation: ClassVar[bool] = False

    # "Sequence of trial classes designed for the experiment class"
    trial_sequence: ClassVar[tuple[Trial, ...]] = ...

    @classmethod
    @property
    def trial_classes(cls) -> set[Trial]:
        """All trials designed for the experiment class"""
        return set(cls.trial_sequence)

    @classmethod
    @property
    def exclude_from_settings_schema(cls) -> set[str]:
        """
        Some required fields for a class are sometimes highly specific to its respective object. These fields should
        be recorded in this class-property to be excluded by the settings generator function in the ingress module
        :return:
        """
        return super().exclude_from_settings_schema.union(
            {
                "trial_id_to_trial_class_name",
                "trial_id_to_keyword_arguments",
                "trial_class_name_to_keyword_arguments",
                "trial_id_range_to_keyword_arguments",
                "common_trial_keyword_arguments",
            }
        )

    @validator("trial_id_to_trial_class_name")
    def sort_trial_id_to_trial_class_name_ascending(cls, value):
        return dict(sorted(value.items()))

    @validator("trial_id_to_trial_class_name")
    def sort_trial_id_to_keyword_arguments_ascending(cls, value):
        return dict(sorted(value.items()))

    def __getitem__(self, item: int):
        return self.trial_id_to_trial_object[item]

    def save(self):
        self._trial_class_to_trial_series_set
        super().save()

    @classmethod
    def trial_sequence_repetition(cls, repetitions: int) -> "Experiment":
        cls.trial_sequence = tuple(cls.trial_classes) * repetitions
        return cls

    @classmethod
    def set_first_trial_to_habituation(cls) -> "Experiment":
        if not cls.habituation_trial_class:
            msg = f"Habituation trial class for {cls.__name__} has not been defined, contact the maintainers"
            raise AttributeError(msg)
        if cls._first_trial_is_habituation:
            msg = "The first trial has already been set to habituation"
            raise AttributeError(msg)

        cls.habituation_trial_class.experiment_class_name = cls.__name__
        cls.trial_sequence = (cls.habituation_trial_class, *cls.trial_classes)
        cls._first_trial_is_habituation = True

        return cls

    @classmethod
    @property
    def trial_class_name_to_stage_index(cls) -> dict[str, int]:
        return {trial_class_name: i for trial_class_name, i in enumerate(cls.trial_sequence)}

    @classmethod
    @property
    def trial_sequence_length(cls) -> int:
        return len(cls.trial_sequence)

    @classmethod
    @property
    def has_stages(cls) -> bool:
        return cls.trial_sequence_length != 1

    @classmethod
    @property
    def trial_class_names(cls) -> tuple[str, ...]:
        return tuple(trial_class.__name__ for trial_class in cls.trial_sequence)

    @classmethod
    @property
    def trial_class_labels(cls) -> tuple[str, ...]:
        return tuple(trial_class.trial_label for trial_class in cls.trial_sequence)

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
        return cls.trial_sequence[0]

    @classmethod
    @property
    def stage_index_to_trial_class(cls) -> dict[int, Trial]:
        if not cls.has_stages:
            msg = f"{cls.__name__}: stage_index_to_trial_class is undefined in non-sequential experiment classes"
            raise AttributeError(msg)

        try:
            return {i: trial_class for i, trial_class in enumerate(cls.trial_sequence)}
        except AttributeError:
            msg = "experiment_stage_index must be defined for each trial class when working with a sequence of trial classes"
            raise AttributeError(msg)

    @classmethod
    @property
    def stage_index_to_trial_class_name(cls):
        if isinstance(tuple(cls.stage_index_to_trial_class.values())[-1], FieldInfo):
            msg = (
                f"To the developers: Setting the trial_sequence class variable is required; "
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
            return {trial_class.__name__: trial_class for trial_class in cls.trial_sequence}
        except AttributeError:
            msg = "experiment_stage_index must be defined for each trial class when working with a sequence of trial classes"
            raise AttributeError(msg)

    def trial_keyword_arguments(self, trial_id: TrialId) -> dict:
        """
        Function useful for customizing initiation parameters for trial objects
        """
        result = {**self.video.dict(exclude_unset=True), "data_format_label": self.data_format_label}

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

        assert result["framewise_coordinates_path"]

        if "animal_id" not in result:
            result["animal_id"] = trial_id

        result["inspect_arg"] = self.inspect_arg

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
            for trial_id, trial_msg in bad_trial_ids_to_error_msg.items():
                msg += f"\n{trial_id}:\n{trial_msg}\n"
            if self.trial_init_error_out_dir:
                error_out_path = self.trial_init_error_out_dir / "trial_init_error.log"
                with open(error_out_path, "w") as out_file:
                    out_file.write(msg)
                    msg += f"\nError logs were saved to {error_out_path}\n"
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
    def animals_ids(self) -> set:
        return set(self.animal_id_to_trial_objects)

    @cached_property
    def number_of_trials(self) -> int:
        return len(self.trial_ids)

    @property
    def _animal_id_to_sequential_features_list(self) -> list[pd.DataFrame]:
        """
        example:
            base = super()._animal_id_to_sequential_features_list
            animal_id_to_features = {}
            for animal_id, trial_ids in self.animal_id_to_trial_ids.items():
                animal_series = []
                for trial_class_name in self.trial_class_names:
                    pass
            return base
        :return:
        """
        return []

    # DataFrame methods =========================================

    def analyze_trials(self):
        result = defaultdict(dict)
        if not runtime_settings.disable_process_pooling:
            with yaspin(Spinners.pong, text="Computing experiment features..."):
                with ProcessPoolExecutor(max_workers=runtime_settings.threads_to_use) as executor:
                    for trial_class_label, trial_objects in self._trial_class_label_to_trial_objects.items():
                        result[trial_class_label] = {
                            trial_object.label: features
                            for trial_object, features in zip(
                                trial_objects, executor.map(attrgetter("trial_feature_series"), trial_objects)
                            )
                        }
        else:
            for trial_class_label, trial_objects in tqdm(
                self._trial_class_label_to_trial_objects.items(), desc="Computing experiment features"
            ):
                result[trial_class_label] = {
                    trial_object.label: trial_object.trial_feature_series for trial_object in trial_objects
                }
        return result

    @cached_property
    def _trial_class_to_trial_series_set(self):
        return self.analyze_trials()

    @cached_property
    def _animal_id_to_sequential_features(self) -> pd.DataFrame | None:
        if self._animal_id_to_sequential_features_list:
            # Concatenate and reverse the order
            return pd.concat(self._animal_id_to_sequential_features_list[::-1], axis=0)

    @cached_property
    def trial_label_to_df(self) -> dict[str | PositiveInt, pd.DataFrame]:
        result = {}
        for trial_class_label, data_dict in self._trial_class_to_trial_series_set.items():
            result[trial_class_label] = pd.DataFrame.from_dict(data_dict, orient="index")
            result[trial_class_label].index.name = "Trial ID"

        if self._animal_id_to_sequential_features:
            result["AnimalToSequential"] = self._animal_id_to_sequential_features

        return result

    @cached_property
    def animal_id_indexed_feature_df(self) -> pd.DataFrame:
        @lru_cache(self.trial_sequence_length)
        def add_trial_class_to_top_of_multi_index():
            return pd.MultiIndex.from_product([[trial_class_name], list(series.index)])

        animal_id_to_feature_series = {}
        for animal_id, trial_ids in self.animal_id_to_trial_ids.items():
            animal_series = []
            for trial_class_name in self.trial_class_names:
                trial_class_related_feature_series_data = self._trial_class_to_trial_series_set[trial_class_name]
                for trial_id in trial_ids:
                    try:
                        series = trial_class_related_feature_series_data[trial_id]
                        series.index = add_trial_class_to_top_of_multi_index()
                        break  # One animal instance per trial_class in sequence
                    except KeyError:
                        pass
            assert len(animal_series) == self.trial_sequence_length
            animal_id_to_feature_series[animal_id] = pd.concat(animal_series, axis=1)

        result = pd.DataFrame.from_dict(animal_id_to_feature_series, orient="index")
        if self._animal_id_to_sequential_features:
            result = pd.concat((self._animal_id_to_sequential_features, result), axis=0)
        result.index.name = "Animal"

        return result

    # Helper methods =====================================

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
        def filter_trial_objects(trial_objects):
            if self.compute_first_two_feature_series_only:
                return trial_objects[:2]
            return trial_objects

        if not self.has_stages:
            return {self.trial_class.trial_label: filter_trial_objects(self.trial_objects)}

        if self.skip_habituation:
            if not self._first_trial_is_habituation:
                msg = (
                    f"skip_habituation is True, but the experiment has no habituation trial set. Possible mistakes:\n"
                    f"  - skip_habituation was set to True by mistake.\n"
                    f'  - users of ingress forgot to set {FIRST_TRIAL_IS_HABITUATION_INGRESS_FIELD} to "true" '
                    f"in the project settings.yaml file.\n"
                    f"  - advanced users did not run set_first_trial_to_habituation, "
                    f"an Experiment classmethod that is required for habituation inclusive workflows."
                )
                raise AttributeError(msg)
            logger.warning(
                "skip_habituation is True, skipping the the habituation class, thereby, the belonging trial object set"
            )

        return {
            self.trial_class_name_to_label[trial_class_name]: filter_trial_objects(trial_objects)
            for trial_class_name, trial_objects in self.trial_class_name_to_trial_objects.items()
            if not (self.skip_habituation and self.trial_class_name_to_label[trial_class_name] == "Habituation")
        }

    @cached_property
    def _class_labels(self):
        return tuple(trial_class.trial_label for trial_class in self.trial_sequence)

    @staticmethod
    def _neither_singular_trial_class_or_trial_id_to_trial_class_name(self):
        msg = (
            "Either trial_class has to be singularly defined, "
            "or trial_id_to_trial_class_name have to be exclusively defined"
        )
        raise AttributeError(msg)


Experiment = TypeVar("Experiment", bound=BaseExperiment)


def compute_trial_series_and_destroy_trial(trial_obj: Trial) -> pd.Series:
    result = trial_obj.trial_feature_series
    del globals()[trial_obj]
    return result
