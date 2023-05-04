from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property, lru_cache
from logging import getLogger
from operator import attrgetter
from typing import Any, ClassVar, Hashable, Literal, Optional, Type, TypeVar

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    ValidationError,
    validator,
)
from pydantic.fields import FieldInfo
from pydantic_numpy import NDArray
from pydantic_numpy.dtype import NDArrayFp64, NDArrayInt16, NDArrayUint8
from tqdm import tqdm
from typing_inspect import is_generic_type
from yaspin import yaspin
from yaspin.spinners import Spinners

from bikipy import runtime_settings
from bikipy._dev_utils.fields import enclosure_field, timestamp_index_field
from bikipy.core.base import BikipyHashable
from bikipy.core.mixin import AbstractFeatureCollectorMixin, InspectPlotMixin
from bikipy.core.typing import Label
from bikipy.core.video import VideoMetadataMixin
from bikipy.feature.motion import Motion, motion_multi_indexer
from bikipy.feature.physical_object.mixin import PhysicalObjectTrialMixin
from bikipy.perimeter import PERIMETER_CLASS_NAME_TO_CLASS
from bikipy.perimeter.base import (
    BaseSinglePerimeter,
    Perimeter,
    PerimeterCLS,
    PerimeterSet,
    SinglePerimeter,
)
from bikipy.perimeter.mixin import TrialWithPerimeterMixin
from bikipy.reader import READER_CLASS_LABEL_TO_CLASS
from bikipy.reader.base import Reader, ReaderCLS
from bikipy.reader.data_with_likelihood import DeepLabCutReader
from bikipy.utils.ranged_dict import RangeDict

LABEL_to_DATA_READER = {"deeplabcut": DeepLabCutReader}

logger = getLogger(__name__)


class Behaviour(BikipyHashable, InspectPlotMixin, VideoMetadataMixin):
    pass


class BaseTrial(Behaviour, AbstractFeatureCollectorMixin):
    framewise_coordinates_path: FilePath = Field(..., description="Path to file storing coordinate data")
    manual_reader_kwargs: Optional[dict] = Field(
        default_factory=dict, description="Keyword arguments that will be passed on the reader objects on init"
    )
    animal_id: Label = Field(..., description="The ID of the animal in the trial")
    animal_profile: Literal["rodent"] = "rodent"
    coordinate_timestamp_index: Optional[NDArray] = timestamp_index_field
    manual_center_pixels: Optional[NDArrayInt16]
    enclosure: Optional[Perimeter] = enclosure_field
    crop_time_seconds: float = 0.0
    crop_from_end: bool = Field(
        False,
        description="Only affective if crop_time_seconds is not 0.0. Will crop from start instead when set to False",
    )

    reader_class_label: str = Field(
        "DeepLabCutReader", description="Label of the reader class to use for reading coordinate data"
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
    excel_sheet_name: ClassVar[str] = ...

    second_tolerance: ClassVar[float] = 0.15

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(
            (
                "framewise_coordinates_path",
                "enclosure",
                "coordinate_timestamp_set_path",
                "animal_id",
            )
        )
        return upstream

    @classmethod
    @property
    def experiment_class(cls) -> "ExperimentCLS":
        from bikipy.behaviour.mapping import experiment_name_to_class

        return experiment_name_to_class[cls.experiment_class_name]

    @classmethod
    @property
    def has_perimeter(cls) -> bool:
        return is_generic_type(cls) or issubclass(cls, TrialWithPerimeterMixin)

    @classmethod
    @property
    def has_physical_object(cls) -> bool:
        return issubclass(cls, PhysicalObjectTrialMixin)

    @classmethod
    @property
    def perimeter_field_name_to_perimeter_class(cls) -> dict | None:
        if cls.has_perimeter:
            result = {}
            schema = cls.schema()["properties"]
            for perimeter_label in cls.perimeter_labels:
                try:
                    field_schema = schema[perimeter_label]
                except KeyError as e:
                    msg = (
                        f"{perimeter_label} is defined as a perimeter label "
                        f"yet it is not a field in the class {cls.__name__}"
                    )
                    raise AttributeError(msg) from e

                perimeter_class_name = field_schema["$ref"].split("/")[-1]  # hacky pydantic-oriented solution
                result[perimeter_label] = PERIMETER_CLASS_NAME_TO_CLASS[perimeter_class_name]

            return result

    @property
    def reader_class(self) -> ReaderCLS:
        try:
            return READER_CLASS_LABEL_TO_CLASS[self.reader_class_label]
        except KeyError as e:
            msg = (
                f"Invalid reader_class_label defined in Trial class, "
                f"{self.reader_class_label}. Pick from: {tuple(READER_CLASS_LABEL_TO_CLASS)}"
            )
            raise AttributeError(msg) from e

    @property
    def perimeter_to_derive_meters_per_pixel(self) -> Perimeter:
        if self.manual_perimeter_to_derive_meters_per_pixel:
            try:
                return self._label_to_perimeter[self.manual_perimeter_to_derive_meters_per_pixel]
            except TypeError:
                # self._label_to_perimeter is None -> TypeError
                msg = (
                    f"The class, {self.__class__.__name__}, does not define _label_to_perimeter, "
                    f"which makes the mapping of manual_perimeter_to_derive_meters_per_pixel "
                    f"to a Perimeter object impossible"
                )
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
            "timestamp_index": self.coordinate_timestamp_index,
            "label": self.framewise_coordinates_path.stem,
            "manual_video": self.video,
            "crop_time_seconds": self.crop_time_seconds,
            "enclosure": self.enclosure,
            **self.manual_reader_kwargs,
        }

    @cached_property
    def reader(self) -> Reader:
        result = self.reader_class(**self._reader_kwargs)

        # In case the reader finds no time index, see fps property in reader
        if result.fps:
            self.fps = result.fps

        return result

    @property
    def number_of_frames(self) -> int:
        return self.reader.frames

    @cached_property
    def experiment_seconds(self) -> int:
        return self.reader.kinematic_coordinates.shape[0] / self.video.fps

    @cached_property
    def motion(self) -> Motion:
        return Motion(
            coordinate_sequence=self.reader.kinematic_coordinates,
            fps=self.video.fps,
        )

    # Miscellaneous
    @property
    def _analysis_series_list(self) -> list[pd.Series, ...]:
        return [self.reader.info, pd.Series(self.motion.as_tuple, index=motion_multi_indexer("All", 2))]

    @cached_property
    def _uint_zeros_based_on_frame_length(self) -> NDArrayUint8:
        return np.zeros(self.number_of_frames, dtype=np.uint8)

    @cached_property
    def _frame_tolerance(self) -> int:
        return round(self.second_tolerance * self.video.fps)

    def _post_feature_collection_flush(self) -> None:
        self.reader.flush_reads()
        self.video.flush()


TrialCLS = Type[BaseTrial]
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

    habituation_trial_class: ClassVar[Optional[TrialCLS]] = Field(
        description="The trial class that will be used in case set_first_trial_to_habituation is called"
    )
    _first_trial_is_habituation: ClassVar[bool] = False

    # "Sequence of trial classes designed for the experiment class"
    trial_sequence: ClassVar[tuple[TrialCLS, ...]] = ...

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
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(
            (
                "trial_id_to_trial_class_name",
                "trial_id_to_keyword_arguments",
                "trial_class_name_to_keyword_arguments",
                "trial_id_range_to_keyword_arguments",
                "common_trial_keyword_arguments",
            )
        )
        return upstream

    @classmethod
    @property
    def trial_classes(cls) -> set[TrialCLS]:
        """All trials designed for the experiment class"""
        return set(cls.trial_sequence)

    @classmethod
    def trial_sequence_repetition(cls, repetitions: int) -> "ExperimentCLS":
        cls.trial_sequence = tuple(cls.trial_classes) * repetitions
        return cls

    @classmethod
    def set_first_trial_to_habituation(cls) -> "ExperimentCLS":
        if not cls.habituation_trial_class:
            msg = f"Habituation trial class for {cls.__name__} has not been defined, contact the maintainers"
            raise AttributeError(msg)
        if cls._first_trial_is_habituation:
            msg = "The first trial has already been set to habituation"
            raise AttributeError(msg)

        cls.habituation_trial_class.experiment_class_name = cls.__name__
        cls.trial_sequence = (cls.habituation_trial_class, *cls.trial_sequence)
        cls._first_trial_is_habituation = True

        return cls

    @classmethod
    @property
    def at_least_one_trial_has_perimeter(cls) -> bool:
        return any(trial_class.has_perimeter for trial_class in cls.trial_classes)

    @classmethod
    @property
    def at_least_one_trial_has_physical_object(cls) -> bool:
        return any(trial_class.has_physical_object for trial_class in cls.trial_classes)

    @classmethod
    @property
    def trial_perimeter_label_to_perimeter_class(cls) -> dict[str, PerimeterCLS]:
        result = {}
        for trial_class in cls.trial_classes:
            if not trial_class.perimeter_field_name_to_perimeter_class:
                continue

            for label, perimeter_class in trial_class.perimeter_field_name_to_perimeter_class.items():
                if label in result:
                    assert result[label] == perimeter_class
                    continue
                result[label] = perimeter_class

        return result

    @classmethod
    @property
    def trial_class_name_to_stage_index(cls) -> dict[str, int]:
        return {trial_class.__name__: i for i, trial_class in enumerate(cls.trial_sequence)}

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
    def stage_index_to_trial_class(cls) -> dict[int, TrialCLS]:
        if not cls.has_stages:
            msg = f"{cls.__name__}: stage_index_to_trial_class is undefined in non-sequential experiment classes"
            raise AttributeError(msg)

        try:
            return {i: trial_class for i, trial_class in enumerate(cls.trial_sequence)}
        except AttributeError:
            msg = (
                "experiment_stage_index must be defined for each trial "
                "class when working with a sequence of trial classes"
            )
            raise AttributeError(msg)

    @classmethod
    @property
    def stage_index_to_trial_class_name(cls) -> dict[int, TrialCLS]:
        if isinstance(tuple(cls.stage_index_to_trial_class.values())[-1], FieldInfo):
            msg = (
                f"To the developers: Setting the trial_sequence class variable is required; "
                f"please do so for {cls.__name__}"
            )
            raise AttributeError(msg)
        return {i: trial_class.__name__ for i, trial_class in cls.stage_index_to_trial_class.items()}

    @classmethod
    @property
    def trial_class_name_to_trial_class(cls) -> dict[str, TrialCLS]:
        if not cls.has_stages:
            msg = (
                f"{cls.__name__}: trial_class_name_to_trial_class attribute can only be utilized when "
                f"there are many Trial classes"
            )
            raise AttributeError(msg)

        try:
            return {trial_class.__name__: trial_class for trial_class in cls.trial_sequence}
        except AttributeError:
            msg = (
                "experiment_stage_index must be defined for each trial class when "
                "working with a sequence of trial classes"
            )
            raise AttributeError(msg)

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
            msg = "Some trial IDs yielded pydantic validation errors:"
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
    def trial_class_to_trial_objects(self) -> dict[str, Trial]:
        return (
            {
                self.trial_class_name_to_trial_class[trial_class_name]: [
                    self.trial_id_to_trial_object[trial_id] for trial_id in trial_ids
                ]
                for trial_class_name, trial_ids in self._trial_class_name_to_trial_ids.items()
            }
            if self.has_stages
            else {self.trial_class: self.trial_objects}
        )

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

    @cached_property
    def trial_class_to_trial_analysis_series(self):
        result = defaultdict(dict)
        if runtime_settings.disable_process_pooling:
            for trial_class, trial_objects in tqdm(
                self.trial_class_to_trial_objects.items(), desc="Computing experiment features"
            ):
                result[trial_class] = {
                    trial_object.label: trial_object.analysis_series for trial_object in trial_objects
                }

        else:
            with yaspin(Spinners.pong, text="Computing experiment features..."):
                with ProcessPoolExecutor(max_workers=runtime_settings.max_workers_in_process_pool) as executor:
                    for trial_class, trial_objects in self.trial_class_to_trial_objects.items():
                        result[trial_class] = {
                            trial_object.label: features
                            for trial_object, features in zip(
                                trial_objects, executor.map(attrgetter("analysis_series"), trial_objects)
                            )
                        }

        return dict(result)

    def analyze_trials(self):
        assert self.trial_class_to_trial_analysis_series

    @cached_property
    def _animal_id_to_sequential_features(self) -> pd.DataFrame | None:
        if self._animal_id_to_sequential_features_list:
            # Concatenate and reverse the order
            return pd.concat(self._animal_id_to_sequential_features_list[::-1], axis=0)

    @cached_property
    def trial_label_to_df(self) -> dict[Label, pd.DataFrame]:
        result = {}
        for trial_class, data_dict in self.trial_class_to_trial_analysis_series.items():
            result[trial_class.excel_sheet_name] = pd.DataFrame.from_dict(data_dict, orient="index")
            result[trial_class.excel_sheet_name].index.name = "Trial ID"

        if self._animal_id_to_sequential_features:
            result["AnimalToSequential"] = self._animal_id_to_sequential_features

        return result

    def trial_keyword_arguments(self, trial_id: Label) -> dict[str, Any]:
        """
        Function useful for customizing initiation parameters for trial objects
        """
        from bikipy.ingress.plugin.perimeter.change_reference import (
            PluginChangeReference,
        )
        from bikipy.ingress.plugin.perimeter.radial_maze import PluginRadial

        result = self.video.dict(exclude_unset=True)

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

        if PluginRadial.default_trial_argument_key in result:
            perimeter_set_group: dict = result.pop(PluginRadial.default_trial_argument_key)
            result.update(perimeter_set_group)

        if PluginChangeReference.default_trial_argument_key in result:
            val = result.pop(PluginChangeReference.default_trial_argument_key)
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
    def animal_id_indexed_feature_df(self) -> pd.DataFrame:
        @lru_cache(self.trial_sequence_length)
        def add_trial_class_to_top_of_multi_index():
            return pd.MultiIndex.from_product([[trial_class_name], list(series.index)])

        animal_id_to_feature_series = {}
        for animal_id, trial_ids in self.animal_id_to_trial_ids.items():
            animal_series = []
            for trial_class_name in self.trial_class_names:
                trial_class_related_feature_series_data = self.trial_class_to_trial_analysis_series[trial_class_name]
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
    def _trial_class_to_trial_objects(self) -> dict:
        def filter_trial_objects(trial_objects):
            if self.compute_first_two_feature_series_only:
                return trial_objects[:2]
            return trial_objects

        if not self.has_stages:
            return {self.trial_class: filter_trial_objects(self.trial_objects)}

        if self.skip_habituation:
            if not self._first_trial_is_habituation:
                msg = (
                    "skip_habituation is True, but the experiment has no habituation trial set. Possible mistakes:\n"
                    "  - skip_habituation was set to True by mistake.\n"
                    '  - Forgot to set ingress.first_stage_is_habituation to "True" '
                    "in the project settings.yaml file.\n"
                    "  - advanced users did not run set_first_trial_to_habituation, "
                    "an Experiment classmethod that is required for habituation inclusive workflows."
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

    def save(self, **kwargs):
        self.analyze_trials()
        super().save(**kwargs)


ExperimentCLS = Type[BaseExperiment]
Experiment = TypeVar("Experiment", bound=BaseExperiment)
