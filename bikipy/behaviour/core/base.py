from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property, lru_cache
from operator import attrgetter
from time import sleep
from typing import Any, ClassVar, Hashable, Literal, Optional, Self

import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from pydantic import BaseModel, DirectoryPath, Field, FilePath, ValidationError
from pydantic.fields import FieldInfo, computed_field
from pydantic_numpy.typing import (
    Np2DArrayFp64,
    NpNDArray,
    NpNDArrayInt16,
    NpNDArrayUint8,
)
from tqdm import tqdm
from typing_inspect import is_generic_type
from yaspin import yaspin
from yaspin.spinners import Spinners

from bikipy import runtime_settings
from bikipy._constant import BIKIPY_ANALYSIS_VIDEO_PREFIX
from bikipy._dev_utils.fields import timestamp_index_field
from bikipy.behaviour.core.constant import ExperimentStage
from bikipy.core.base import BikipyHashable
from bikipy.core.mixin import AbstractFeatureCollectorMixin, InspectPlotMixin
from bikipy.core.typing import Label
from bikipy.core.video import VideoMetadata, VideoMetadataMixin
from bikipy.feature.motion import Motion, motion_analysis_indexer
from bikipy.feature.qualia.physical_object.trial_mixin import PhysicalObjectTrialMixin
from bikipy.perimeter import PERIMETER_CLASS_NAME_TO_CLASS
from bikipy.perimeter.base import BaseSinglePerimeter, PerimeterCLS, PerimeterSet
from bikipy.perimeter.trial_mixin import TrialWithPerimeterMixin
from bikipy.reader import READER_CLASS_LABEL_TO_CLASS
from bikipy.reader.base import BaseReader, ReaderCLS
from bikipy.utils.collection_utils import dict_deep_update
from bikipy.utils.memory import wait_for_more_physical_memory
from bikipy.utils.ranged_dict import RangeDict


class Behaviour(BikipyHashable, InspectPlotMixin, VideoMetadataMixin):
    pass


class BaseTrial(Behaviour, AbstractFeatureCollectorMixin):
    project_kit_config: dict[str, Any]

    framewise_coordinates_path: FilePath = Field(description="Path to file storing coordinate data")
    manual_reader_kwargs: Optional[dict] = Field(
        default_factory=dict, description="Keyword arguments that will be passed on the reader objects on init"
    )
    animal_id: Label = Field(description="The ID of the animal in the trial")
    animal_profile: Literal["rodent"] = "rodent"
    coordinate_timestamp_index: Optional[NpNDArray] = timestamp_index_field
    manual_center_pixels: Optional[NpNDArrayInt16] = None
    # enclosure: Optional[BasePerimeter] = enclosure_field

    # Class variables
    category = "trial"

    perimeter_labels: ClassVar[set[str]] = set()
    label_to_perimeter: ClassVar[dict[str, BaseSinglePerimeter] | None]

    # Variables for trials with zones, see doc for more info.
    trial_start_perimeter: Optional[str] = None

    # Label of the reader class to use for reading coordinate data
    reader_class_label: ClassVar[str] = "DeepLabCutReader"

    required_video_metadata_fields = {"meters_per_pixel", "resolution", "fps"}

    experiment_class_name: ClassVar[str]
    experiment_stage: ClassVar[ExperimentStage]
    trial_label: ClassVar[Optional[str]] = None

    second_tolerance: ClassVar[float] = 0.15

    @classmethod
    @property
    def fields_to_exclude_from_single_schema(cls) -> set[str]:
        result = super().fields_to_exclude_from_single_schema
        result.update(
            (
                "framewise_coordinates_path",
                "enclosure",
                "coordinate_timestamp_set_path",
                "animal_id",
                "analysis_series_cache_directory_path",
            )
        )
        return result

    @classmethod
    def experiment_class(cls) -> type[Self]:
        from bikipy.behaviour.mapping import experiment_name_to_class

        return experiment_name_to_class[cls.experiment_class_name]

    @classmethod
    def has_perimeter(cls) -> bool:
        from bikipy.behaviour.radial_arm.base import BaseRadialMazeTrial

        return is_generic_type(cls) or issubclass(cls, (TrialWithPerimeterMixin, BaseRadialMazeTrial))

    @classmethod
    @property
    def has_physical_object(cls) -> bool:
        return issubclass(cls, PhysicalObjectTrialMixin)

    @classmethod
    def perimeter_field_name_to_perimeter_class(cls) -> dict | None:
        if cls.has_perimeter():
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

    @classmethod
    @property
    def reader_class(cls) -> ReaderCLS:
        try:
            return READER_CLASS_LABEL_TO_CLASS[cls.reader_class_label]
        except KeyError as e:
            msg = (
                f"Invalid reader_class_label defined in Trial class, "
                f"{cls.reader_class_label}. Pick from: {tuple(READER_CLASS_LABEL_TO_CLASS)}"
            )
            raise AttributeError(msg) from e

    @classmethod
    @property
    def excel_sheet_name(cls) -> str:
        return (
            f"{cls.experiment_stage.value.capitalize()}{cls.trial_label.capitalize()}"
            if cls.trial_label
            else cls.experiment_stage.value.capitalize()
        )

    @computed_field(return_type=Np2DArrayFp64 | None)  # type: ignore[misc]
    @cached_property
    def manual_center_meters(self) -> Np2DArrayFp64 | None:
        if self.manual_center_pixels is not None:
            return self.manual_center_pixels * self.meters_per_pixel

    @computed_field(return_type=Np2DArrayFp64 | None)  # type: ignore[misc]
    @cached_property
    def center_meter_translation(self) -> Np2DArrayFp64 | None:
        if self.manual_center_meters is not None:
            return self.manual_center_meters - self.video.center_meters

    @computed_field  # type: ignore[misc]
    @property
    def _reader_kwargs(self) -> dict[str, Any]:
        return dict(
            df_path=self.framewise_coordinates_path,
            manual_timestamp_index=self.coordinate_timestamp_index,
            label=self.framewise_coordinates_path.stem,
            manual_video=self.video,
            enclosure=self.enclosure,
            **self.manual_reader_kwargs,
        )

    @computed_field(return_type=BaseReader)  # type: ignore[misc]
    @cached_property
    def reader(self) -> BaseReader:
        result = self.reader_class(**self._reader_kwargs)

        # In case the reader finds no time index, see fps property in reader
        if result.fps_from_timestamped_index:
            self.fps = result.fps_from_timestamped_index

        return result

    @computed_field  # type: ignore[misc]
    @property
    def number_of_frames(self) -> int:
        return self.reader.frames

    @computed_field  # type: ignore[misc]
    @cached_property
    def motion(self) -> Motion:
        return Motion(
            coordinate_sequence=self.reader.kinematic_coordinates,
            fps=self.fps,
        )

    def generate_inspection_video(
        self, output_directory: Optional[DirectoryPath] = None, *, codec: Optional[str] = None, **kwargs
    ) -> None:
        raise NotImplementedError

    def _video_file_name(
        self, output_directory: Optional[DirectoryPath] = None, context_label: Optional[str] = None
    ) -> FilePath:
        output_directory = output_directory or self.framewise_coordinates_path.parent

        stem_components = [BIKIPY_ANALYSIS_VIDEO_PREFIX]
        if self.experiment_stage:
            stem_components.append(self.experiment_stage.value)
        if self.label:
            stem_components.append(self.label)
        if context_label:
            stem_components.append(context_label)

        return output_directory / f"{'_'.join(stem_components)}.mp4"

    # Miscellaneous
    @computed_field  # type: ignore[misc]
    @property
    def _analysis_series_list(self) -> list[pd.Series]:
        return [self.reader.info, pd.Series(self.motion.as_tuple, index=motion_analysis_indexer("All", 2))]

    @computed_field  # type: ignore[misc]
    @cached_property
    def _uint_zeros_based_on_frame_length(self) -> NpNDArrayUint8:
        return np.zeros(self.number_of_frames, dtype=np.uint8)

    @computed_field  # type: ignore[misc]
    @cached_property
    def _frame_tolerance(self) -> int:
        return round(self.second_tolerance * self.fps)

    def _post_feature_collection_flush(self) -> None:
        self.reader.flush_reads()
        self.video.flush()


TrialCLS = type[BaseTrial]


class HabituationTrialMixin(BaseModel):
    experiment_stage: ClassVar[ExperimentStage] = ExperimentStage.HABITUATION


class BaseExperiment(Behaviour):
    trial_init_error_out_dir: DirectoryPath = Field(description="Directory to write Trial class init errors")

    manual_trial_ids: Optional[tuple] = None

    trial_id_to_trial_class_name: Optional[dict] = Field(default_factory=dict)
    trial_id_to_keyword_arguments: Optional[dict] = Field(default_factory=dict)
    trial_class_name_to_keyword_arguments: Optional[dict] = Field(default_factory=dict)
    trial_id_range_to_keyword_arguments: Optional[RangeDict] = Field(default_factory=RangeDict)
    common_trial_keyword_arguments: Optional[dict] = Field(default_factory=dict)

    stage: Optional[str] = Field(
        None, description="Experiment stage label, if experiment object is in a sequence of experiment objects"
    )
    skip_habituation: bool = Field(
        False, description="Skip the habituation class during analysis, practically skipping the the habituation class"
    )

    category = "experiment"

    experiment_labels: ClassVar[set[str]]

    # The trial class that will be used in case set_first_trial_to_habituation is called
    habituation_trial_class: ClassVar[Optional[TrialCLS]] = None

    first_trial_is_habituation: ClassVar[bool] = False

    # "Sequence of trial classes designed for the experiment class"
    trial_sequence: ClassVar[tuple[TrialCLS, ...]]

    def __getitem__(self, item: int):
        return self.trial_id_to_trial_object[item]

    @classmethod
    @property
    def fields_to_exclude_from_single_schema(cls) -> set[str]:
        result = super().fields_to_exclude_from_single_schema
        result.update(
            (
                "trial_id_to_trial_class_name",
                "trial_id_to_keyword_arguments",
                "trial_class_name_to_keyword_arguments",
                "trial_id_range_to_keyword_arguments",
                "common_trial_keyword_arguments",
            )
        )
        return result

    @classmethod
    @property
    def trial_classes(cls) -> OrderedSet[TrialCLS]:
        """All trials designed for the experiment class"""
        return OrderedSet(cls.trial_sequence)

    @classmethod
    def set_first_trial_to_habituation(cls) -> type[Self]:
        if not cls.habituation_trial_class:
            msg = f"Habituation trial class for {cls.__name__} has not been defined, contact the maintainers"
            raise AttributeError(msg)
        if cls.first_trial_is_habituation:
            msg = "The first trial has already been set to habituation"
            raise AttributeError(msg)

        cls.habituation_trial_class.experiment_class_name = cls.__name__
        cls.trial_sequence = (cls.habituation_trial_class, *cls.trial_sequence)
        cls.first_trial_is_habituation = True

        return cls

    @classmethod
    def trial_sequence_repetition(cls, repetitions: int) -> type[Self]:
        cls.trial_sequence = tuple(cls.trial_classes) * repetitions
        return cls

    @classmethod
    def set_custom_trial_sequence(cls, custom_trial_sequence: tuple[TrialCLS | str, ...]) -> type[Self]:
        from bikipy.behaviour.mapping import resolve_trial

        new_sequence = []
        for trial_cls in custom_trial_sequence:
            trial_cls = resolve_trial(trial_cls, cls.__name__)
            assert trial_cls in cls.trial_classes, (
                f"The new sequence must consist of classes that are defined for the experiment ({cls.__name__}), "
                f"{trial_cls} is not included"
            )
            new_sequence.append(trial_cls)

        cls.trial_sequence = tuple(new_sequence)
        return cls

    @classmethod
    @property
    def at_least_one_trial_has_perimeter(cls) -> bool:
        return any(trial_class.has_perimeter() for trial_class in cls.trial_classes)

    @classmethod
    @property
    def at_least_one_trial_has_physical_object(cls) -> bool:
        return any(trial_class.has_physical_object for trial_class in cls.trial_classes)

    @classmethod
    @property
    def trial_perimeter_label_to_perimeter_class(cls) -> dict[str, PerimeterCLS]:
        result = {}
        for trial_class in cls.trial_classes:
            if not trial_class.perimeter_field_name_to_perimeter_class():
                continue

            for label, perimeter_class in trial_class.perimeter_field_name_to_perimeter_class().items():
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
        return tuple(trial_class.experiment_stage for trial_class in cls.trial_sequence)

    @classmethod
    @property
    def trial_class_name_to_label(cls) -> dict:
        return dict(zip(cls.trial_class_names, cls.trial_class_labels))
        # return {name: label for name, label in zip(cls.trial_class_names, cls.trial_class_labels)}

    @classmethod
    @property
    def trial_class(cls) -> BaseTrial:
        if cls.has_stages:
            msg = f"{cls.__name__}: trial_class attribute can only be utilized when there is only one Trial class"
            raise AttributeError(msg)
        return cls.trial_sequence[0]

    @classmethod
    @property
    def habituation_trial_class_name(cls) -> str:
        return cls.habituation_trial_class.__name__

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

    _trial_objects: list[BaseTrial] = []
    _trial_id_to_trial_object: dict[Label, BaseTrial] = {}
    _bad_trial_ids_to_error_msg: dict[Label, str] = {}
    __analysed_trials: set[Label] = set()

    def get_trial_object(self, trial_id: Label) -> BaseTrial:
        if trial_id in self.__analysed_trials:
            return self._trial_id_to_trial_object[trial_id]

        trial_class = (
            self.trial_class_name_to_trial_class[self.trial_id_to_trial_class_name[trial_id]]
            if self.has_stages
            else self.trial_class
        )
        try:
            result = trial_class(**self.trial_keyword_arguments(trial_id))
            # TODO: Pydantic v2
            # if cached_instance := result.load_self_from_cache():
            #     # We replace the new instance with the cached instances, saving compute
            #     result = cached_instance

            self._trial_objects.append(result)
            self._trial_id_to_trial_object[trial_id] = result

            return result
        except ValidationError as e:
            self._bad_trial_ids_to_error_msg[trial_id] = str(e)
        finally:
            self.__analysed_trials.add(trial_id)

    @computed_field  # type: ignore[misc]
    @cached_property
    def trial_objects(self) -> list[BaseTrial]:
        for trial_id in self.trial_id_set.difference(self.__analysed_trials):
            self.get_trial_object(trial_id)

        if self._bad_trial_ids_to_error_msg:
            msg = "Some trial IDs yielded pydantic validation errors:"
            for trial_id, trial_msg in self._bad_trial_ids_to_error_msg.items():
                msg += f"\n{trial_id}:\n{trial_msg}\n"
            if self.trial_init_error_out_dir:
                error_out_path = self.trial_init_error_out_dir / "trial_init_error.log"
                with open(error_out_path, "w") as out_file:
                    out_file.write(msg)
                    msg += f"\nError logs were saved to {error_out_path}\n"
            raise ValueError(
                f"{msg}\n{len(self._bad_trial_ids_to_error_msg)} errors out of {len(self.trial_ids)} Trials"
            )

        return self._trial_objects

    @computed_field  # type: ignore[misc]
    @cached_property
    def trial_class_name_to_trial_ids(self) -> dict[str, BaseTrial]:
        return {
            trial_class.__name__: trial_ids for trial_class, trial_ids in self._trial_class_name_to_trial_ids.items()
        }

    @computed_field  # type: ignore[misc]
    @cached_property
    def trial_id_to_trial_object(self) -> dict[Label, BaseTrial]:
        assert self.trial_objects
        return self._trial_id_to_trial_object

    @computed_field  # type: ignore[misc]
    @cached_property
    def trial_class_to_trial_objects(self) -> dict[TrialCLS, list[BaseTrial]]:
        assert not self.skip_habituation or (self.skip_habituation and self.first_trial_is_habituation), (
            "skip_habituation is True, but the experiment has no habituation trial set. Possible mistakes:\n"
            "  - skip_habituation was set to True by mistake.\n"
            '  - Forgot to set ingress.first_stage_is_habituation to "True" '
            "in the project settings.yaml file.\n"
            "  - advanced users did not run set_first_trial_to_habituation, "
            "an Experiment classmethod that is required for habituation inclusive workflows."
        )

        if not self.has_stages:
            return {self.trial_classes[0]: self.trial_objects}

        return {
            self.trial_class_name_to_trial_class[trial_class_name]: [
                self.trial_id_to_trial_object[trial_id] for trial_id in trial_ids
            ]
            for trial_class_name, trial_ids in self._trial_class_name_to_trial_ids.items()
            if not self.skip_habituation
            or (self.skip_habituation and trial_class_name != self.habituation_trial_class_name)
        }

    @computed_field  # type: ignore[misc]
    @cached_property
    def animal_id_to_trial_objects(self) -> dict[Hashable, BaseTrial]:
        result = {}
        for trial in self.trial_objects:
            if trial.animal_id in result:
                result[trial.animal_id].append(trial)
            else:
                result[trial.animal_id] = [trial]
        for trials in result.values():
            trials.sort(key=lambda t: t.label)
        return dict(sorted(result.items()))

    @computed_field  # type: ignore[misc]
    @cached_property
    def animal_id_to_trial_ids(self) -> dict[Label, list[Label]]:
        return {
            animal_id: [trial_object.int_id for trial_object in trial_objects]
            for animal_id, trial_objects in self.animal_id_to_trial_objects.items()
        }

    @computed_field  # type: ignore[misc]
    @cached_property
    def trial_id_to_animal_id(self) -> dict[Label, Label]:
        result = {trial_id: kwargs["animal_id"] for trial_id, kwargs in self.trial_id_to_keyword_arguments.items()}
        return dict(sorted(result.items(), key=lambda item: item[1]))

    @computed_field  # type: ignore[misc]
    @cached_property
    def trial_ids(self) -> tuple:
        if self.manual_trial_ids:
            result = self.manual_trial_ids
        elif not self.has_stages and self.trial_class:
            result = tuple(self.trial_id_to_keyword_arguments)
        elif self.trial_id_to_trial_class_name:
            result = tuple(self.trial_id_to_trial_class_name)
        else:
            msg = (
                "Either trial_class has to be singularly defined, "
                "or trial_id_to_trial_class_name have to be exclusively defined"
            )
            raise AttributeError(msg)

        assert result, "No trials were found"
        first_trial_id = result[0]
        first_type = type(first_trial_id)

        if not all(isinstance(trial_id, first_type) for trial_id in result):
            msg = "The Trial IDs must have the same type"
            raise AttributeError(msg)

        return result

    @computed_field  # type: ignore[misc]
    @cached_property
    def trial_id_set(self) -> frozenset[Label]:
        return frozenset(self.trial_ids)

    @computed_field  # type: ignore[misc]
    @cached_property
    def animals_ids(self) -> set:
        return set(self.animal_id_to_trial_objects)

    @computed_field  # type: ignore[misc]
    @cached_property
    def number_of_trials(self) -> int:
        return len(self.trial_ids)

    @computed_field  # type: ignore[misc]
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

    def analyze_trials(self):
        assert self.trial_class_to_trial_analysis_series

    # DataFrame methods =========================================
    @computed_field  # type: ignore[misc]
    @cached_property
    def trial_class_to_trial_analysis_series(self) -> dict:
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
                        for trial_object in trial_objects:
                            wait_for_more_physical_memory()
                            result[trial_class][trial_object.label] = executor.submit(
                                attrgetter("analysis_series"), trial_object
                            )
                            sleep(2)

                for trial_class, trial_objects in self.trial_class_to_trial_objects.items():
                    for trial_object in trial_objects:
                        result[trial_class][trial_object.label] = result[trial_class][trial_object.label].result()

        return dict(result)

    @computed_field  # type: ignore[misc]
    @cached_property
    def _animal_id_to_sequential_features(self) -> pd.DataFrame | None:
        if self._animal_id_to_sequential_features_list:
            # Concatenate and reverse the order
            return pd.concat(self._animal_id_to_sequential_features_list[::-1], axis=0)

    @computed_field  # type: ignore[misc]
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

        result = {}

        if self.common_trial_keyword_arguments:
            dict_deep_update(result, self.common_trial_keyword_arguments)

        if (
            self.has_stages
            and (trial_class_name := self.trial_id_to_trial_class_name[trial_id])
            in self.trial_class_name_to_keyword_arguments
        ):
            dict_deep_update(result, self.trial_class_name_to_keyword_arguments[trial_class_name])

        if self.trial_id_to_keyword_arguments:
            dict_deep_update(result, self.trial_id_to_keyword_arguments[trial_id])

        if self.trial_id_range_to_keyword_arguments:
            if not isinstance(trial_id, int):
                msg = "Trial IDs must be integers when trial_id_range_to_keyword_arguments is used"
                raise AttributeError(msg)
            dict_deep_update(result, self.trial_id_range_to_keyword_arguments[trial_id])

        assert result["framewise_coordinates_path"]

        if "animal_id" not in result:
            result["animal_id"] = trial_id

        result["inspection_fig_output_path"] = self.inspection_fig_output_path

        if "label_to_perimeter" in result:
            dict_deep_update(result, result.pop("label_to_perimeter"))

        if "perimeter_set" in result:
            perimeter_set: PerimeterSet = result.pop("perimeter_set")
            dict_deep_update(result, perimeter_set.label_to_perimeter)

        if PluginRadial.default_trial_argument_key in result:
            perimeter_set_group: dict = result.pop(PluginRadial.default_trial_argument_key)
            dict_deep_update(result, perimeter_set_group)

        if PluginChangeReference.default_trial_argument_key in result:
            val = result.pop(PluginChangeReference.default_trial_argument_key)
            if isinstance(val, PerimeterSet):
                dict_deep_update(result, val.label_to_perimeter)
            elif isinstance(val, BaseSinglePerimeter):
                result[val.label] = val
            elif isinstance(val, dict):
                dict_deep_update(result, val)
            else:
                msg = f"Unsupported type for PluginChangeReference: {type(val)}"
                raise AttributeError(msg)

        if "manual_video" in result:
            result.update(
                VideoMetadata.join(
                    result.pop("manual_video"),
                    self.video,
                    ignore_incongruity=True,
                    # TODO: Replace after computed_field exclude method added to model_dump
                ).metadata
            )
        else:
            # TODO: Replace after computed_field exclude method added to model_dump
            result.update(self.video.metadata)

        return result

    @computed_field  # type: ignore[misc]
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
    @computed_field  # type: ignore[misc]
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

    @computed_field  # type: ignore[misc]
    @cached_property
    def _class_labels(self) -> tuple[ExperimentStage, ...]:
        return tuple(trial_class.experiment_stage for trial_class in self.trial_sequence)

    def save(self, **kwargs):
        self.analyze_trials()
        super().save(**kwargs)


ExperimentCLS = type[BaseExperiment]
