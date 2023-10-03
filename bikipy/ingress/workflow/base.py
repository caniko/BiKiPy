import os
import pstats
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
from cProfile import Profile
from functools import cached_property, partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Optional, TypeVar

import numpy as np
import pandas as pd
from inflection import underscore
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    computed_field,
    validate_call,
)
from pydantic_numpy import NpNDArrayFp64
from schemantic import SchemanticProjectModelMixin

from bikipy import runtime_settings
from bikipy._constant import (
    ANALYSIS_CACHE_STEM_ID,
    AUGMENTED_COORDINATE_CACHED_FILE_LABEL,
    BIKIPY_ANALYSIS_VIDEO_PREFIX,
    READER_MAP_NAME,
    TRIAL_MAP_NAME,
)
from bikipy.behaviour.core.base import BaseExperiment, ExperimentCLS, TrialCLS
from bikipy.core.base import BikipyConfigModel
from bikipy.core.typing import Label
from bikipy.ingress.plugin.core.plugin_scope import PluginScope
from bikipy.ingress.plugin.perimeter.constant import LABEL_TO_TRIAL_SHEET_NAME
from bikipy.ingress.utils.io import (
    get_inspect_directory_path,
    get_plugin_directory_path,
    get_project_settings_path,
    infer_metadata_path,
    result_directory_path,
)
from bikipy.perimeter.base import BasePerimeter
from bikipy.utils.collection_utils import get_first_value_in_dict
from bikipy.utils.constants import TO_PARQUET_KWARGS
from bikipy.utils.misc import sheet_names_from_path
from bikipy.utils.pandas import copycat_assumes_levels_of_icon

if TYPE_CHECKING:
    from bikipy.ingress.plugin.core.base import BasePlugin, PluginType


defaultdict_dict = partial(defaultdict, dict)


class PluginDefinitions(BaseModel):
    meters_per_pixel: frozenset[PluginScope]
    perimeter: Optional[frozenset[PluginScope]] = None
    enclosure: Optional[frozenset[PluginScope]] = None
    radial: Optional[frozenset[PluginScope]] = None
    change_reference: Optional[frozenset[PluginScope]] = None
    frame: Optional[frozenset[PluginScope]] = None
    video: Optional[frozenset[PluginScope]] = None
    timestamp: Optional[frozenset[PluginScope]] = None
    center: Optional[frozenset[PluginScope]] = None

    def __iter__(self):
        """So `dict(model)` works."""
        yield from [(k, v) for (k, v) in self.__dict__.items() if v is not None and not k.startswith("_")]
        extra = self.__pydantic_extra__
        if extra:
            yield from extra.items()


class BaseIngressWorkflow(BikipyConfigModel, SchemanticProjectModelMixin, ABC):
    """
    This model stores methods to ingest data for bikipy-based analysis. The workflow differs slightly between daughter
    classes. The commonality are the levels in which data is introduced, which is quite similar to the bikipy experiment
    class:
        - Common trial keyword arguments are defined by the settings.yaml file, or from global plugin values defined in
        the plugin_files folder. Note that each plugin file may have their values assigned to trial IDs with mappers
        defined in the metadata files; the values are assigned to their respective trial ID in this case.

        - Trial class name keyword arguments are defined in settings.yaml

        - Trial ID keyword arguments are defined by a combination of the ingress class, the settings, and metadata.

    This data is passed onto the experiment class, which is the runner of the analysis. The best method to initiate
    analysis is to run analyze_and_save(), a function at the bottom of this file, through the BiKiPy CLI.
    """

    project_directory: DirectoryPath
    dataset_directory: DirectoryPath

    experiment_class_name: str
    project_kit_config: dict[str, Any]

    create_inspection_plots: bool = True

    first_stage_is_habituation: bool = False
    metadata_trial_ids_are_higher_level: bool = False
    custom_trial_sequence: tuple[str, ...] = Field(default_factory=tuple)
    trial_sequence_loops: int = 1
    only_one_instance_of_trial_class: bool = Field(
        False,
        description="When troubleshooting a runtime, avoid ingesting all data, "
        "and only focus on one of each trial class",
    )
    trial_ids_to_analyse: frozenset[Label] = Field(default_factory=frozenset)

    no_cache: bool = False

    plugin_definitions: PluginDefinitions

    ingress_defined_perimeters: dict[str, BasePerimeter] = Field(default_factory=dict)

    lazy_dev_mode: bool = Field(
        False,
        description="Quality of life improvements for the lazy developer. "
        "Currently changes the name of the inspection if there is an exception during analysis",
    )
    no_inspection: bool = False

    trial_id_to_trial_class_name: dict[Label, str] = Field(default_factory=dict)
    common_trial_keyword_arguments: dict[str, Any] = Field(default_factory=dict)
    trial_id_to_keyword_arguments: dict[Label, dict[str, Any]] = Field(default_factory=defaultdict_dict)
    trial_class_name_to_keyword_arguments: dict[str, dict[str, Any]] = Field(default_factory=defaultdict_dict)

    trial_id_to_designator_id: dict[str, str] = Field(default_factory=dict)
    designator_id_to_kwargs: dict[str, dict] = Field(default_factory=defaultdict_dict)

    ingress_method: ClassVar[str]
    coordinate_file_index_delimiter: ClassVar[str] = "."
    profile_runtime: ClassVar[bool] = True

    @abstractmethod
    def _dataset_reader(self) -> None:
        """
        This component reads the contents of the dataset folder. It is an abstractmethod because projects often have
        different layouts. Refer to submodules in the same directory as this module
        to explore the different implementations
        """
        ...

    def model_post_init(self, __context: Any) -> None:
        self.common_trial_keyword_arguments["project_kit_config"] = self.project_kit_config

        if not self.no_cache:
            self.common_trial_keyword_arguments["analysis_series_cache_directory_path"] = self.cache_directory_path

        for plugin_model in self._global_plugins:
            first_file = next(self.plugin_directory_path.glob(f"{plugin_model.code_key}*"))
            self.common_trial_keyword_arguments[plugin_model.default_trial_argument_key] = self._define_plugin(
                plugin_model, PluginScope.GLOBAL, data_path=first_file
            ).globally_defined

        metadata_trial_target_dict = (
            self.designator_id_to_kwargs
            if self.metadata_trial_ids_are_higher_level
            else self.trial_id_to_keyword_arguments
        )

        for plugin_model in self._plugins_metadata:
            label_to_file_path = {
                file_path.stem.split("-")[-1]: file_path
                for file_path in self.plugin_directory_path.glob(f"{plugin_model.code_key}*")
            }

            for trial_id, row in self.metadata_plugin_to_correct_sheet(plugin_model).iterrows():
                if self._to_skip_trial_id(trial_id):
                    continue

                if plugin_model.human_readable_index in row:
                    trial_id_plugin_label = row[plugin_model.human_readable_index]
                    if isinstance(trial_id_plugin_label, float) and np.isnan(trial_id_plugin_label):
                        continue

                    metadata_trial_target_dict[trial_id][plugin_model.default_trial_argument_key] = self._define_plugin(
                        plugin_model,
                        PluginScope.METADATA,
                        data_path=label_to_file_path[str(trial_id_plugin_label)],
                        **self.get_plugin_config(plugin_model),
                    ).trialwise_and_metadata(trial_id)

                elif plugin_model.plural_entries:
                    for key, trial_id_plugin_label in row.items():
                        if isinstance(trial_id_plugin_label, float) and np.isnan(trial_id_plugin_label):
                            continue

                        if plugin_model.human_readable_index in key:
                            metadata_trial_target_dict[trial_id][underscore(key)] = self._define_plugin(
                                plugin_model,
                                PluginScope.METADATA,
                                data_path=label_to_file_path[str(trial_id_plugin_label)],
                                manual_trial_argument_key=underscore(key),
                                ingress=self,
                                **self.get_plugin_config(plugin_model),
                            ).trialwise_and_metadata(trial_id, naive=True)

                else:
                    msg = (
                        f"{plugin_model.human_readable_index} is active in the settings, but is missing in the metadata"
                    )
                    raise ValueError(msg)

        if self.experiment_class.has_stages:
            for trial_class_name in self.experiment_class.trial_class_names:
                assert (
                    trial_class_name in self.project_kit_config[TRIAL_MAP_NAME]
                ), "Couldn't find trial class name in configuration file under trial"
                self.trial_class_name_to_keyword_arguments[trial_class_name] = self.project_kit_config[TRIAL_MAP_NAME][
                    trial_class_name
                ]
        else:
            self.common_trial_keyword_arguments.update(self.project_kit_config[TRIAL_MAP_NAME])

        if READER_MAP_NAME in self.project_kit_config:
            self.common_trial_keyword_arguments["manual_reader_kwargs"] = self.project_kit_config[READER_MAP_NAME]

        self.common_trial_keyword_arguments["project_kit_config"] = self.project_kit_config

        self._dataset_reader()

        self.trial_id_to_keyword_arguments = {
            k: v
            for k, v in sorted(self.trial_id_to_keyword_arguments.items(), key=lambda kv: kv[0])
            if not self._to_skip_trial_id(k)
        }

    @classmethod
    @property
    def fields_to_exclude_from_single_schema(cls) -> set[str]:
        result = super().fields_to_exclude_from_single_schema
        result.update(("project_directory", "experiment_class_name"))
        return result

    @computed_field(return_type=type)  # type: ignore[misc]
    @cached_property
    def experiment_class(self) -> "ExperimentCLS":
        from bikipy.behaviour.mapping import experiment_name_to_class

        try:
            experiment = experiment_name_to_class[self.experiment_class_name]
        except KeyError:
            msg = (
                f"experiment_class in settings is set to an invalid value: {self.experiment_class_name}; "
                f"this value should not be changed after initialization of the project."
            )
            raise ValueError(msg)

        if self.trial_sequence_loops:
            experiment = experiment.trial_sequence_repetition(self.trial_sequence_loops)

        if (
            self.first_stage_is_habituation
            or "habituation" in self.project_kit_config
            and self.project_kit_config["habituation"]
        ):
            experiment = experiment.set_first_trial_to_habituation()
            self.trial_class_name_to_keyword_arguments[experiment.habituation_trial_class.__name__].update(
                self.project_kit_config["habituation"]
            )

        if self.custom_trial_sequence:
            experiment = experiment.set_custom_trial_sequence(self.custom_trial_sequence)

        return experiment

    def ingress_defined_fields(self) -> dict[str, set]:
        result = {
            "common_trial_keyword_arguments": set(self.common_trial_keyword_arguments),
            "trial_id_to_keyword_arguments": set(get_first_value_in_dict(self.trial_id_to_keyword_arguments)),
        }
        if self.experiment_class.has_stages:
            trial_class_name_to_keyword_arguments_fields = []

            for trial_class_keyword_arguments in self.trial_class_name_to_keyword_arguments.values():
                if trial_class_keyword_arguments:
                    trial_class_name_to_keyword_arguments_fields.extend(trial_class_keyword_arguments.keys())

            result["trial_class_name_to_keyword_arguments"] = set(trial_class_name_to_keyword_arguments_fields)

        return result

    # I/O ============================
    @computed_field  # type: ignore[misc]
    @cached_property
    def metadata_sheet_names(self) -> list[str]:
        return sheet_names_from_path(self.metadata_path)

    @computed_field  # type: ignore[misc]
    @cached_property
    def ranged_metadata(self) -> pd.DataFrame | None:
        if "ranged" in self.metadata_sheet_names:
            column_names = set(pd.read_excel(self.metadata_path, sheet_name="ranged").columns)
            if "Phase" in column_names:
                return pd.read_excel(self.metadata_path, sheet_name="ranged", index_col=[0, 1, 2])

    @computed_field  # type: ignore[misc]
    @cached_property
    def animal_metadata(self) -> pd.DataFrame | None:
        if "animal" in self.metadata_sheet_names:
            animal_df = pd.read_excel(
                self.metadata_path,
                sheet_name="animal",
                index_col=0,
            )
            animal_df.columns.names = ["Animal"]
            return animal_df

    @computed_field  # type: ignore[misc]
    @cached_property
    def metadata(self) -> pd.DataFrame:
        def join_trial_df_with_animal_metadata(df: pd.DataFrame) -> pd.DataFrame:
            if self.animal_metadata is not None:
                df = df.join(self.animal_metadata, how="inner")
            return df

        if "trial_id" in self.metadata_sheet_names:
            trial_id_df = pd.read_excel(
                self.metadata_path,
                sheet_name="trial_id",
                index_col=0,
            )
            trial_id_df.index.names = ["Trial"]

            trial_id_df = join_trial_df_with_animal_metadata(trial_id_df)

        elif "phase" in self.metadata_sheet_names:
            """
            The phase layout of trial ID is very similar to the original layout, with one minor difference:
            There are three index columns the 1st is the Phase column, the 2nd is the PhasePart column, and the
            3rd is the trial ID column. This column are merged into a single index for bikipy ingress
            """
            trial_id_df = pd.read_excel(self.metadata_path, sheet_name="phase", index_col=[0, 1, 2])
            trial_id_df.index = trial_id_df.index.map(lambda x: f"{x[0]}{x[1]}_{x[2]}")

            trial_id_df = join_trial_df_with_animal_metadata(trial_id_df)

        elif "animal_sequence" in self.metadata_sheet_names:
            """
            The Animal-Sequence layout derives trial ID from Animal and Sequence ID.
            """
            trial_id_df = pd.read_excel(
                self.metadata_path,
                sheet_name="animal_sequence",
                index_col=[0, 1],
            )
            trial_id_df.index.names = ["Animal", "Sequence"]
            trial_id_df = join_trial_df_with_animal_metadata(trial_id_df)

            trial_id_df["Animal"] = trial_id_df.index.get_level_values(level="Animal")

            trial_id_df.index = trial_id_df.index.map(lambda idx: f"{idx[0]}_{idx[1]}")
            trial_id_df.index.names = ["Trial"]

        elif "animal_day" in self.metadata_sheet_names:
            """
            The Animal-Sequence layout derives trial ID from Animal and Day.
            """
            trial_id_df = pd.read_excel(
                self.metadata_path,
                sheet_name="animal_day",
                index_col=[0, 1],
            )
            trial_id_df.index.names = ["Animal", "Day"]
            trial_id_df = join_trial_df_with_animal_metadata(trial_id_df)

            new_index = trial_id_df.index.map(lambda idx: f"{idx[0]}_{idx[1]}")

            trial_id_df.reset_index(inplace=True)
            trial_id_df.set_index(new_index, inplace=True)

            trial_id_df.index.names = ["Trial"]

        elif self.animal_metadata is not None:
            if "Phase" in self.animal_metadata.columns:
                """
                The phase layout of animal metadata consists of two column types:
                    - The "Phase" column, which defines the phase of the trial. This is the highest point in the
                    hierarchy
                    - "PhasePart-1", "PhasePart-2", ..., "PhasePart-N"; where N is the number of parts in each phase.
                    These column store the experiment number that animal belonged to in the respective part of the given
                    phase (stored in the previous type of column). See examples in phase...

                Additional information from a phase may be stored in the sheet with the same label as the phase.
                Phases are ASCII letters; the other parts of the index (part, and trial number) are integers.
                """

                # Detect all the Phase columns to iterate over them again when iterating over animals
                phase_to_df = {
                    phase: pd.read_excel(self.metadata_path, sheet_name=phase, index_col=[0, 1])
                    for phase in np.unique(self.animal_metadata["Phase"])
                    if phase in self.metadata_sheet_names
                }

                df_data = {}
                for _, row in self.animal_metadata.reset_index().iterrows():
                    phase = row.pop("Phase")
                    for column in self.animal_metadata.columns:
                        if "PhasePart" not in column:
                            continue

                        phase_part_trial_number = row.pop(column)
                        phase_part = int(column.split("-")[1])  # from "PhasePart-N"

                        try:
                            row = pd.concat([row, phase_to_df[phase].loc[(phase_part, phase_part_trial_number), :]])
                        except KeyError:
                            pass

                        df_data[f"{phase}{phase_part}_{phase_part_trial_number}"] = row

                trial_id_df = pd.DataFrame.from_dict(df_data, orient="index")

            else:
                de_indexed_animal_metadata = self.animal_metadata.reset_index()
                trial_id_df = pd.concat(
                    [de_indexed_animal_metadata for _ in range(self.experiment_class.trial_sequence_length)], axis=0
                )
                trial_id_df.sort_index(inplace=True)

                new_index = []
                for animal_id in self.animal_metadata.index.values:
                    for sequence_idx in range(self.experiment_class.trial_sequence_length):
                        new_index.append(f"{animal_id}_{sequence_idx}")

                trial_id_df.index = new_index

        else:
            msg = (
                f"Either trial_id or animal_sequence sheet must be defined in metadata "
                f"when using metadata plugins that are stageful. This allows BiKiPy ingress "
                f"to define a trial id for each trial object. Following plugins are set "
                f"to metadata:\n{self._plugins_metadata}"
            )
            raise ValueError(msg)

        assert "Animal" in trial_id_df, "Animal column must be present trial_id and animal metadata sheets"

        return trial_id_df

    @computed_field  # type: ignore[misc]
    @cached_property
    def metadata_perimeter_label_sheet(self) -> pd.DataFrame | None:
        """
        The sheet has Trial IDs as row indices; trial perimeter attributes
        as column indices; the value is the label of the perimeter belonging to
        the respective trial in the given column index
        :return:
        """
        if LABEL_TO_TRIAL_SHEET_NAME in self.metadata_sheet_names:
            return pd.read_excel(self.metadata_path, sheet_name=LABEL_TO_TRIAL_SHEET_NAME, index_col=0)

    def metadata_plugin_to_correct_sheet(self, plugin_model: "PluginType") -> pd.DataFrame:
        from bikipy.ingress.plugin.perimeter.single import PluginSinglePerimeter

        if plugin_model == PluginSinglePerimeter and self.metadata_perimeter_label_sheet is not None:
            return self.metadata_perimeter_label_sheet

        return self.metadata

    @computed_field  # type: ignore[misc]
    @property
    def settings_path(self) -> FilePath:
        return get_project_settings_path(self.project_directory)

    @computed_field  # type: ignore[misc]
    @property
    def metadata_path(self) -> FilePath:
        return infer_metadata_path(self.project_directory)

    @computed_field  # type: ignore[misc]
    @property
    def plugin_directory_path(self) -> DirectoryPath:
        return get_plugin_directory_path(self.project_directory)

    @computed_field  # type: ignore[misc]
    @cached_property
    def inspect_directory_path(self) -> DirectoryPath | None:
        if self.no_inspection:
            return

        result = get_inspect_directory_path(self.project_directory)
        if (
            not runtime_settings.ignore_pre_existing_inspection_directory
            and result.exists()
            and tuple(result.glob("**/*"))
        ):
            if not self.lazy_dev_mode:
                already_exists_prompt = input(
                    f"Inspection directory, {result}, already exists. "
                    f"Proceeding would result in deletion of directory tree. "
                    f"Would you like to proceed? y/N "
                )
                if already_exists_prompt.strip().lower() != "y":
                    import sys

                    print("Aborted by user, inspection directory already exists")
                    sys.exit(0)

            shutil.rmtree(result)

        result.mkdir(exist_ok=True)
        return result

    @computed_field  # type: ignore[misc]
    @property
    def result_directory_path(self) -> DirectoryPath:
        return result_directory_path(self.project_directory)

    @computed_field  # type: ignore[misc]
    @cached_property
    def cache_directory_path(self) -> DirectoryPath:
        os.makedirs(result := self.project_directory / "bikipy_ingress_cache", exist_ok=True)
        return result

    # Plugin methods ============================== Read more about plugins in respective __init__.py file

    @computed_field  # type: ignore[misc]
    @cached_property
    def _global_plugins(self) -> list:
        from bikipy.ingress.plugin.map import ingress_key_to_model

        return [
            ingress_key_to_model[ingress_key]
            for ingress_key, strategy in self.plugin_definitions
            if PluginScope.GLOBAL in strategy
        ]

    @computed_field(return_type=list)  # type: ignore[misc]
    @cached_property
    def _plugins_metadata(self) -> list:
        from bikipy.ingress.plugin.map import ingress_key_to_model

        return [
            ingress_key_to_model[ingress_key]
            for ingress_key, strategy in self.plugin_definitions
            if PluginScope.METADATA in strategy
        ]

    @computed_field(return_type=list)  # type: ignore[misc]
    @cached_property
    def _trialwise_plugins(self) -> list:
        from bikipy.ingress.plugin.map import ingress_key_to_model

        return [
            ingress_key_to_model[ingress_key]
            for ingress_key, strategy in self.plugin_definitions
            if PluginScope.TRIALWISE in strategy
        ]

    # Backend methods =================================

    def trial_id_exists(self, trial_id: Label) -> bool:
        if trial_id in self.metadata.index:
            return True
        if self.metadata_trial_ids_are_higher_level and isinstance(trial_id, str):
            for designator_id in self.metadata.index:
                if f"{designator_id}_" in trial_id:
                    self.trial_id_to_designator_id[trial_id] = designator_id
                    return True

        return False

    @computed_field(repr=False)  # type: ignore[misc]
    @cached_property
    def experiment(self) -> BaseExperiment:
        situational_kwargs = {}

        if self.experiment_class.has_stages:
            assert self.trial_id_to_trial_class_name, "Trial ID to trial class map must be defined"

            situational_kwargs["trial_id_to_trial_class_name"] = self.trial_id_to_trial_class_name
            situational_kwargs["trial_class_name_to_keyword_arguments"] = self.trial_class_name_to_keyword_arguments

        return self.experiment_class(
            inspection_fig_output_path=self.inspect_directory_path if self.create_inspection_plots else False,
            trial_init_error_out_dir=self.project_directory,
            common_trial_keyword_arguments=self.common_trial_keyword_arguments,
            trial_id_to_keyword_arguments=self.trial_id_to_keyword_arguments,
            project_kit_config=self.project_kit_config,
            **self.project_kit_config["experiment"],
            **situational_kwargs,
        )

    # Motion <-> Feature fitting ===================================
    @computed_field  # type: ignore[misc]
    @cached_property
    def animal_metadata_fit_to_combined_feature_motion_df(self) -> pd.DataFrame:
        if self.animal_metadata.columns.nlevels >= self.experiment.combined_feature_motion_df.columns.nlevels:
            return self.animal_metadata
        return copycat_assumes_levels_of_icon(
            self.animal_metadata, self.experiment.combined_feature_motion_df, "Global"
        )

    def metadata_fit_to_combined_feature_motion_df(self) -> pd.DataFrame:
        if self.metadata.columns.nlevels >= self.experiment.combined_feature_motion_df.columns.nlevels:
            return self.metadata
        return copycat_assumes_levels_of_icon(self.metadata, self.experiment.combined_feature_motion_df, "Global")

    # Client-side methods ===============================
    @computed_field(repr=False)  # type: ignore[misc]
    @cached_property
    def trial_label_to_df(self) -> dict[Label, pd.DataFrame]:
        if self.metadata_trial_ids_are_higher_level:
            return self.experiment.trial_label_to_df

        result = {}
        for trial_label, analysis_df in self.experiment.trial_label_to_df.items():
            metadata = (
                self.metadata
                if self.metadata.columns.nlevels >= analysis_df.columns.nlevels
                else copycat_assumes_levels_of_icon(self.metadata, analysis_df, "")
            )
            analysis_df = (
                analysis_df
                if analysis_df.columns.nlevels >= self.metadata.columns.nlevels
                else copycat_assumes_levels_of_icon(analysis_df, self.metadata, "")
            )
            result[trial_label] = analysis_df.join(metadata, how="inner")

        return result

    @computed_field(repr=False)  # type: ignore[misc]
    @cached_property
    def animal_analysis_df(self) -> pd.DataFrame:
        df = pd.concat(
            (self.animal_metadata_fit_to_combined_feature_motion_df, self.experiment.combined_feature_motion_df),
            axis=1,
        )
        df.columns.names = (
            ["Stage", "Feature", "Location/Category"]
            if self.experiment_class.has_stages
            else ["Feature", "Location/Category"]
        )
        df.index.names = ["Animal"]

        return df

    def save_analysis_data(self):
        if self.profile_runtime:
            try:
                with Profile() as pr:
                    self.experiment.analyze_trials()
            finally:
                stats = pstats.Stats(pr)
                stats.sort_stats(pstats.SortKey.TIME)
                stats.dump_stats(self.inspect_directory_path / "performance_analysis.prof")

        parquet_dir = (
            self.result_directory_path / "parquet" if len(self.trial_label_to_df) == 1 else self.result_directory_path
        )
        parquet_dir.mkdir(exist_ok=True)

        # We define to_excel and to_parquet in their own loops for atomicity with respect to formats
        # as parquet crashes sometimes.
        with pd.ExcelWriter(self.result_directory_path / f"{self.experiment_class_name}.xlsx") as writer:
            for trial_label, df in self.trial_label_to_df.items():
                df.to_excel(writer, sheet_name=trial_label)

        for trial_label, df in self.trial_label_to_df.items():
            df.to_parquet(parquet_dir / f"{trial_label}.parquet", **TO_PARQUET_KWARGS)

    def create_analysis_videos(self, trial_ids: Iterable[Label], **trial_video_kwargs) -> None:
        for trial_id in trial_ids:
            self.experiment.get_trial_object(trial_id).generate_inspection_video(**trial_video_kwargs)

    def sample_one_trial_from_each_trial_class(self, **trial_video_kwargs) -> None:
        for trial_objects in self.experiment.trial_class_to_trial_objects.values():
            trial_objects[0].generate_inspection_video(**trial_video_kwargs)

    def purge_cached_reads(self, override_pattern: Optional[str] = None, auto: bool = False) -> None:
        pattern = override_pattern or AUGMENTED_COORDINATE_CACHED_FILE_LABEL
        to_delete = [
            f
            for f in chain(
                self.dataset_directory.glob(f"**/**/*{pattern}*"),
                self.dataset_directory.glob(f"**/*{pattern}*"),
                self.dataset_directory.glob(f"*{pattern}*"),
            )
        ]
        if not to_delete:
            print(f"No files found with pattern {pattern}")
            return

        readable_to_delete = "\n".join((str(f.relative_to(self.dataset_directory)) for f in to_delete))
        if (
            auto
            or input(
                f"Pattern: {pattern}\n"
                f"{readable_to_delete}\n===================\nPURGING CACHED DATA\n===================\n"
                f"Will be deleted, are you sure? y/N "
            ).lower()
            == "y"
        ):
            for f in to_delete:
                print(f"Deleting: {f}")
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass

    # Plugin methods ============================== Read more about plugins in respective __init__.py file

    def get_meter_per_pixel(self, trial_id: Optional[Label] = None) -> NpNDArrayFp64:
        from bikipy.ingress.plugin.meters_per_pixel import (
            PluginMeterPerPixel,
            detect_meters_per_pixel_in_perimeter_directory,
        )

        if PluginScope.OTHER in self.plugin_definitions.meters_per_pixel:
            if self.plugin_definitions.radial:
                # Radial defines the meters per pixel on the respective PerimeterSet
                from bikipy.ingress.plugin.perimeter.radial_maze import PluginRadial

                return (
                    self.common_trial_keyword_arguments[PluginRadial.default_trial_argument_key]
                    if PluginScope.GLOBAL in self.plugin_definitions.radial
                    else self.trial_id_to_keyword_arguments[trial_id][PluginRadial.default_trial_argument_key]
                ).meters_per_pixel

        if PluginScope.TRIALWISE in self.plugin_definitions.meters_per_pixel:
            try:
                return self.trial_id_to_keyword_arguments[trial_id][PluginMeterPerPixel.default_trial_argument_key]
            except KeyError:
                pass

        if PluginScope.METADATA in self.plugin_definitions.meters_per_pixel:
            try:
                file_label = self.metadata.loc[trial_id, PluginMeterPerPixel.human_readable_index]
                return detect_meters_per_pixel_in_perimeter_directory(self.plugin_directory_path)[file_label]
            except KeyError:
                pass

        if PluginScope.GLOBAL in self.plugin_definitions.meters_per_pixel:
            return self.common_trial_keyword_arguments[PluginMeterPerPixel.default_trial_argument_key]

        raise AttributeError(f"Could not find meters_per_pixel for trial {trial_id}")

    def get_plugin_config(self, plugin_model: "BasePlugin") -> dict[str, Any]:
        return self.project_kit_config["plugin"].get(plugin_model.__name__, {})

    # Private methods ===============================

    def _define_plugin(self, plugin_model: "PluginType", plugin_scope: PluginScope, **field_kwargs) -> "BasePlugin":
        additional_field_args = {}
        if plugin_model.__name__ in self.project_kit_config:
            additional_field_args.update(self.project_kit_config[plugin_model.__name__])

        return plugin_model(plugin_scope=plugin_scope, ingress=self, **additional_field_args, **field_kwargs)

    def _trial_class_from_stage_index(self, stage_index: Label) -> "TrialCLS":
        return self.experiment_class.stage_index_to_trial_class[stage_index]

    def _trialwise_plugins_for_trial_id(
        self, trial_id: Label, trial_directory: DirectoryPath, trial_id_plugin_glob_format_string: str
    ) -> dict:
        result = {}
        for plugin_model in self._trialwise_plugins:
            glob_str = trial_id_plugin_glob_format_string.format(
                trial_id=trial_id, plugin_code_key=plugin_model.code_key
            )
            plugin_data_files = tuple(trial_directory.glob(glob_str))
            if not plugin_data_files:
                continue

            if len(plugin_data_files) > 1:
                msg = f"Plugin {plugin_model.human_readable_index}: Only one file per trial"
                raise ValueError(msg)

            result[plugin_model.default_trial_argument_key] = self._define_plugin(
                plugin_model,
                PluginScope.TRIALWISE,
                data_path=plugin_data_files[0],
                **self.get_plugin_config(plugin_model),
            ).trialwise_and_metadata(trial_id)

        return result

    @validate_call
    def _coordinate_files_in_directory(self, directory_path: DirectoryPath) -> list[FilePath]:
        available_indices = {
            int(file.stem.split(self.coordinate_file_index_delimiter)[0])
            for file in directory_path.iterdir()
            if file.is_file()
            and ANALYSIS_CACHE_STEM_ID not in file.stem
            and AUGMENTED_COORDINATE_CACHED_FILE_LABEL not in file.stem
            and BIKIPY_ANALYSIS_VIDEO_PREFIX not in file.stem
            and "~lock" not in file.stem
        }

        result = []
        for index in available_indices:
            index_files = [
                f for f in directory_path.glob(f"{index}{self.coordinate_file_index_delimiter}*") if f.is_file()
            ]
            if any((current_file := file).stem.endswith("timestamped") for file in index_files):
                result.append(current_file)
                continue

            if any(
                (current_file := file).suffix == ".parquet" and AUGMENTED_COORDINATE_CACHED_FILE_LABEL not in file.stem
                for file in index_files
            ):
                result.append(current_file)
                continue

            if any((current_file := file).suffix == ".h5" for file in index_files):
                result.append(current_file)
                continue

            if any((current_file := file).suffix == ".csv" for file in index_files):
                result.append(current_file)
                continue

            raise ValueError(f"Could not find a supported coordinate file for index {index} in {directory_path}")

        return result

    @validate_call
    def _to_skip_trial_id(self, trial_id: Label) -> bool:
        return (
            self.trial_ids_to_analyse
            and trial_id not in self.trial_ids_to_analyse
            or not self.trial_id_exists(trial_id)
        )

    @staticmethod
    def _get_id_from_path_stem(path: Path) -> Label:
        stem = path.stem
        if "-" in stem:
            stem = path.stem.split("-")[0]
        return int(stem) if stem.isdigit() else stem


IngressWorkflow = TypeVar("IngressWorkflow", bound=BaseIngressWorkflow)
