import os
import pstats
from abc import ABC, abstractmethod
from cProfile import Profile
from functools import cached_property
from itertools import chain
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Optional, TypeVar

import numpy as np
import pandas as pd
from inflection import underscore
from schemantic.model.project import SchemanticMixin
from pydantic import DirectoryPath, FilePath, PositiveInt, validate_arguments, Field
from pydantic_numpy.dtype import NDArrayFp64

from bikipy import set_bikipy_settings_from_dict
from bikipy.core.base_class import BikipyModel
from bikipy.core.typing import Label
from bikipy.ingress.plugin_scope import PluginScope

from bikipy.ingress.utils.io import (
    get_inspect_directory_path,
    get_plugin_directory_path,
    get_project_settings_path,
    infer_metadata_path,
    result_directory_path,
)
from bikipy.perimeter.base import Perimeter
from bikipy.reader.base import BaseReader
from bikipy.utils.collection_utils import (
    copycat_assumes_levels_of_icon,
    get_first_value_in_dict,
)
from bikipy.utils.constants import TO_PARQUET_KWARGS
from bikipy.utils.misc import defaultdict_dict_factory, sheet_names_from_path

if TYPE_CHECKING:
    from bikipy.behaviour.core import Experiment, ExperimentCLS, TrialCLS
    from bikipy.ingress.plugin.base import Plugin

logger = getLogger(__name__)


class BaseIngressWorkflow(BikipyModel, SchemanticMixin, ABC):
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

    project_directory: DirectoryPath = ...
    dataset_directory: DirectoryPath = ...
    experiment_class_name: str = ...

    first_stage_is_habituation: bool = False
    metadata_trial_ids_are_higher_level: bool = False
    trial_sequence_loops: int = 1

    definition_meters_per_pixel: frozenset[PluginScope]
    definition_single_perimeter: Optional[frozenset[PluginScope]]
    definition_enclosure: Optional[frozenset[PluginScope]]
    definition_radial: Optional[frozenset[PluginScope]]
    definition_change_reference: Optional[frozenset[PluginScope]]
    definition_frame: Optional[frozenset[PluginScope]]
    definition_video: Optional[frozenset[PluginScope]]
    definition_timestamp: Optional[frozenset[PluginScope]]
    definition_center: Optional[frozenset[PluginScope]]

    ingress_defined_perimeters: dict[str, Perimeter] = {}

    deprecated_project_settings_file_name: bool = False

    _experiment_data_defined: bool = False
    _trial_id_to_trial_class_name: dict[Label, str] = {}
    _common_trial_keyword_arguments: dict[str, Any] = {}
    _trial_id_to_keyword_arguments: dict[Label, dict[str, Any]] = Field(default_factory=defaultdict_dict_factory)
    _trial_class_name_to_keyword_arguments: dict[str, dict] = {}

    _trial_id_to_designator_id: dict[str, str] = {}
    _designator_id_to_kwargs: dict[str, dict] = Field(default_factory=defaultdict_dict_factory)

    profile_runtime: ClassVar[bool] = True
    ingress_method: ClassVar[str]

    _project_kit_jit = True

    class Config:
        keep_untouched = (cached_property,)

    @abstractmethod
    def _dataset_reader(self) -> None:
        """
        This component reads the contents of the dataset folder. It is an abstractmethod because projects often have
        different layouts. Refer to submodules in the same directory as this module
        to explore the different implementations
        """
        ...

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(("project_directory", "experiment_class_name"))
        return upstream

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
        if self.first_stage_is_habituation:
            experiment = experiment.set_first_trial_to_habituation()

        return experiment

    @property
    def experiment_class_kwargs(self) -> dict[str, dict]:
        result = {
            "common_trial_keyword_arguments": self.common_trial_keyword_arguments,
            "trial_id_to_keyword_arguments": self.trial_id_to_keyword_arguments,
        }
        if self.experiment_class.has_stages:
            result["trial_id_to_trial_class_name"] = self.trial_id_to_trial_class_name
            result["trial_class_name_to_keyword_arguments"] = self.trial_class_name_to_keyword_arguments

        return result

    @property
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

    @property
    def trial_id_to_trial_class_name(self):
        if not self._experiment_data_defined:
            self._define_experiment_data()
        assert self._trial_id_to_trial_class_name
        return self._trial_id_to_trial_class_name

    @property
    def common_trial_keyword_arguments(self):
        if not self._experiment_data_defined:
            self._define_experiment_data()
        return self._common_trial_keyword_arguments

    @property
    def trial_id_to_keyword_arguments(self):
        self._define_experiment_data_if_not_defined()
        return dict(self._trial_id_to_keyword_arguments)

    @property
    def trial_class_name_to_keyword_arguments(self):
        self._define_experiment_data_if_not_defined()
        return self._trial_class_name_to_keyword_arguments

    @property
    def metadata_index_to_trial_id(self):
        if not self._experiment_data_defined:
            self._define_experiment_data()
        return self._metadata_index_to_trial_id

    # I/O ============================

    @cached_property
    def _metadata_sheet_names(self) -> list[str]:
        return sheet_names_from_path(self.metadata_path)

    @cached_property
    def ranged_metadata(self) -> pd.DataFrame | None:
        if "ranged" in self._metadata_sheet_names:
            column_names = set(pd.read_excel(self.metadata_path, sheet_name="ranged").columns)
            if "Phase" in column_names:
                return pd.read_excel(self.metadata_path, sheet_name="ranged", index_col=[0, 1, 2])

    @cached_property
    def animal_metadata(self) -> pd.DataFrame | None:
        if "animal" in self._metadata_sheet_names:
            animal_df = pd.read_excel(
                self.metadata_path,
                sheet_name="animal",
                index_col=0,
            )
            animal_df.columns.names = ["Animal"]
            return animal_df

    @cached_property
    def metadata(self) -> pd.DataFrame:
        def join_trial_df_with_animal_metadata(df: pd.DataFrame) -> pd.DataFrame:
            if self.animal_metadata is not None:
                df = df.join(self.animal_metadata, how="inner")
            return df

        if "trial_id" in self._metadata_sheet_names:
            trial_id_df = pd.read_excel(
                self.metadata_path,
                sheet_name="trial_id",
                index_col=0,
            )
            trial_id_df.index.names = ["Trial"]

            trial_id_df = join_trial_df_with_animal_metadata(trial_id_df)

        elif "phase" in self._metadata_sheet_names:
            """
            The phase layout of trial ID is very similar to the original layout, with one minor difference:
            There are three index columns the 1st is the Phase column, the 2nd is the PhasePart column, and the
            3rd is the trial ID column. This column are merged into a single index for bikipy ingress
            """
            trial_id_df = pd.read_excel(self.metadata_path, sheet_name="phase", index_col=[0, 1, 2])
            trial_id_df.index = trial_id_df.index.map(lambda x: f"{x[0]}{x[1]}_{x[2]}")

            trial_id_df = join_trial_df_with_animal_metadata(trial_id_df)

        elif "animal_sequence" in self._metadata_sheet_names:
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

        elif "animal_day" in self._metadata_sheet_names:
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
                    if phase in self._metadata_sheet_names
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

    @property
    def settings_path(self) -> FilePath:
        return get_project_settings_path(self.project_directory, self.deprecated_project_settings_file_name)

    @property
    def metadata_path(self) -> FilePath:
        return infer_metadata_path(self.project_directory)

    @property
    def plugin_directory_path(self) -> DirectoryPath:
        return get_plugin_directory_path(self.project_directory)

    @property
    def inspect_directory_path(self) -> DirectoryPath:
        return get_inspect_directory_path(self.project_directory)

    @property
    def result_directory_path(self) -> DirectoryPath:
        return result_directory_path(self.project_directory)

    # Plugin methods ============================== Read more about plugins in respective __init__.py file

    @cached_property
    def _plugin_definitions(self) -> dict[str, frozenset[PluginScope]]:
        result = {
            "perimeter": self.definition_perimeter,
            "enclosure": self.definition_enclosure,
            "radial": self.definition_radial,
            "change_reference": self.definition_change_reference,
            "frame": self.definition_frame,
            "video": self.definition_video,
            "timestamp": self.definition_timestamp,
            "center": self.definition_center,
        }
        for k, v in result.items():
            if not v:
                del result[k]
        return result

    @cached_property
    def _global_plugins(self) -> list["Plugin", ...]:
        from bikipy.ingress.plugin.map import ingress_key_to_model

        return [
            ingress_key_to_model[ingress_key]
            for ingress_key, strategy in self._plugin_definitions.items()
            if PluginScope.GLOBAL in strategy
        ]

    @cached_property
    def _plugins_metadata(self) -> list["Plugin", ...]:
        from bikipy.ingress.plugin.map import ingress_key_to_model

        return [
            ingress_key_to_model[ingress_key]
            for ingress_key, strategy in self._plugin_definitions.items()
            if PluginScope.METADATA in strategy
        ]

    @cached_property
    def _trialwise_plugins(self) -> list["Plugin", ...]:
        from bikipy.ingress.plugin.map import ingress_key_to_model

        return [
            ingress_key_to_model[ingress_key]
            for ingress_key, strategy in self._plugin_definitions.items()
            if PluginScope.TRIALWISE in strategy
        ]

    # Backend functions =================================

    def _define_experiment_data_if_not_defined(self):
        if not self._experiment_data_defined:
            self._define_experiment_data()

    def _define_experiment_data(self) -> None:
        for plugin_model in self._global_plugins:
            first_file = next(self.plugin_directory_path.glob(f"{plugin_model.code_key}*"))
            self._common_trial_keyword_arguments[plugin_model.default_trial_argument_key] = plugin_model(
                data_path=first_file, ingress=self
            ).globally_defined

        metadata_trial_target_dict = (
            self._designator_id_to_kwargs
            if self.metadata_trial_ids_are_higher_level
            else self._trial_id_to_keyword_arguments
        )

        for plugin_model in self._plugins_metadata:
            label_to_file_path = {
                file_path.stem.split("-")[-1]: file_path
                for file_path in self.plugin_directory_path.glob(f"{plugin_model.code_key}*")
            }
            for trial_id, row in self.metadata.iterrows():
                if plugin_model.human_readable_index in row:
                    trial_id_plugin_label = row[plugin_model.human_readable_index]
                    if isinstance(trial_id_plugin_label, float) and np.isnan(trial_id_plugin_label):
                        continue

                    metadata_trial_target_dict[trial_id][plugin_model.default_trial_argument_key] = plugin_model(
                        data_path=label_to_file_path[str(trial_id_plugin_label)], ingress=self
                    ).trialwise_and_metadata(trial_id)

                elif plugin_model.plural_entries:
                    for key, trial_id_plugin_label in row.items():
                        if isinstance(trial_id_plugin_label, float) and np.isnan(trial_id_plugin_label):
                            continue

                        if plugin_model.human_readable_index in key:
                            metadata_trial_target_dict[trial_id][underscore(key)] = plugin_model(
                                data_path=label_to_file_path[str(trial_id_plugin_label)],
                                manual_trial_argument_key=underscore(key),
                                ingress=self,
                            ).trialwise_and_metadata(trial_id, naive=True)

                else:
                    msg = (
                        f"{plugin_model.human_readable_index} is active in the settings, but is missing in the metadata"
                    )
                    raise ValueError(msg)

        for trial_class_name, kwargs in self._project_kit_config["trial"].items():
            assert trial_class_name in self.experiment_class.trial_class_names
            self._trial_class_name_to_keyword_arguments[trial_class_name] = kwargs

        self._dataset_reader()

        self._experiment_data_defined = True

    def trial_id_exists(self, trial_id: Label) -> bool:
        if trial_id in self.metadata.index:
            return True
        if self.metadata_trial_ids_are_higher_level and isinstance(trial_id, str):
            for designator_id in self.metadata.index:
                if f"{designator_id}_" in trial_id:
                    self._trial_id_to_designator_id[trial_id] = designator_id
                    return True

        return False

    @cached_property
    def experiment(self) -> "Experiment":
        set_bikipy_settings_from_dict(self.runtime_settings)

        return self.experiment_class(
            **self._project_kit_config["experiment"],
            **self.experiment_class_kwargs,
            inspect_arg=self.inspect_directory_path,
            trial_init_error_out_dir=self.result_directory_path,
        )

    # Motion <-> Feature fitting ===================================

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

    # Client-side functions ===============================

    @cached_property
    def trial_label_to_df(self) -> dict[str | PositiveInt, pd.DataFrame]:
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
            with Profile() as pr:
                assert self.experiment.trial_label_to_df

            stats = pstats.Stats(pr)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.dump_stats(self.inspect_directory_path / "performance_analysis.prof")

        with pd.ExcelWriter(self.result_directory_path / f"{self.experiment_class_name}.xlsx") as writer:
            for trial_label, df in self.trial_label_to_df.items():
                df.to_excel(writer, sheet_name=str(trial_label))

        parquet_dir = (
            self.result_directory_path / "parquet" if len(self.trial_label_to_df) == 1 else self.result_directory_path
        )
        parquet_dir.mkdir(exist_ok=True)

        for trial_label, df in self.trial_label_to_df.items():
            df.to_parquet(parquet_dir / f"{trial_label}-{self.experiment_class_name}.parquet", **TO_PARQUET_KWARGS)

    def purge_cached_reads(self, override_pattern: Optional[str] = None) -> None:
        pattern = override_pattern or BaseReader.augmented_coordinate_cached_file_label
        to_delete = [
            f
            for f in chain(
                self.dataset_directory.glob(f"**/**/*{pattern}*"),
                self.dataset_directory.glob(f"**/*{pattern}*"),
                self.dataset_directory.glob(f"*{pattern}*"),
            )
        ]
        if not to_delete:
            logger.info(f"No files found with pattern {pattern}")
            return

        readable_to_delete = "\n".join((f.name for f in to_delete))
        if (
            input(
                f"Pattern: {pattern}\n"
                f"{readable_to_delete}\n===================\nPURGING CACHED DATA\n===================\n"
                f"Will be deleted, are you sure? y/N "
            ).lower()
            == "y"
        ):
            for f in to_delete:
                logger.debug(f"Deleting: {f}")
                os.remove(f)

    # Plugin methods ============================== Read more about plugins in respective __init__.py file

    def get_meter_per_pixel(self, trial_id: Optional[str | PositiveInt] = None) -> NDArrayFp64:
        from bikipy.ingress.plugin.map import PluginMeterPerPixel
        from bikipy.ingress.plugin.meters_per_pixel import detect_meters_per_pixel_in_perimeter_directory

        if PluginScope.TRIALWISE in self.definition_meters_per_pixel:
            try:
                return self.trial_id_to_keyword_arguments[trial_id]["meters_per_pixel"]
            except KeyError:
                pass
        if PluginScope.METADATA in self.definition_meters_per_pixel:
            try:
                file_label = self.metadata["MetersPerPixel"][trial_id]
                return detect_meters_per_pixel_in_perimeter_directory(self.plugin_directory_path)[file_label]
            except KeyError:
                pass
        if PluginScope.GLOBAL in self.definition_meters_per_pixel:
            return self._common_trial_keyword_arguments[PluginMeterPerPixel.default_trial_argument_key]

        raise AttributeError(f"Could not find meters_per_pixel for trial {trial_id}")

    # Private methods ===============================

    def _trial_class_from_stage_index(self, stage_index: str | PositiveInt) -> "TrialCLS":
        return self.experiment_class.stage_index_to_trial_class_name[stage_index]

    def _trialwise_plugins_for_trial_id(
        self, trial_id: Label, trial_directory: DirectoryPath, trial_id_plugin_glob_format_string: str
    ) -> dict:
        result = {}
        for plugin_model in self.plugins_trialwise:
            glob_str = trial_id_plugin_glob_format_string.format(
                trial_id=trial_id, plugin_code_key=plugin_model.code_key
            )
            plugin_data_files = tuple(trial_directory.glob(glob_str))
            if len(plugin_data_files) > 1:
                msg = f"Plugin {plugin_model.human_readable_index}: Only one file per trial"
                raise ValueError(msg)

            try:
                result[plugin_model.default_trial_argument_key] = plugin_model(
                    data_path=plugin_data_files[0], ingress=self
                ).trialwise_and_metadata(trial_id)
            except IndexError:
                pass

        return result

    @validate_arguments
    def coordinate_files_in_directory(self, directory_path: DirectoryPath) -> list[FilePath, ...]:
        available_indices = {int(file.stem.split(".")[0]) for file in directory_path.iterdir() if file.is_file()}

        result = []
        for index in available_indices:
            index_files = [f for f in directory_path.glob(f"{index}.*") if f.is_file()]
            if any((current_file := file).stem.endswith("timestamped") for file in index_files):
                result.append(current_file)
                continue

            if any(
                (current_file := file).suffix == ".parquet"
                and BaseReader.augmented_coordinate_cached_file_label not in file.stem
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

    @validate_arguments
    def _gather_coordinates_and_potential_timestamp_data(self, coordinate_path: FilePath) -> dict[str, FilePath]:
        timestamp_stem = coordinate_path.stem.replace("coordinates", "timestamps").split("-")[0]
        potential_timestamp_set_path_finder = tuple(
            coordinate_path.parent.glob(f"{timestamp_stem}*.{self.timestamp_file_suffix}")
        )

        result = {"framewise_coordinates_path": coordinate_path}
        if potential_timestamp_set_path_finder:
            assert len(potential_timestamp_set_path_finder) == 1
            result["coordinate_timestamp_set_path"] = potential_timestamp_set_path_finder[0]

        return result

    @staticmethod
    def _get_id_from_path_stem(path: Path) -> str | PositiveInt:
        stem = path.stem
        if "-" in stem:
            stem = path.stem.split("-")[0]
        return int(stem) if stem.isdigit() else stem

    @staticmethod
    def _raise_unsupported_plugin_method(plugin_name: str, supported_methods: Iterable[str]) -> None:
        msg = f"{plugin_name} only supports: {', '.join(supported_methods)}"
        raise ValueError(msg)


IngressWorkflow = TypeVar("IngressWorkflow", bound=BaseIngressWorkflow)
