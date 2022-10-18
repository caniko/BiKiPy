import json
import pstats
from abc import ABC, abstractmethod
from collections import defaultdict
from cProfile import Profile
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Hashable, Iterable, Optional, TypeVar

import numpy as np
import pandas as pd
import yaml
from pydantic import DirectoryPath, FilePath, PositiveInt, validate_arguments
from pydantic_numpy.dtype import NDArrayFp64

from bikipy.core.base_class import BaseBikipy
from bikipy.core.typing import TrialId
from bikipy.ingress.plugin import ALL_PLUGINS, PluginMeterPerPixel, ingress_key_to_model
from bikipy.ingress.plugin.base import Plugin
from bikipy.ingress.plugin.meters_per_pixel import (
    detect_meters_per_pixel_in_perimeter_directory,
)
from bikipy.ingress.utils import settings
from bikipy.ingress.utils.io import (
    get_dataset_directory_path,
    get_inspect_directory_path,
    get_plugin_directory_path,
    get_project_settings_path,
    infer_metadata_path,
    load_settings,
)
from bikipy.ingress.utils.model_schema import extended_group_schema, extended_schema
from bikipy.ingress.utils.settings import get_definable_settings
from bikipy.perimeter.base import BaseSinglePerimeter, Perimeter
from bikipy.reader import DeepLabCutReader
from bikipy.utils.collection_utils import (
    copycat_assumes_levels_of_icon,
    get_first_value_in_dict,
)
from bikipy.utils.misc import sheet_names_from_path

if TYPE_CHECKING:
    from bikipy.behaviour.core.base import Experiment, Trial


FIRST_TRIAL_IS_HABITUATION_INGRESS_FIELD = "first_stage_is_habituation"


logger = getLogger(__name__)


class BaseIngress(BaseBikipy, ABC):
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

    project_root_directory: DirectoryPath

    ingress_defined_perimeters: dict[str, Perimeter] = {}

    _experiment_data_defined: bool = False
    _trial_id_to_trial_class_name: dict[Hashable, str] = {}
    _common_trial_keyword_arguments: dict[str, Any] = {}
    _trial_id_to_keyword_arguments: dict[Hashable, dict[str, Any]] = defaultdict(dict)
    _trial_class_name_to_keyword_arguments: dict[str, Any] = {}

    ingress_method: ClassVar[str]

    @abstractmethod
    def _dataset_reader(self) -> None:
        """
        This component reads the contents of the dataset folder. It is an abstractmethod because projects often have
        different layouts. Refer to submodules in the same directory as this module
        to explore the different implementations
        """
        ...

    @classmethod
    def from_project_root_directory(cls, project_root_directory: DirectoryPath):
        kwargs = {"project_root_directory": project_root_directory}
        match auto_define_ingress_object(project_root_directory).ingress_method:
            case "animal":
                from bikipy.ingress import AnimalIngress

                return AnimalIngress(**kwargs)
            case "phase":
                from bikipy.ingress import PhaseIngress

                return PhaseIngress(**kwargs)
            case _:
                msg = (
                    f"Defined ingress method, {auto_define_ingress_object(project_root_directory).ingress_method}, "
                    f"is not supported"
                )
                raise AttributeError(msg)

    @property
    def experiment_name(self) -> str:
        return self.settings["immutable"]["experiment_class"]

    @cached_property
    def experiment_class(self) -> "Experiment":
        from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS

        try:
            experiment = EXPERIMENT_NAME_TO_CLASS[self.experiment_name]
        except KeyError:
            msg = (
                f"experiment_class in settings is set to an invalid value: "
                f"{self.settings['immutable']['experiment_class']}; "
                f"this value should not be changed after initialization of the project."
            )
            raise ValueError(msg)

        if self.settings["ingress"][FIRST_TRIAL_IS_HABITUATION_INGRESS_FIELD]:
            experiment = experiment.set_first_trial_to_habituation()

        if self.settings["ingress"]["manual_trial_sequence_repetitions"]:
            experiment.trial_sequence_repetition = self.settings["ingress"]["manual_trial_sequence_repetitions"]

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

    @property
    def settings(self) -> dict:
        return load_settings(self.project_root_directory)

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
                f"to metadata:\n{self._metadata_plugins}"
            )
            raise ValueError(msg)

        assert "Animal" in trial_id_df, "Animal column must be present trial_id and animal metadata sheets"

        return trial_id_df

    @property
    def settings_path(self) -> FilePath:
        return get_project_settings_path(self.project_root_directory)

    @property
    def metadata_path(self) -> FilePath:
        return infer_metadata_path(self.project_root_directory)

    @property
    def dataset_directory_path(self) -> DirectoryPath:
        if self.settings["ingress"]["dataset_directory"]:
            if not (path := Path(self.settings["ingress"]["dataset_directory"])).exists():
                msg = f"dataset_directory must exist: {path}"
                raise AttributeError(msg)
            return path
        return get_dataset_directory_path(self.project_root_directory)

    @property
    def plugin_directory_path(self) -> DirectoryPath:
        return get_plugin_directory_path(self.project_root_directory)

    @property
    def inspect_directory_path(self) -> DirectoryPath:
        return get_inspect_directory_path(self.project_root_directory)

    @property
    def result_directory_path(self) -> DirectoryPath:
        result_directory = self.project_root_directory / "result"
        result_directory.mkdir(exist_ok=True)
        return result_directory

    # Constants =============================

    @cached_property
    def framewise_coordinates_file_suffix(self):
        return self.settings["ingress"]["framewise_coordinates_file_suffix"]

    # Backend functions =================================

    def _define_experiment_data_if_not_defined(self):
        if not self._experiment_data_defined:
            self._define_experiment_data()

    def _define_experiment_data(self) -> None:
        for plugin_model in self._global_plugins:
            first_file = next(self.plugin_directory_path.glob(f"{plugin_model.code_key}*"))
            self._common_trial_keyword_arguments[plugin_model.bikipy_trial_key] = plugin_model(
                data_path=first_file, ingress=self
            ).globally_defined

        for plugin_model in self._metadata_plugins:
            label_to_file_path = {
                file_path.stem.split("-")[-1]: file_path
                for file_path in self.plugin_directory_path.glob(f"{plugin_model.code_key}*")
            }
            for trial_id, row in self.metadata.iterrows():
                trial_id_plugin_label = row[plugin_model.human_readable_index]

                if isinstance(trial_id_plugin_label, float) and np.isnan(trial_id_plugin_label):
                    continue

                self._trial_id_to_keyword_arguments[trial_id][plugin_model.bikipy_trial_key] = plugin_model(
                    data_path=label_to_file_path[str(trial_id_plugin_label)], ingress=self
                ).trialwise_and_metadata(trial_id)

        self._dataset_reader()

        for field, value in self.settings["trial"]["common"]["defined"].items():
            if value is not None:
                if any(
                    field in keyword_arguments and np.any(keyword_arguments[field])
                    for keyword_arguments in self._trial_id_to_keyword_arguments.values()
                ):
                    msg = f"Field, {field}, has been defined in settings, yet is also defined by the ingress method"
                    raise ValueError(msg)
                self._common_trial_keyword_arguments[field] = value

        global_reader_kwargs = {}
        # Data source priority in ascending order
        if "defined" in self.settings["trial"]["common"] and self.settings["trial"]["common"]["defined"]:
            for key, value in self.settings["trial"]["common"]["defined"].items():
                if value is None:
                    continue
                global_reader_kwargs[key] = value
        if "defined" in self.settings["reader_kwargs"] and self.settings["reader_kwargs"]["defined"]:
            global_reader_kwargs.update(self.settings["reader_kwargs"]["defined"])
        assert global_reader_kwargs, "reader_kwargs must be defined"
        self._common_trial_keyword_arguments["reader_kwargs"] = global_reader_kwargs

        if self.experiment_class.has_stages:
            for trial_class_name, dataset in self.settings["trial"]["specific"].items():
                if not dataset["defined"]:
                    continue
                if trial_class_name not in self._trial_class_name_to_keyword_arguments:
                    self._trial_class_name_to_keyword_arguments[trial_class_name] = {}
                for field, value in dataset["defined"].items():
                    if not value:
                        continue
                    trial_class_dict = self._trial_class_name_to_keyword_arguments[trial_class_name]
                    if field not in trial_class_dict or not trial_class_dict[field]:
                        self._trial_class_name_to_keyword_arguments[trial_class_name][field] = value

        self._experiment_data_defined = True

    @cached_property
    def experiment(self) -> "Experiment":
        settings_defined = set(self.settings["experiment"]["defined"])

        for kwarg_dict_name, fields in self.ingress_defined_fields.items():
            if intersection := settings_defined.intersection(fields):
                msg = f"{kwarg_dict_name}: Setting defines fields also defined by the ingress: {intersection}"
                raise ValueError(msg)

        return self.experiment_class(
            **self.settings["experiment"]["defined"],
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
        if self.settings["ingress"]["profile_runtime"]:
            with Profile() as pr:
                self.experiment.trial_label_to_df
            stats = pstats.Stats(pr)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.dump_stats(self.inspect_directory_path / "analysis_performance.prof")
        else:
            self.experiment.trial_label_to_df

        with pd.ExcelWriter(self.result_directory_path / f"{self.experiment_name}.xlsx") as writer:
            for trial_label, df in self.trial_label_to_df.items():
                df.to_excel(writer, sheet_name=trial_label)

        if len(self.trial_label_to_df) != 1:
            parquet_dir = self.result_directory_path / "parquet"
            parquet_dir.mkdir(exist_ok=True)
        else:
            parquet_dir = self.result_directory_path
        for trial_label, df in self.trial_label_to_df.items():
            df.to_parquet(parquet_dir / f"{trial_label}-{self.experiment_name}.parquet")

    def update_settings(self, delete_outdated: bool = False, dry_run: bool = False) -> dict:
        new_settings = init_settings(
            self.ingress_method,
            self.experiment_name,
            self.project_root_directory,
            self.framewise_coordinates_file_suffix,
            dry_run=True,
            silent=True,
        )

        kwargs = {"delete_outdated": delete_outdated}

        for key in ("ingress", "definition_strategies", "perimeter"):
            if key in self.settings:
                try:
                    new_settings[key] = settings.update_dictionary(self.settings[key], new_settings[key], **kwargs)
                except KeyError:
                    pass

        for key in ("reader_kwargs", "experiment"):
            if key in self.settings:
                try:
                    new_settings[key] = settings.update_defined_values(self.settings[key], new_settings[key], **kwargs)
                except KeyError:
                    pass

        if "trial" in self.settings:
            new_settings["trial"]["common"] = settings.update_defined_values(
                self.settings["trial"]["common"], new_settings["trial"]["common"], **kwargs
            )
            common_settings_between_trials = get_definable_settings(new_settings["trial"]["common"])
            for trial_class_name, trial_class_settings in new_settings["trial"]["specific"].items():
                try:
                    new_settings["trial"]["specific"][trial_class_name] = settings.update_defined_values(
                        self.settings["trial"]["specific"][trial_class_name],
                        trial_class_settings,
                        common_settings=common_settings_between_trials,
                        **kwargs,
                    )
                except KeyError:
                    pass

        for field, value in self.settings.items():
            if isinstance(value, dict):
                continue
            if field in new_settings and value:
                new_settings[field] = value

        if dry_run:
            print(json.dumps(new_settings, indent=2))
        else:
            with open(self.settings_path, "w") as out_file:
                yaml.safe_dump(new_settings, out_file, sort_keys=False)

        return new_settings

    # Plugin methods ============================== Read more about plugins in respective __init__.py file

    @cached_property
    def _global_plugins(self) -> list[Plugin, ...]:
        return [
            ingress_key_to_model[ingress_key]
            for ingress_key, strategy in self.settings["definition_strategies"].items()
            if strategy == "global"
        ]

    @cached_property
    def _metadata_plugins(self) -> list[Plugin, ...]:
        return [
            ingress_key_to_model[ingress_key]
            for ingress_key, strategy in self.settings["definition_strategies"].items()
            if strategy == "metadata"
        ]

    @cached_property
    def _trial_wise_plugins(self) -> list[Plugin]:
        return [
            ingress_key_to_model[ingress_key]
            for ingress_key, strategy in self.settings["definition_strategies"].items()
            if strategy == "trial-wise"
        ]

    def get_meter_per_pixel(self, trial_id: Optional[str | PositiveInt] = None) -> NDArrayFp64:
        match self.settings["definition_strategies"]["meters_per_pixel"]:
            case "global":
                return self._common_trial_keyword_arguments[PluginMeterPerPixel.bikipy_trial_key]
            case "metadata":
                file_label = self.metadata["MetersPerPixel"][trial_id]
                return detect_meters_per_pixel_in_perimeter_directory(self.plugin_directory_path)[file_label]
            case "trial-wise":
                return self.trial_id_to_keyword_arguments[trial_id]["meters_per_pixel"]

    # Private methods ===============================

    def _trial_class_from_stage_index(self, stage_index: str | PositiveInt) -> "Trial":
        return self.experiment_class.stage_index_to_trial_class_name[stage_index]

    def _trialwise_plugins_for_trial_id(
        self, trial_id: TrialId, trial_directory: DirectoryPath, trial_id_plugin_glob_format_string: str
    ) -> dict:
        result = {}
        for plugin_model in self._trial_wise_plugins:
            glob_str = trial_id_plugin_glob_format_string.format(
                trial_id=trial_id, plugin_code_key=plugin_model.code_key
            )
            plugin_data_files = tuple(trial_directory.glob(glob_str))
            if len(plugin_data_files) > 1:
                msg = f"Plugin {plugin_model.human_readable_index}: Only one file per trial"
                raise ValueError(msg)

            try:
                result[plugin_model.bikipy_trial_key] = plugin_model(
                    data_path=plugin_data_files[0], ingress=self
                ).trialwise_and_metadata(trial_id)
            except IndexError:
                pass

        return result

    @validate_arguments
    def _glob_coordinate_files_in_directory(self, directory_path: DirectoryPath) -> Iterable:
        return directory_path.glob(f"*{self.framewise_coordinates_file_suffix}")

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


Ingress = TypeVar("Ingress", bound=BaseIngress)


def init_settings(
    ingress_method: str,
    experiment_name: str,
    project_root_directory: DirectoryPath,
    framewise_coordinates_file_suffix: str,
    dry_run: bool = False,
    silent: bool = False,
) -> dict[str, str | dict]:
    from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS

    logger.info(f"Generating experiment configuration at {project_root_directory}")

    experiment_class = EXPERIMENT_NAME_TO_CLASS[experiment_name]
    experiment_class._ignore_unset_trial_sequence_repetition = True

    generic_settings = {
        "ingress": {
            "method": ingress_method,
            FIRST_TRIAL_IS_HABITUATION_INGRESS_FIELD: False,
            "manual_trial_sequence_repetitions": None,
            "framewise_coordinates_file_suffix": framewise_coordinates_file_suffix,
            "minimum_frame_length": 500,
            "dataset_directory": None,
            "profile_runtime": True,
        },
        "definition_strategies": {plugin.ingress_key: None for plugin in ALL_PLUGINS},
        "perimeter": {
            "label_prefix": None,
            "label_suffix": None,
            "perimeter_names_in_metadata": False,
            "radial_arm_rectangle_diagonal": None,
            "fields": extended_schema(BaseSinglePerimeter),
        },
        "reader_kwargs": extended_schema(DeepLabCutReader, with_required=False),
        "trial": extended_group_schema(experiment_class.trial_sequence),
        "experiment": extended_schema(experiment_class),
        "debug": {"activate_debugging": False, "no_numba": False, "no_process_pooling": False},  # TODO: Implement
        "immutable": {
            "metadata_filename": "metadata.xlsx",
            "experiment_class": experiment_name,
        },
    }

    if experiment_class.has_stages:
        generic_settings["immutable"][
            "stage_index_to_trial_class_name"
        ] = experiment_class.stage_index_to_trial_class_name
    else:
        generic_settings["immutable"]["trial_sequence"] = experiment_class.trial_class_names

    if dry_run:
        if not silent:
            print(json.dumps(generic_settings, indent=2))
    else:
        settings_path = get_project_settings_path(project_root_directory)
        with open(settings_path, "w") as out_file:
            yaml.safe_dump(generic_settings, out_file, sort_keys=False)

    return generic_settings


@validate_arguments
def auto_define_ingress_object(project_root_directory: DirectoryPath) -> Ingress:
    from bikipy.ingress import INGRESS_METHOD_NAME_TO_INGRESS_CLASS

    with open(project_root_directory / "settings.yaml", "r") as in_file:
        settings = yaml.safe_load(in_file)
    return INGRESS_METHOD_NAME_TO_INGRESS_CLASS[settings["ingress"]["method"]](
        project_root_directory=project_root_directory
    )


def analyze_and_save(project_root_directory: DirectoryPath):
    BaseIngress.from_project_root_directory(project_root_directory).save_analysis_data()
