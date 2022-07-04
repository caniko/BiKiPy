import json
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, ClassVar, Hashable, Iterable, TypeVar

import numpy as np
import pandas as pd
import yaml
from pydantic import DirectoryPath, FilePath, PositiveInt, validate_arguments

from bikipy.behaviour.base import Experiment, Trial
from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS
from bikipy.core.base_class import BaseBikipy
from bikipy.core.typing import NDArrayFp64
from bikipy.ingress.plugin import PLUGIN_NAME_TO_MODEL, ingress_key_to_model, PluginPerimeter, PluginRadial
from bikipy.ingress.plugin.base import Plugin
from bikipy.ingress.plugin.meters_per_pixel import (
    detect_meters_per_pixel_in_perimeter_directory,
    first_meters_per_pixel_in_perimeter_directory,
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
from bikipy.perimeter.base import BaseSinglePerimeter
from bikipy.reader import DeepLabCutReader
from bikipy.utils.collection_utils import copycat_assumes_levels_of_icon
from bikipy.utils.misc import sheet_names_from_path, dict_deepmerge

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

    _experiment_data_defined: bool = False
    _trial_id_to_trial_class_name: dict[Hashable, str] = {}
    _common_trial_keyword_arguments: dict[str, Any] = {}
    _trial_id_to_keyword_arguments: dict[Hashable, dict[str, Any]] = {}
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

    @property
    def experiment_class(self):
        try:
            return EXPERIMENT_NAME_TO_CLASS[self.experiment_name]
        except KeyError:
            msg = (
                f"experiment_class in settings is set to an invalid value: "
                f"{self.settings['immutable']['experiment_class']}; "
                f"this value should not be changed after initialization of the project."
            )
            raise ValueError(msg)

    @cached_property
    def experiment_class_kwargs(self):
        return {
            "trial_id_to_trial_class_name": self.trial_id_to_trial_class_name,
            "common_trial_keyword_arguments": self.common_trial_keyword_arguments,
            "trial_id_to_keyword_arguments": self.trial_id_to_keyword_arguments,
            "trial_class_name_to_keyword_arguments": self.trial_class_name_to_keyword_arguments,
        }

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
        return self._trial_id_to_keyword_arguments

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
        if "trial_id" in self._metadata_sheet_names:
            trial_id_df = pd.read_excel(
                self.metadata_path,
                sheet_name="trial_id",
                index_col=0,
            )
            trial_id_df.index.names = ["Trial"]

            assert "Animal" in trial_id_df, "Animal ID column must be present trial_id and animal metadata sheets"

            if self.animal_metadata is not None:
                trial_id_df = trial_id_df.join(self.animal_metadata, how="inner")

        elif "animal_sequence" in self._metadata_sheet_names:
            trial_id_df = pd.read_excel(
                self.metadata_path,
                sheet_name="animal_sequence",
                index_col=[0, 1],
            )
            trial_id_df.index.names = ["Animal", "Sequence"]
            if self.animal_metadata is not None:
                trial_id_df = trial_id_df.join(self.animal_metadata, how="inner")

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
                for _, row in self.animal_metadata.iterrows():
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
                trial_id_df = pd.concat(
                    [self.animal_metadata for _ in range(self.experiment_class.trial_sequence_length)], axis=0
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

        return trial_id_df

    @property
    def settings_path(self) -> FilePath:
        return get_project_settings_path(self.project_root_directory)

    @property
    def metadata_path(self) -> FilePath:
        return infer_metadata_path(self.project_root_directory)

    @property
    def dataset_directory_path(self) -> DirectoryPath:
        if self.settings["manual_dataset_directory"]:
            if not (path := DirectoryPath(self.settings["manual_dataset_directory"])).exists():
                msg = f"manual_dataset_directory must exist: {path}"
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

    @property
    def kinematic_data_file_extension(self):
        return self.settings["immutable"]["kinematic_data_file_extension"]

    # Backend functions =================================

    def _define_experiment_data_if_not_defined(self):
        if not self._experiment_data_defined:
            self._define_experiment_data()

    def _define_experiment_data(self) -> None:
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
            global_reader_kwargs.update(self.settings["trial"]["common"]["defined"])
        if "defined" in self.settings["reader_kwargs"] and self.settings["reader_kwargs"]["defined"]:
            global_reader_kwargs.update(self.settings["reader_kwargs"]["defined"])
        assert global_reader_kwargs, "reader_kwargs must be defined"
        self._common_trial_keyword_arguments["reader_kwargs"] = global_reader_kwargs

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
    def experiment(self) -> Experiment:
        intersection = set(self.settings["experiment"]["defined"]).intersection(self.experiment_class_kwargs)
        if intersection:
            msg = f"The setting defines fields defined by the ingress method:\n{intersection}"
            raise ValueError(msg)
        return self.experiment_class(
            **self.settings["experiment"]["defined"],
            **self.experiment_class_kwargs,
            inspect_directory=self.inspect_directory_path,
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
        return {
            trial_label: df.join(self.metadata, how="inner")
            for trial_label, df in self.experiment.trial_label_to_df.items()
        }

    @cached_property
    def animal_analysis_df(self) -> pd.DataFrame:
        df = pd.concat(
            (self.animal_metadata_fit_to_combined_feature_motion_df, self.experiment.combined_feature_motion_df),
            axis=1,
        )
        df.columns.names = (
            ["Stage", "Feature", "Location/Category"]
            if self.experiment.has_trials_in_stages
            else ["Feature", "Location/Category"]
        )
        df.index.names = ["Animal ID"]

        return df

    def save_analysis_data(self):
        # self.analysis_df.to_parquet(self.result_directory_path / f"animal_id_indexed_result_data.parquet")
        # self.experiment.combined_feature_motion_df.to_excel(
        #     self.result_directory_path / "animal_id_indexed_result_data.xlsx"
        # )
        self.trial_label_to_df
        with pd.ExcelWriter(self.result_directory_path / "trial_id_indexed_result_data.xlsx") as writer:
            for trial_label, df in self.trial_label_to_df.items():
                df.to_excel(writer, sheet_name=trial_label)

    def update_settings(self, delete_outdated: bool = False, dry_run: bool = False) -> dict:
        new_settings = init_settings(
            self.ingress_method,
            self.experiment_name,
            self.project_root_directory,
            self.kinematic_data_file_extension,
            dry_run=True,
            silent=True,
        )

        kwargs = {"delete_outdated": delete_outdated}

        for key in ("ingress", "perimeter"):
            if key in self.settings:
                new_settings[key] = settings.update_dictionary(self.settings[key], new_settings[key], **kwargs)

        if "reader_kwargs" in self.settings:
            new_settings["reader_kwargs"] = settings.update_defined_values(
                self.settings["reader_kwargs"], new_settings["reader_kwargs"], **kwargs
            )

        if "trial" in self.settings:
            new_settings["trial"]["common"] = settings.update_defined_values(
                self.settings["trial"]["common"], new_settings["trial"]["common"], **kwargs
            )
            common_settings_between_trials = get_definable_settings(new_settings["trial"]["common"])
            for trial_class_name, trial_class_settings in new_settings["trial"]["specific"].items():
                new_settings["trial"]["specific"][trial_class_name] = settings.update_defined_values(
                    self.settings["trial"]["specific"][trial_class_name],
                    trial_class_settings,
                    common_settings=common_settings_between_trials,
                    **kwargs,
                )

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
            for ingress_key, strategy in self.settings["ingress"].items()
            if strategy == "global"
        ]

    @cached_property
    def _metadata_plugins(self) -> list[Plugin, ...]:
        return [
            ingress_key_to_model[ingress_key]
            for ingress_key, strategy in self.settings["ingress"].items()
            if strategy == "metadata"
        ]

    @cached_property
    def plugin_to_label_to_global_plugin_objects(self) -> dict[str, dict[str, Plugin]]:
        result = {}
        for plugin_model in self._global_plugins:
            result[plugin_model[plugin_model.bikipy_trial_key]] = {
                file_path.stem.split("-")[-1]: plugin_model(data_path=file_path, ingress=self)
                for file_path in self.plugin_directory_path.glob(f"{plugin_model.code_key}*")
            }
        return result

    @cached_property
    def plugin_to_label_to_metadata_plugin_objects(self) -> dict[str, dict[str, Plugin]]:
        result = {}
        for plugin_model in self._metadata_plugins:
            result[plugin_model[plugin_model.bikipy_trial_key]] = {
                file_path.stem.split("-")[-1]: plugin_model(data_path=file_path, ingress=self)
                for file_path in self.plugin_directory_path.glob(f"{plugin_model.code_key}*")
            }
        return result

    @cached_property
    def metadata_and_global_perimeter_and_perimeter_set(self) -> dict[str, dict[str, Plugin]]:
        return dict_deepmerge(
            self.plugin_to_label_to_global_plugin_objects, deepcopy(self.plugin_to_label_to_metadata_plugin_objects)
        )

    @cached_property
    def _trial_wise_plugins(self) -> list[Plugin]:
        return [
            ingress_key_to_model[ingress_key]
            for ingress_key, strategy in self.settings["ingress"].items()
            if strategy == "trial-wise"
        ]

    # Private methods ===============================

    def _trial_class_from_stage_index(self, stage_index: str | PositiveInt) -> Trial:
        return self.experiment_class.stage_index_to_trial_class_name[stage_index]

    @staticmethod
    def _get_id_from_path_stem(path: Path) -> str | PositiveInt:
        stem = path.stem
        if "-" in stem:
            stem = path.stem.split("-")
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
    kinematic_data_file_extension: str,
    dry_run: bool = False,
    silent: bool = False,
) -> dict[str, str | dict]:
    logger.info(f"Generating experiment configuration at {project_root_directory}")

    experiment_class = EXPERIMENT_NAME_TO_CLASS[experiment_name]

    generic_settings = {
        "ingress_method": ingress_method,
        "manual_dataset_directory": None,
        "ingress": {
            "meters_per_pixel_definition_strategy": "global",
            "perimeter_definition_strategy": "metadata",
            "video_definition_strategy": None,
            "center_definition_strategy": None,
        },
        "perimeter": {
            "label_prefix": None,
            "label_suffix": None,
            "perimeter_names_in_metadata": False,
            "perimeter_mapper_key": "label",
            "fields": extended_schema(BaseSinglePerimeter),
        },
        "reader_kwargs": extended_schema(DeepLabCutReader, with_required=False),
        "trial": extended_group_schema(experiment_class.trial_classes),
        "experiment": extended_schema(experiment_class),
        "immutable": {
            "metadata_filename": "metadata.xlsx",
            "kinematic_data_file_extension": kinematic_data_file_extension,
            "experiment_class": experiment_name,
        },
    }

    if experiment_class.has_trials_in_stages:
        generic_settings["immutable"][
            "stage_index_to_trial_class_name"
        ] = experiment_class.stage_index_to_trial_class_name
    else:
        generic_settings["immutable"]["trial_classes"] = experiment_class.trial_class_names

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
    return INGRESS_METHOD_NAME_TO_INGRESS_CLASS[settings["ingress_method"]](
        project_root_directory=project_root_directory
    )


def analyze_and_save(project_root_directory: DirectoryPath):
    BaseIngress.from_project_root_directory(project_root_directory).save_analysis_data()
