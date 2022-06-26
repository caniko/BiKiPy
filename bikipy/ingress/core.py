import json
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property
from logging import getLogger
from typing import Any, Callable, ClassVar, Hashable, Iterable, Optional, TypeVar

import numpy as np
import openpyxl
import pandas as pd
import yaml
from pydantic import DirectoryPath, FilePath, PositiveInt, validate_arguments

from bikipy.behaviour.base import Experiment
from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS
from bikipy.core.base_class import BaseBikipy
from bikipy.core.typing import NDArrayFp64
from bikipy.ingress.plugin import ingress_key_to_plugin_name
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
from bikipy.perimeter.base import BasePerimeter
from bikipy.reader import DeepLabCutReader
from bikipy.utils.collection_utils import copycat_assumes_levels_of_icon

logger = getLogger(__name__)


class BaseIngress(BaseBikipy, ABC):
    project_root_directory: DirectoryPath

    _experiment_data_defined: bool = False
    _trial_id_to_trial_class_name: dict[Hashable, str] = {}
    _common_trial_keyword_arguments: dict[str, Any] = {}
    _trial_id_to_keyword_arguments: dict[Hashable, dict[str, Any]] = {}
    _trial_class_name_to_keyword_arguments: dict[str, Any] = {}

    ingress_method: ClassVar[str]

    @abstractmethod
    def _ingress_reader(self):
        ...

    @abstractmethod
    def verify_project_structure(self):
        ...

    @classmethod
    def from_project_root_directory(cls, project_root_directory: DirectoryPath):
        kwargs = {"project_root_directory": project_root_directory}
        match auto_define_ingress_object(project_root_directory).ingress_method:
            case "animal":
                from bikipy.ingress import AnimalIngress

                return AnimalIngress(**kwargs)

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
    def _metadata_sheet_names(self):
        return openpyxl.load_workbook(self.metadata_path, read_only=True).sheetnames

    @property
    def settings(self) -> dict:
        return load_settings(self.project_root_directory)

    @cached_property
    def animal_metadata(self):
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

        elif not self._metadata_plugins and self.settings["ingress"]["stageful_metadata"]:
            msg = (
                f"Either trial_id or animal_sequence sheet must be defined in metadata "
                f"when using metadata plugins that are stageful:\n{self._metadata_plugins}"
            )
            raise ValueError(msg)
        elif self.animal_metadata is not None:
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
            raise RuntimeError()

        return trial_id_df

    @property
    def settings_path(self) -> FilePath:
        return get_project_settings_path(self.project_root_directory)

    @property
    def metadata_path(self) -> FilePath:
        return infer_metadata_path(self.project_root_directory)

    @property
    def dataset_directory_path(self) -> DirectoryPath:
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
        if self._metadata_plugins:
            self._trial_id_to_keyword_arguments = defaultdict(dict)
            for plugin_info in self._metadata_plugins:
                label_to_file_path = {
                    file_path.stem.split("-")[-1]: file_path
                    for file_path in self.plugin_directory_path.glob(f"{plugin_info['code_key']}*")
                }
                for trial_id, row in self.metadata.iterrows():
                    self._trial_id_to_keyword_arguments[trial_id][plugin_info["bikipy_trial_key"]] = plugin_info[
                        "file_path_to_value"
                    ](label_to_file_path[row[plugin_info["human_readable_index"]]], self, trial_id)

        self._ingress_reader()

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
    def _trial_wise_plugins(self) -> list[str, ...]:
        return [
            ingress_key_to_plugin_name[ingress_key]
            for ingress_key, strategy in self.settings["ingress"].items()
            if strategy == "trial-wise"
        ]

    @cached_property
    def _metadata_plugins(self) -> list[dict[str, str | Callable], ...]:
        return [
            ingress_key_to_plugin_name[ingress_key]
            for ingress_key, strategy in self.settings["ingress"].items()
            if strategy == "metadata"
        ]

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

    # Client-side functions ===============================

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
        self.experiment.combined_feature_motion_df.to_excel(
            self.result_directory_path / "animal_id_indexed_result_data.xlsx"
        )

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

    # Plugin methods ==============================

    def get_meter_per_pixel(self, trial_id: str | PositiveInt) -> NDArrayFp64:
        match self.settings["ingress"]["meters_per_pixel_definition_strategy"]:
            case "global_perimeter":
                return first_meters_per_pixel_in_perimeter_directory(self.plugin_directory_path)
            case "metadata":
                file_label = self.metadata["MetersPerPixel"][trial_id]
                return detect_meters_per_pixel_in_perimeter_directory(self.plugin_directory_path)[file_label]
            case "trial-wise":
                return self.trial_id_to_keyword_arguments[trial_id]["meters_per_pixel"]

    # Private methods ===============================

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
        "ingress": {
            "stageful_metadata": False,
            "skip_absent_trials_absent_from_metadata_index": False,
            "meters_per_pixel_definition_strategy": "global_perimeter",
            "perimeter_definition_strategy": "metadata",
            "center_definition_strategy": None,
        },
        "perimeter": {
            "label_prefix": None,
            "label_suffix": None,
            "perimeter_names_in_metadata": False,
            "fields": extended_schema(BasePerimeter),
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
