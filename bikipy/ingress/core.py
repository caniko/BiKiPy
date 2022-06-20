import json
from abc import ABC, abstractmethod
from functools import cached_property
from logging import getLogger
from typing import Any, Callable, ClassVar, Hashable, Iterable, Optional, TypeVar

import numpy as np
import openpyxl
import pandas as pd
import yaml
from pydantic import DirectoryPath, FilePath, validate_arguments, PositiveInt

from bikipy.behaviour.base import Experiment
from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS
from bikipy.core.base_class import BikipyBase
from bikipy.core.typing import NDArrayFp64
from bikipy.ingress.plugin import ingress_key_to_plugin_name
from bikipy.ingress.plugin.meters_per_pixel import (
    detect_meters_per_pixel_in_perimeter_directory,
    first_meters_per_pixel_in_perimeter_directory,
)
from bikipy.ingress.plugin.perimeter import get_perimeter_data, get_perimeter_name_df
from bikipy.ingress.utils import settings
from bikipy.ingress.utils.io import (
    get_dataset_directory_path,
    get_inspect_directory_path,
    get_plugin_directory_path,
    get_project_settings_path,
    load_settings,
)
from bikipy.ingress.utils.model_schema import extended_group_schema, extended_schema
from bikipy.ingress.utils.settings import get_definable_settings
from bikipy.perimeter.base import (
    BasePerimeter,
    PerimeterSet,
    StringPerimeterShapes,
    perimeter_set_from_makesense,
)
from bikipy.reader import DeepLabCutReader
from bikipy.utils.collection_utils import copycat_assumes_levels_of_icon

logger = getLogger(__name__)


class BaseIngress(BikipyBase, ABC):
    project_root_directory: DirectoryPath

    _trial_id_to_trial_class_name: dict[Hashable, str] = {}
    _common_trial_keyword_arguments: dict[str, Any] = {}
    _trial_id_to_keyword_arguments: dict[Hashable, dict[str, Any]] = {}
    _trial_class_name_to_keyword_arguments: dict[str, Any] = {}

    _experiment_data_defined: bool = False

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

    @property
    def settings(self) -> dict:
        return load_settings(self.project_root_directory)

    @cached_property
    def metadata(self) -> pd.DataFrame:
        metadata_path = next(self.project_root_directory.glob("metadata.*"))
        sheet_names = openpyxl.load_workbook(metadata_path, read_only=True).sheetnames

        if "animal" in sheet_names:
            animal_df = pd.read_excel(
                metadata_path,
                sheet_name="animal",
                index_col=0,
            )
            animal_df.columns.names = ["Animal"]
        else:
            animal_df = None

        if "trial_id" in sheet_names:
            trial_id_df = pd.read_excel(
                metadata_path,
                sheet_name="trial_id",
                index_col=0,
            )
            trial_id_df.index.names = ["Trial"]

            assert "Animal" in trial_id_df, "Animal ID column must be present trial_id and animal metadata sheets"

            trial_id_df = trial_id_df.join(animal_df, how="inner")

        elif "animal_sequence" in sheet_names:
            trial_id_df = pd.read_excel(
                metadata_path,
                sheet_name="animal_sequence",
                index_col=[0, 1],
            )
            trial_id_df.index.names = ["Animal", "Sequence"]
            trial_id_df = trial_id_df.join(animal_df, how="inner")

            trial_id_df["Animal"] = trial_id_df.index.get_level_values(level="Animal")

            trial_id_df.index = trial_id_df.index.map(lambda idx: f"{idx[0]}_{idx[1]}")
            trial_id_df.index.names = ["Trial"]

        else:
            msg = "Either trial_id or animal_sequence sheet must be defined in metadata"
            raise ValueError(msg)

        return trial_id_df

    @property
    def settings_path(self) -> FilePath:
        return get_project_settings_path(self.project_root_directory)

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
            self._trial_id_to_keyword_arguments = dict.fromkeys(self.metadata.index.values, dict())
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
    def metadata_fit_to_combined_feature_motion_df(self) -> pd.DataFrame:
        if self.metadata.columns.nlevels >= self.experiment.combined_feature_motion_df.columns.nlevels:
            return self.metadata
        return copycat_assumes_levels_of_icon(self.metadata, self.experiment.combined_feature_motion_df, "Global")

    # Client-side functions ===============================

    @cached_property
    def analysis_df(self) -> pd.DataFrame:
        df = pd.concat(
            (self.metadata_fit_to_combined_feature_motion_df, self.experiment.combined_feature_motion_df),
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
        self.analysis_df
        self.analysis_df.to_excel(self.result_directory_path / "animal_id_indexed_result_data.xlsx")

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

        new_settings["ingress"] = settings.update_dictionary(
            self.settings["ingress"], new_settings["ingress"], **kwargs
        )
        new_settings["perimeter"] = settings.update_dictionary(
            self.settings["perimeter"], new_settings["perimeter"], **kwargs
        )
        new_settings["reader_kwargs"] = settings.update_defined_values(
            self.settings["reader_kwargs"], new_settings["reader_kwargs"], **kwargs
        )

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

    @validate_arguments
    def first_perimeter_set_from_makesense(
        self,
        perimeter_path: FilePath,
        trial_id: str | PositiveInt,
        manual_shape: Optional[StringPerimeterShapes] = None,
    ) -> PerimeterSet:
        shape, label = get_perimeter_data(perimeter_path)

        image_name_to_perimeter_set = perimeter_set_from_makesense(
            perimeter_path, manual_shape or shape, manual_meters_per_pixel=self._get_meter_per_pixel(trial_id)
        )

        perimeter_set = tuple(image_name_to_perimeter_set.values())[0]
        perimeter_set.apply_label_prefix_suffix(
            self.settings["perimeter"]["label_prefix"], self.settings["perimeter"]["label_suffix"]
        )
        for perimeter in perimeter_set.all_perimeters:
            for field, value in self.settings["perimeter"]["fields"]["defined"].items():
                if value is not None:
                    perimeter.__setattr__(field, value)

        return perimeter_set

    def _get_meter_per_pixel(self, trial_id: str | PositiveInt) -> NDArrayFp64:
        match self.settings["ingress"]["meters_per_pixel_definition_strategy"]:
            case "global_perimeter":
                return first_meters_per_pixel_in_perimeter_directory(self.plugin_directory_path)
            case "metadata":
                file_label = self.metadata["MetersPerPixel"][trial_id]
                return detect_meters_per_pixel_in_perimeter_directory(self.plugin_directory_path)[file_label]
            case "trial-wise":
                return self.trial_id_to_keyword_arguments[trial_id]["manual_meters_per_pixel"]

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
            "perimeter_naming_strategy": None,
            "center_definition_strategy": None,
        },
        "perimeter": {
            "label_prefix": None,
            "label_suffix": None,
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
