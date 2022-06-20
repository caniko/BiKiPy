import json
from abc import ABC, abstractmethod
from copy import copy
from functools import cached_property
from typing import Any, Callable, ClassVar, Hashable, Iterable, Optional, TypeVar

import numpy as np
import pandas as pd
import yaml
from pydantic import DirectoryPath, FilePath, validate_arguments

from bikipy.behaviour.base import Experiment
from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS
from bikipy.core.base_class import BikipyBase
from bikipy.ingress.plugin import PLUGIN_NAME_TO_KEYRING
from bikipy.ingress.plugin.center import detect_center_in_perimeter_directory
from bikipy.ingress.plugin.meters_per_pixel import (
    detect_meters_per_pixel_in_perimeter_directory,
)
from bikipy.ingress.plugin.perimeter import get_perimeter_data, get_perimeter_name_df
from bikipy.ingress.utils import settings
from bikipy.ingress.utils.io import (
    get_dataset_directory_path,
    get_inspect_directory_path,
    get_perimeter_directory_path,
    get_project_settings_path,
    load_settings,
)
from bikipy.ingress.utils.model_schema import extended_group_schema, extended_schema
from bikipy.ingress.utils.settings import get_definable_settings
from bikipy.perimeter.base import (
    AnyPerimeter,
    BasePerimeter,
    PerimeterSet,
    StringPerimeterShapes,
    perimeter_set_from_makesense,
)
from bikipy.reader import DeepLabCutReader
from bikipy.utils.collection_utils import copycat_assumes_levels_of_icon


class BaseIngress(BikipyBase, ABC):
    project_root_directory: DirectoryPath

    _trial_id_to_trial_class_name: dict[Hashable, str] = {}
    _common_trial_keyword_arguments: dict[str, Any] = {}
    _trial_id_to_keyword_arguments: dict[Hashable, dict[str, Any]] = {}
    _trial_class_name_to_keyword_arguments: dict[str, Any] = {}

    _metadata_index_to_trial_id: dict = {}
    _experiment_data_defined: bool = False

    ingress_method: ClassVar[str]

    @abstractmethod
    def _experiment_class_kwargs_and_metadata_index_to_trial_id_and_metadata_index_to_trial_id_define_function(self):
        ...

    @abstractmethod
    def verify_project_structure(self):
        ...

    @property
    @abstractmethod
    def method_settings(self):
        ...

    @classmethod
    def from_project_root_directory(cls, project_root_directory: DirectoryPath):
        kwargs = {"project_root_directory": project_root_directory}
        match auto_define_ingress_object(project_root_directory).ingress_method:
            case "sequence":
                from bikipy.ingress import SequenceIngress

                return SequenceIngress(**kwargs)

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
    def reader_kwargs(self) -> dict:
        result = {}
        # Data source priority in ascending order
        if "reader_kwargs" in self.settings["trial"]["common"]["defined"]:
            result.update(self.settings["trial"]["common"]["defined"])
        if "reader_kwargs" in self.settings["reader_kwargs"]:
            result.update(self.settings["reader_kwargs"]["defined"])
        return result

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

    def _register_to_trial_class_name_to_keyword_arguments(self, setting_key: str, from_stage_index: bool = False):
        for trial_class_name, value in self.settings.items():
            if from_stage_index:
                trial_class_name = self.experiment_class.stage_index_to_trial_class_name[trial_class_name]
            assert trial_class_name in self.experiment_class.trial_class_names

            if trial_class_name not in self._trial_class_name_to_keyword_arguments:
                self._trial_class_name_to_keyword_arguments[trial_class_name] = {}
            self._trial_class_name_to_keyword_arguments[trial_class_name][setting_key] = value

    @cached_property
    def activate_plugins_and_parse_metadata_plugins(self) -> dict[str, dict]:
        """
        This method initializes all the plugins that have been activated in the settings.yaml file.

        Plugins that have metadata strategies often requires knowledge about the state. The state includes whatever
        is relevant to the ingress method during parameter retrieval, but most often is just limited to
        trial_id or animal_id. See get_plugin_parameter for more information.

        :return: dict[str, dict]: plugin_name
        """
        result = {}
        match self.settings["ingress"]["meters_per_pixel_definition_strategy"]:
            case "metadata":
                result["meters_per_pixel"] = detect_meters_per_pixel_in_perimeter_directory(
                    self.perimeter_directory_path
                )
            case "trial-wise":
                # TODO
                pass
            case None:
                pass
            case _:
                self._raise_unsupported_plugin_method(
                    "meters_per_pixel_definition_strategy", ("metadata", "trial-wise", None)
                )

        match self.settings["ingress"]["perimeter_definition_strategy"]:
            case "metadata":
                pass
            case "trial-wise" | None:
                pass
            case _:
                self._raise_unsupported_plugin_method("perimeter_definition_strategy", ("metadata", None))

        match self.settings["ingress"]["perimeter_naming_strategy"]:
            case "metadata":
                pass
            case None:
                pass
            case _:
                self._raise_unsupported_plugin_method("perimeter_naming_strategy", ("metadata", None))

        match self.settings["ingress"]["center_definition_strategy"]:
            case "metadata":
                result["center"] = detect_center_in_perimeter_directory(self.perimeter_directory_path)
            case "trial-wise":
                # TODO
                pass
            case None:
                pass
            case _:
                self._raise_unsupported_plugin_method("center_definition_strategy", ("metadata", "trial-wise", None))

        return result

    # I/O ============================

    @property
    def settings(self) -> dict:
        return load_settings(self.project_root_directory)

    @cached_property
    def metadata(self) -> pd.DataFrame:
        df = pd.read_excel(
            next(self.project_root_directory.glob("metadata.*")),
            index_col=0,
            header=(0, 1) if self.stageful_metadata else 0,
        )
        df.columns.names = ["Feature", "Location_Category"] if self.stageful_metadata else ["Feature"]
        return df

    @property
    def settings_path(self) -> FilePath:
        return get_project_settings_path(self.project_root_directory)

    @property
    def dataset_directory_path(self) -> DirectoryPath:
        return get_dataset_directory_path(self.project_root_directory)

    @property
    def perimeter_directory_path(self) -> DirectoryPath:
        return get_perimeter_directory_path(self.project_root_directory)

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
    def stageful_metadata(self):
        return self.settings["ingress"]["stageful_metadata"]

    @property
    def kinematic_data_file_extension(self):
        return self.settings["immutable"]["kinematic_data_file_extension"]

    # Backend functions =================================

    def _define_experiment_data_if_not_defined(self):
        if not self._experiment_data_defined:
            self._define_experiment_data()

    def _define_experiment_data(self) -> None:
        self._experiment_class_kwargs_and_metadata_index_to_trial_id_and_metadata_index_to_trial_id_define_function()

        for field, value in self.settings["trial"]["common"]["defined"].items():
            if value is not None:
                if any(
                    field in keyword_arguments and np.any(keyword_arguments[field])
                    for keyword_arguments in self._trial_id_to_keyword_arguments.values()
                ):
                    msg = f"Field, {field}, has been defined in settings, yet is also defined by the ingress method"
                    raise ValueError(msg)
                self._common_trial_keyword_arguments[field] = value

        for trial_class_name, dataset in self.settings["trial"]["specific"].items():
            if not dataset["defined"]:
                continue
            if trial_class_name not in self._trial_class_name_to_keyword_arguments:
                self._trial_class_name_to_keyword_arguments[trial_class_name] = {}
            for field, value in dataset["defined"].items():
                trial_class_dict = self._trial_class_name_to_keyword_arguments[trial_class_name]
                if field not in trial_class_dict or not trial_class_dict[field]:
                    self._trial_class_name_to_keyword_arguments[trial_class_name][field] = value

        if self.settings["ingress"]["meters_per_pixel_definition_strategy"] == "global_perimeter":
            self._common_trial_keyword_arguments[
                "manual_meters_per_pixel"
            ] = detect_meters_per_pixel_in_perimeter_directory(self.perimeter_directory_path, return_first=True)

        self._experiment_data_defined = True

    @cached_property
    def detect_perimeters_in_perimeter_directory(self) -> list:
        detection_data = []
        for filename in self.perimeter_directory_path.glob("perimeter-*"):
            perimeter_path = self.project_root_directory / filename

            shape, label = get_perimeter_data(perimeter_path)
            data = {
                "perimeter": self.first_perimeter_set_from_makesense(perimeter_path),
                "label": label,
                "shape": shape,
            }

            detection_data.append(data)
        if not detection_data:
            msg = (
                "No perimeter data was found. Set perimeter strategy to None or "
                'revise perimeter filenames to the correct format, "perimeter-{label}"'
            )
            raise ValueError(msg)
        return detection_data

    def generate_label_to_object_field(self):
        return {
            perimeter_data.pop("label"): perimeter_data
            for perimeter_data in self.detect_perimeters_in_perimeter_directory
        }

    @validate_arguments
    def first_perimeter_set_from_makesense(
        self,
        perimeter_path: FilePath,
        manual_shape: Optional[StringPerimeterShapes] = None,
    ) -> PerimeterSet:
        shape, label = get_perimeter_data(perimeter_path)

        image_name_to_perimeter_set = perimeter_set_from_makesense(
            perimeter_path, manual_shape or shape, manual_meters_per_pixel=self.settings
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

    def register_perimeter_to_trial_id(self, trial_id: Hashable, label_to_perimeter: dict[str, AnyPerimeter]):
        if self.settings["ingress"]["perimeter_naming_strategy"] == "metadata":
            df = get_perimeter_name_df(self.project_root_directory)
            for label, perimeter in label_to_perimeter.items():
                new_label = df.loc[trial_id, label]

                new_perimeter = copy(perimeter)
                new_perimeter.label = new_label

                self._trial_id_to_keyword_arguments[trial_id][new_label] = new_perimeter
        else:
            self._trial_id_to_keyword_arguments[trial_id].update(label_to_perimeter)

    def get_plugin_parameter(self, parameter_label: str, metadata_index_getter: Callable, *args, **kwargs):
        """
        Method that retrieves plugin parameters from the metadata file. This requires the use of a metadata_index_getter
        that is defined for each ingress method.

        One may define constants ahead of function call with `functools.partial` to the metadata_index_getter.
        Additionally, *args and **kwargs in this function are relayed to the getter.

        :param parameter_label:
        :param metadata_index_getter:
        :return:
        """
        parameter_indexes = PLUGIN_NAME_TO_KEYRING[parameter_label]
        return self.activate_plugins_and_parse_metadata_plugins[parameter_indexes["code_key"]][
            metadata_index_getter(parameter_indexes["human_readable_index"], *args, **kwargs)
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
            if self.experiment.is_trial_sequence
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
            self.project_root_directory,
            self.experiment_class,
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
        # new_settings["reader_kwargs"] = settings.update_defined_values(
        #     self.settings["reader_kwargs"], new_settings["reader_kwargs"], **kwargs
        # )

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

    # Private methods ===============================

    @staticmethod
    def _raise_unsupported_plugin_method(plugin_name: str, supported_methods: Iterable[str]) -> None:
        msg = f"{plugin_name} only supports: {', '.join(supported_methods)}"
        raise ValueError(msg)


Ingress = TypeVar("Ingress", bound=BaseIngress)


def init_settings(
    ingress_method: str,
    project_root_directory: DirectoryPath,
    experiment_class: Any,
    kinematic_data_file_extension: str,
    method_kwargs: Optional[dict],
    method_immutable: Optional[dict],
    dry_run: bool = False,
    silent: bool = False,
) -> dict[str, str | dict]:
    generic_settings = {
        "ingress_method": ingress_method,
        "ingress": {
            **(method_kwargs or {}),
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
            "experiment_class": experiment_class.__name__,
            "trial_classes/stages": experiment_class.trial_class_names,
            "method_specific": method_immutable,
        },
    }

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
