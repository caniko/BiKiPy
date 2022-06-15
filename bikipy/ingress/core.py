import json
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Callable, Hashable, Optional, TypeVar

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
from bikipy.ingress.plugin.perimeter import get_perimeter_data
from bikipy.ingress.utils.io import (
    get_dataset_directory_path,
    get_inspect_directory_path,
    get_perimeter_directory_path,
    get_project_settings_path,
    load_settings,
)
from bikipy.ingress.utils.model_schema import extended_group_schema, extended_schema
from bikipy.perimeter.base import (
    AnyPerimeter,
    BasePerimeter,
    PerimeterSet,
    StringPerimeterShapes,
)
from bikipy.perimeter.polygon.base import PolygonPerimeter
from bikipy.perimeter.radial.circle import CirclePerimeter
from bikipy.reader import DeepLabCutReader


class BaseIngress(BikipyBase, ABC):
    project_root_directory: DirectoryPath

    _trial_id_to_keyword_arguments: dict[Hashable, dict[str, Any]] = {}
    _common_trial_keyword_arguments: dict[str, Any] = {}
    _trial_id_to_trial_class_name: dict[Hashable, str] = {}
    _metadata_index_to_trial_id: dict = {}

    _experiment_data_defined: bool = False

    @abstractmethod
    def _experiment_class_kwargs_and_metadata_index_to_trial_id_and_metadata_index_to_trial_id_define_function(self):
        ...

    @abstractmethod
    def verify_project_structure(self):
        ...

    @property
    def experiment_class(self):
        try:
            return EXPERIMENT_NAME_TO_CLASS[self.settings["immutable"]["experiment_class"]]
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
            "trial_id_to_keyword_arguments": self.trial_id_to_keyword_arguments,
            "common_trial_keyword_arguments": self.common_trial_keyword_arguments,
            "trial_id_to_trial_class_name": self.trial_id_to_trial_class_name,
        }

    @property
    def trial_id_to_keyword_arguments(self):
        self._define_experiment_data_if_not_defined()
        return self._trial_id_to_keyword_arguments

    @property
    def common_trial_keyword_arguments(self):
        if not self._experiment_data_defined:
            self._define_experiment_data()
        return self._common_trial_keyword_arguments

    @property
    def trial_id_to_trial_class_name(self):
        if not self._experiment_data_defined:
            self._define_experiment_data()
        return self._trial_id_to_trial_class_name

    @property
    def metadata_index_to_trial_id(self):
        if not self._experiment_data_defined:
            self._define_experiment_data()
        return self._metadata_index_to_trial_id

    @property
    def ingress_method(self):
        return self.settings["ingress_method"]

    @cached_property
    def metadata_plugin_name_to_label_to_parameter(self) -> dict[str, dict]:
        result = {}
        if self.settings["ingress"]["perimeter_definition_strategy"] == "metadata":
            # trial-wise, None
            result["perimeter"] = self.generate_label_to_object_field()
        if self.settings["ingress"]["perimeter_naming_strategy"] == "metadata":
            # metadata, None
            result["perimeter_name"] = detect_meters_per_pixel_in_perimeter_directory(self.perimeter_directory_path)
        if self.settings["ingress"]["center_definition_strategy"] == "metadata":
            # TODO: trial-wise
            # None
            result["center"] = detect_center_in_perimeter_directory(self.perimeter_directory_path)
        if self.settings["ingress"]["meters_per_pixel_definition_strategy"] == "metadata":
            # TODO: trial-wise
            # None
            result["meters_per_pixel"] = detect_meters_per_pixel_in_perimeter_directory(self.perimeter_directory_path)
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

        # TODO
        # for trial_class_name, dataset in self.settings["trial"]["specific"].items():
        #     if not dataset["defined"]:
        #         continue

        self._experiment_data_defined = True

    @validate_arguments
    def detect_perimeters_in_project(self, create_object: bool = False) -> list:
        detection_data = []
        for filename in self.perimeter_directory_path.glob("perimeter-*"):
            perimeter_path = self.project_root_directory / filename

            shape, label = get_perimeter_data(perimeter_path)
            data = {"label": label, "shape": shape}

            if create_object:
                data["perimeter"] = self.first_perimeter_set_from_makesense(perimeter_path)
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
            for perimeter_data in self.detect_perimeters_in_project(create_object=True)
        }

    @validate_arguments
    def first_perimeter_set_from_makesense(
        self,
        perimeter_path: FilePath,
        manual_shape: Optional[StringPerimeterShapes] = None,
    ) -> PerimeterSet:
        shape, label = get_perimeter_data(perimeter_path)
        match manual_shape or shape:
            case "circle":
                image_name_to_perimeter_set = CirclePerimeter.from_makesense_line(perimeter_path)
            case "rectangle":
                image_name_to_perimeter_set = PolygonPerimeter.from_makesense_csv_rectangle(perimeter_path)
            case "polygon" | "parallelogram":
                image_name_to_perimeter_set = PolygonPerimeter.from_makesense_coco_polygon(perimeter_path)
            case _:
                raise ValueError

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
        self._trial_id_to_keyword_arguments[trial_id].update(label_to_perimeter)

    def get_plugin_parameter(self, parameter_label: str, metadata_index_getter: Callable):
        parameter_indexes = PLUGIN_NAME_TO_KEYRING[parameter_label]
        return self.metadata_plugin_name_to_label_to_parameter[parameter_indexes["code_key"]][
            metadata_index_getter(parameter_indexes["human_readable_index"])
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

    # Client-side functions ===============================

    @cached_property
    def analysis_df(self) -> pd.DataFrame:
        np.seterr(all="ignore")
        return pd.concat(
            (self.metadata, self.experiment.combined_feature_motion_df),
            axis=1,
            keys=["Stage"] if self.stageful_metadata else None,
            # Prepend experiment stage to column MultiIndex:
            # https://stackoverflow.com/a/42094658/9793651
            names=self.experiment.column_multi_index_names,
        )

    def save_analysis_data(self):
        self.analysis_df.to_parquet(self.result_directory_path / f"animal_id_indexed_result_data.parquet")
        self.analysis_df.to_excel(self.result_directory_path / "animal_id_indexed_result_data.xlsx")


Ingress = TypeVar("Ingress", bound=BaseIngress)


def init_settings(
    project_root_directory: DirectoryPath,
    experiment_class: Any,
    method_kwargs: dict,
    kinematic_data_file_extension: str,
    method_immutable: Optional[dict] = None,
    dry_run: bool = False,
) -> dict[str, str | dict]:
    experiment_schema = extended_schema(experiment_class)
    experiment_schema["optional"]["data_reader_kwargs"] = extended_schema(DeepLabCutReader, with_required=False)[
        "optional"
    ]

    settings = {
        **method_kwargs,
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
            "fields": extended_schema(BasePerimeter),
        },
        "experiment": experiment_schema,
        "trial": extended_group_schema(experiment_class.trial_classes),
        "immutable": {
            "metadata_filename": "metadata.xlsx",
            "kinematic_data_file_extension": kinematic_data_file_extension,
            "experiment_class": experiment_class.__name__,
            "trial_classes/stages": experiment_class.trial_class_names,
            "method_specific": method_immutable,
        },
    }

    if dry_run:
        print(json.dumps(settings, indent=2))
    else:
        settings_path = get_project_settings_path(project_root_directory)
        with open(settings_path, "w") as out_file:
            yaml.safe_dump(settings, out_file, sort_keys=False)

    return settings


@validate_arguments
def auto_define_ingress_object(project_root_directory: DirectoryPath) -> Ingress:
    from bikipy.ingress import INGRESS_METHOD_NAME_TO_INGRESS_CLASS

    with open(project_root_directory / "settings.yaml", "r") as in_file:
        settings = yaml.safe_load(in_file)
    return INGRESS_METHOD_NAME_TO_INGRESS_CLASS[settings["ingress_method"]](
        project_root_directory=project_root_directory
    )


def analyze_and_save(project_root_directory: DirectoryPath):
    auto_define_ingress_object(project_root_directory).save_analysis_data()
