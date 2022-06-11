from abc import ABC, abstractmethod
from functools import cached_property, partial
from typing import Any, Callable, Optional

import pandas as pd
import yaml
from pydantic import DirectoryPath, FilePath, validate_arguments

from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS
from bikipy.core.base_class import BikipyBase
from bikipy.ingress.mapping import INGRESS_METHOD_NAME_TO_KEYWORD_ARGUMENT_FUNC
from bikipy.ingress.plugin import PLUGIN_NAME_TO_KEYRING
from bikipy.ingress.plugin.center import detect_center_in_perimeter_directory
from bikipy.ingress.plugin.meters_per_pixel import (
    detect_meters_per_pixel_in_perimeter_directory,
)
from bikipy.ingress.utils.perimeter import (
    first_perimeter_set_from_makesense,
    generate_label_to_object_field,
)
from bikipy.ingress.utils.pydantic import extended_schema
from bikipy.reader import DeepLabCutReader


class BaseIngress(BikipyBase, ABC):
    project_root_dir: DirectoryPath

    @property
    @abstractmethod
    def _experiment_class_kwargs_metadata_index_to_trial_id_getter(self):
        ...

    @property
    def experiment_class_name(self):
        return self.settings["immutable"]["experiment_class"]

    @property
    def experiment_class(self):
        try:
            return EXPERIMENT_NAME_TO_CLASS[self.experiment_class_name]
        except KeyError:
            msg = (
                f"experiment_class in settings is set to an invalid value: "
                f"{self.settings['immutable']['experiment_class']}; "
                f"this value should not be changed after initialization of the project."
            )
            raise ValueError(msg)

    @property
    def experiment_class_kwargs(self):
        return self._experiment_class_kwargs_metadata_index_to_trial_id_getter[0]

    @property
    def metadata_index_to_trial_id(self):
        return self._experiment_class_kwargs_metadata_index_to_trial_id_getter[1]

    @property
    def ingress_method(self):
        return self.settings["ingress_method"]

    @property
    def analysis_keyword_arguments_getter(self):
        try:
            return INGRESS_METHOD_NAME_TO_KEYWORD_ARGUMENT_FUNC[self.ingress_method]
        except KeyError:
            msg = f"ingress_method in settings is set to an invalid value: {self.settings['ingress_method']}."
            raise ValueError(msg)

    @cached_property
    def metadata_plugin_name_to_label_to_parameter(self) -> dict[str, dict]:
        result = {}
        if self.settings["ingress"]["perimeter_definition_strategy"] == "metadata":
            result["perimeter"] = generate_label_to_object_field(self.project_root_dir)
        if self.settings["ingress"]["center_definition_strategy"] == "metadata":
            result["center"] = detect_center_in_perimeter_directory(self.perimeter_directory_path)
        if self.settings["ingress"]["meters_per_pixel_definition_strategy"] == "metadata":
            result["meters_per_pixel"] = detect_meters_per_pixel_in_perimeter_directory(self.perimeter_directory_path)
        return result

    # I/O ============================

    @cached_property
    def settings(self) -> dict:
        with open(self.settings_path, "r") as in_file:
            return yaml.safe_load(in_file)

    @cached_property
    def metadata(self) -> pd.DataFrame:
        return pd.read_excel(
            next(self.project_root_dir.glob("metadata.*")),
            index_col=0,
            header=(0, 1) if self.stageful_metadata else 0,
        )

    @cached_property
    def metadata_for_analysis_data_frame(self):
        result = self.metadata.copy()
        if self.stageful_metadata:
            result = self.metadata.swaplevel(axis=1)

        # Add Location_Category level to the column multi-index. We need to this for pd.concat
        result.columns = pd.MultiIndex.from_product([result.columns, ["Location_Category"]])

        return result

    @property
    def settings_path(self) -> FilePath:
        return self.project_root_dir / "settings.yaml"

    @cached_property
    def dataset_directory_path(self) -> DirectoryPath:
        return self.project_root_dir / "dataset"

    @cached_property
    def perimeter_directory_path(self) -> DirectoryPath:
        return self.project_root_dir / "perimeter"

    @cached_property
    def result_directory_path(self) -> DirectoryPath:
        return self.project_root_dir / "result"

    # Constants =============================

    @cached_property
    def partial_first_perimeter_set_from_makesense_from_settings(self):
        return partial(
            first_perimeter_set_from_makesense,
            label_prefix=self.settings["perimeter"]["label_prefix"],
            label_suffix=self.settings["perimeter"]["label_suffix"],
        )

    @property
    def stageful_metadata(self):
        return self.settings["ingress"]["stageful_metadata"]

    @property
    def kinematic_data_file_extension(self):
        return self.settings["immutable"]["kinematic_data_file_extension"]

    def get_plugin_parameter(self, parameter_label: str, metadata_index_getter: Callable):
        parameter_indexes = PLUGIN_NAME_TO_KEYRING[parameter_label]
        return self.metadata_plugin_name_to_label_to_parameter[parameter_indexes["code_key"]][
            metadata_index_getter(parameter_indexes["human_readable_index"])
        ]

    @cached_property
    def experiment(self):
        return self.experiment_class(**self.settings["experiment"]["defined"], **self.experiment_class_kwargs)

    # Client-side functions ===============================

    @validate_arguments
    def analysis_df(self) -> pd.DataFrame:
        if not self.experiment.animal_id_indexed_feature_frame:
            msg = "Something went wrong with the analysis"
            raise RuntimeError(msg)

        return pd.concat(
            (self.metadata_for_analysis_data_frame, self.experiment.animal_id_indexed_feature_frame),
            axis=1,
            keys=["Stage"] if self.stageful_metadata else None,
            # Prepend experiment stage to column MultiIndex:
            # https://stackoverflow.com/a/42094658/9793651
            names=["Stage", "Feature", "Location_Category"]
            if self.stageful_metadata
            else ["Feature", "Location_Category"],
        )

    def save_analysis_data(self):
        self.analysis_df.to_parquet(self.result_directory_path / f"animal_id_indexed_result_data.parquet")
        self.analysis_df.to_excel(self.result_directory_path / "animal_id_indexed_result_data")


def init_settings(
    experiment_class: Any,
    method_kwargs: dict,
    kinematic_data_file_extension: str,
    method_immutable: Optional[dict] = None,
):
    experiment_schema = extended_schema(experiment_class.schema())
    experiment_schema["optional"]["data_import_kwargs"] = extended_schema(
        DeepLabCutReader.schema(), with_required=False
    )["optional"]
    return {
        **method_kwargs,
        "perimeter": {
            "label_prefix": None,
            "label_suffix": None,
        },
        "ingress": {
            "stageful_metadata": False,
            "meters_per_pixel_definition_strategy": "global_perimeter",
            "perimeter_definition_strategy": "metadata",
            "center_definition_strategy": None,
        },
        "experiment": experiment_schema,
        "immutable": {
            "metadata_filename": "metadata.xlsx",
            "kinematic_data_file_extension": kinematic_data_file_extension,
            "experiment_class": experiment_class.__name__,
            "trial_classes/stages": experiment_class.trial_class_names,
            "method_specific": method_immutable,
        },
    }
