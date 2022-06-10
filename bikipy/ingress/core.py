from functools import cached_property
from typing import Any, Optional

import pandas as pd
import yaml
from pydantic import DirectoryPath, validate_arguments, FilePath

from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS
from bikipy.core.base_class import BikipyBase
from bikipy.ingress.mapping import INGRESS_METHOD_NAME_TO_KEYWORD_ARGUMENT_FUNC
from bikipy.ingress.plugin.center import detect_center_in_perimeter_directory
from bikipy.ingress.plugin.meters_per_pixel import detect_meters_per_pixel_in_perimeter_directory
from bikipy.ingress.utils.perimeter import generate_label_to_object_field
from bikipy.ingress.utils.pydantic import extended_schema
from bikipy.reader import DeepLabCutReader


class Ingress(BikipyBase):
    project_root_dir: DirectoryPath

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
            result["center"] = detect_center_in_perimeter_directory(self.perimeter_dir_path)
        if self.settings["ingress"]["meters_per_pixel_definition_strategy"] == "metadata":
            result["meters_per_pixel"] = detect_meters_per_pixel_in_perimeter_directory(
                self.perimeter_dir_path
            )
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

    @property
    def settings_path(self) -> FilePath:
        return self.project_root_dir / "settings.yaml"

    @cached_property
    def dataset_dir_path(self) -> DirectoryPath:
        return self.project_root_dir / "dataset"

    @cached_property
    def perimeter_dir_path(self) -> DirectoryPath:
        return self.project_root_dir / "perimeter"

    # Constants =============================

    @property
    def stageful_metadata(self):
        return self.settings["ingress"]["stageful_metadata"]


@validate_arguments
def analyze(project_root_dir: DirectoryPath) -> None:
    ingress = Ingress(project_root_dir=project_root_dir)
    experiment_class_kwargs, metadata_index_to_trial_id = analysis_keyword_arguments_getter(
        project_root_dir, metadata_plugin_name_to_label_to_parameter
    )

    experiment = experiment_class(**settings["experiment"]["defined"], **experiment_class_kwargs)
    if not experiment.animal_id_indexed_feature_frame:
        msg = "Something went wrong with the analysis"
        raise RuntimeError(msg)

    if stageful := settings["stageful_metadata"]:
        metadata = metadata.swaplevel(axis=1)

    # Add Location_Category level to the column multi-index. We need to this for pd.concat
    metadata.columns = pd.MultiIndex.from_product([metadata.columns, ["Location_Category"]])

    result_data_frame = pd.concat(
        (metadata, experiment.animal_id_indexed_feature_frame),
        axis=1,
        keys=["Stage"] if stageful else None,
        # Prepend experiment stage to column MultiIndex:
        # https://stackoverflow.com/a/42094658/9793651
        names=["Stage", "Feature", "Location_Category"] if stageful else ["Feature", "Location_Category"],
    )

    result_dir = project_root_dir / "result"
    result_data_frame.to_parquet(result_dir / f"animal_id_indexed_result_data.parquet")
    result_data_frame.to_excel(result_dir / "animal_id_indexed_result_data")


def init_settings(
    experiment_class: Any,
    method_kwargs: dict,
    project_root_dir: DirectoryPath,
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
