from typing import Any

import pandas as pd
from pydantic import DirectoryPath, validate_arguments

from bikipy.behaviour.mapping import EXPERIMENT_NAME_TO_CLASS
from bikipy.ingress.mapping import INGRESS_METHOD_NAME_TO_KEYWORD_ARGUMENT_FUNC
from bikipy.ingress.utils.io import initialize_metadata_data_frame
from bikipy.ingress.utils.perimeter import load_settings, detect_perimeters_in_project
from bikipy.ingress.utils.pydantic import extended_schema
from bikipy.reader import DeepLabCutReader


@validate_arguments
def analyze(root_directory: DirectoryPath) -> None:
    settings = load_settings(root_directory)
    metadata = initialize_metadata_data_frame(root_directory, settings["ingress"]["stageful_metadata"])

    try:
        experiment_class = EXPERIMENT_NAME_TO_CLASS[settings["immutable"]["experiment_class"]]
    except KeyError:
        msg = (
            f"experiment_class in settings is set to an invalid value: {settings['immutable']['experiment_class']}; "
            f"this value should not be changed after initialization of the project."
        )
        raise ValueError(msg)

    try:
        analysis_keyword_arguments_getter = INGRESS_METHOD_NAME_TO_KEYWORD_ARGUMENT_FUNC["sequence"]
    except KeyError:
        msg = f"ingress_method in settings is set to an invalid value: {settings['ingress_method']}."
        raise ValueError(msg)

    experiment = experiment_class(
        **settings["experiment"]["defined"], **analysis_keyword_arguments_getter(root_directory)
    )
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

    result_dir = root_directory / "result"
    result_data_frame.to_parquet(result_dir / f"animal_id_indexed_result_data.parquet")
    result_data_frame.to_excel(result_dir / "animal_id_indexed_result_data")


def init_settings(
    experiment_class: Any,
    method_kwargs: dict,
    root_directory: DirectoryPath,
    kinematic_data_file_extension: str,
    animal_ids: set[str],
    animals_have_plural_trial_sets: bool,
):
    experiment_schema = extended_schema(experiment_class.schema())
    experiment_schema["optional"]["data_import_kwargs"] = extended_schema(
        DeepLabCutReader.schema(), with_required=False
    )["optional"]
    return {
        **method_kwargs,
        "meter_pixel_ratio": "global_perimeter",
        "perimeter": {
            # metadata, trialwise, None
            "perimeter_definition_strategy": "metadata",
        },
        "ingress": {
            "stageful_metadata": False,
            "center_definition_strategy": "metadata",
        },
        "experiment": experiment_schema,
        "immutable": {
            "metadata_filename": "metadata.xlsx",
            "kinematic_data_file_extension": kinematic_data_file_extension,
            "experiment_class": experiment_class.__name__,
            "trial_classes/stages": experiment_class.trial_class_names,
            "Number of animals": len(animal_ids),
            "animals_have_plural_trial_sets": animals_have_plural_trial_sets,
        },
    }
