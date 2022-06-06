from typing import Any

from pydantic import DirectoryPath

from bikipy.ingress.utils.perimeter import load_settings, detect_perimeters_in_project
from bikipy.ingress.utils.pydantic import extended_schema
from bikipy.reader import DeepLabCutReader


def analyse(root_directory: DirectoryPath, *args, **kwargs) -> None:
    settings = load_settings(root_directory)
    match settings["ingress_method"]:
        case "sequence":
            from bikipy.ingress.sequence import analyse_sequence

            ingress_method = analyse_sequence
        case _:
            msg = f"ingress_method in settings, is set to an invalid value: {settings['ingress_method']}"
            raise ValueError(msg)
    return ingress_method(root_directory, *args, **kwargs)


def init_settings(
    experiment_class: Any,
    method_kwargs: dict,
    root_directory: DirectoryPath,
    meter_pixel_ratio: float,
    kinematic_data_file_extension: str,
    animal_ids: set[str],
    animals_have_plural_trial_sets: bool,
):
    experiment_schema = extended_schema(experiment_class.schema())
    experiment_schema["optional"]["data_import_kwargs"] = extended_schema(
        DeepLabCutReader.schema(), with_required=False
    )["optional"]
    return {
        "ingress_method": "sequence",
        **method_kwargs,
        "meter_pixel_ratio": meter_pixel_ratio,
        "perimeters": detect_perimeters_in_project(root_directory),
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
