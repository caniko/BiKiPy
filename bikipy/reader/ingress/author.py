"""
Designed for working with sequences of trials.

All delimiting is with a dash, "-"

The author method:
    - Each trial set has its own directory, the name of the directory must be prefixed with the animal ID (delimit!).
    - Dataset of each component of the trial has the stage index as prefix, stage indexing starts from 0 (delimit!).
      Optionally, for improved readability one can have the stage index followed by the stage label; ex: 0-Habituation.
    - The metadata must be either .xlsx or .odt (xlsx has best support, sorry FOSS), the metadata must be in sheet 0!
        - Animal ID column name must be "Animal"
        - Genetic state column must have the name "Gene"
        - Optional, "Cohort"
        - Optional, "Sex"
        - Make sure your dataset has no junk characters that might lead to problems with string comparisons
"""
import os
from glob import iglob
from logging import getLogger
from pathlib import Path
from typing import Optional, Any, Union

import pandas as pd
import yaml
from pydantic import validate_arguments, DirectoryPath

from bikipy.behaviour.base import BaseExperiment
from bikipy.reader.ingress.cm_pixel_ratio import CentimeterPixelRatio

logger = getLogger(__name__)


@validate_arguments
def author_generate_configuration(
    root_directory: DirectoryPath,
    experiment_class: BaseExperiment,
    cm_pixel_ratio_kwargs: Union[float, dict[str, Any]],
    kinematic_data_file_extension: str = "h5",
    metadata_filename: str = "metadata.xlsx",
    animals_have_several_trial_sets: bool = False,
) -> None:
    cm_pixel_ratio = (
        CentimeterPixelRatio(**cm_pixel_ratio_kwargs)
        if isinstance(cm_pixel_ratio_kwargs, dict)
        else cm_pixel_ratio_kwargs
    )

    logger.info(f"Generating experiment configuration at {root_directory}")

    animal_ids, trial_set_stage_ids = set(), set()
    for trial_set_dir in os.listdir(root_directory):
        animal_id = trial_set_dir.split("-")[0]
        if animals_have_several_trial_sets and animal_id in animal_ids:
            msg = (
                f"Animal ID {animal_id} is repeated across trial sets. "
                f"Set animals_have_several_trial_sets to true if this behaviour is expected"
            )
            raise ValueError(msg)
        animal_ids.add(animal_id)

        stage_ids = set()
        for filename in iglob(str(Path(trial_set_dir) / f"*.{kinematic_data_file_extension}")):
            stage_ids.add(filename.split("-")[0])
        trial_set_stage_ids.add(stage_ids)

    if len(trial_set_stage_ids) != 1:
        msg = f"The trial sets do not have identical trial stage sequence:\n{trial_set_stage_ids}"
        raise ValueError(msg)

    stages = trial_set_stage_ids.pop()
    number_of_stages = len(stages)
    if experiment_class.stage_index_to_trial_class and number_of_stages != len(
        experiment_class.stage_index_to_trial_class
    ):
        msg = (
            f"The trials stage length are incorrect, {number_of_stages}."
            f"experiment_class.stage_index_to_trial_class:\n{experiment_class.stage_index_to_trial_class}"
        )
        raise ValueError(msg)

    with open(metadata_filename, "rb") as in_file:
        metadata_df = pd.read_excel(in_file)

    try:
        metadata_animal_id_column_set = set(metadata_df.loc["Animal"])
    except KeyError:
        msg = f"Animal ID column, Animal, is not defined in the metadata sheet. Defined columns:\n{metadata_df.columns}"
        raise KeyError(msg)

    if metadata_animal_id_column_set != animal_ids:
        msg = (
            "The animal ID sets in the metadata and the trial_set directory names do not match:\n"
            f"- metadata: {metadata_animal_id_column_set}\n- trial_sets: {animal_ids}"
        )
        raise ValueError(msg)

    with open("settings.yaml", "w") as out_file:
        yaml.dump(
            {
                "metadata_filename": metadata_filename,
                "immutable": {
                    "cm_pixel_ratio": cm_pixel_ratio,
                    "kinematic_data_file_extension": kinematic_data_file_extension,
                    "experiment_class": experiment_class.__name__,
                    "Total # animals": len(animal_ids),
                    "animals_have_several_trial_sets": animals_have_several_trial_sets,
                },
            },
            out_file,
        )


@validate_arguments
def author_ingress_method(root_directory: DirectoryPath):
    pass
