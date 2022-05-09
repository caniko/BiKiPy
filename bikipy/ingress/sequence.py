"""
===================
The Sequence Method
===================
Designed for working with sequences of trials.

Rules
=====
- Delimiting is with a dash, "-". Example: 1-Training. Reminder to delimit -> (delimit!)
- The tracking data is segregated into trial-sets. A trial-set consists of a sequence of trial tracking files.
  The trial tracking file has the sequence index stored in as a prefix in the file-stem as a number (delimit!).
  Optionally, for improved readability you can store a sequence label followed by the stage index.
  Example: 0-Habituation.h5, 1-Training.h5, 2-Test.h5.
- Meter pixel ratio must be defined
    - Define it yourself, and plug it into sequence_generate_configuration()
    - If the experiment is in a confined box, or you know the length of a temporally fixed line in your video:
        1. Grab a video frame from one of the trial videos
        2. Define the line in MakeSense
        3. Make "Perimeter" directory in the base folder if it doesn't already exist.
        4. Export as csv and store in the "Perimeter" directory as "meter_pixel_ratio-{meter_length}.csv"; where
           meter_length is the length of the line in meters.

Structure
=========
- Each trial-set is stored in a directory prefixed with the animal ID (delimit!).
- Optional, trial-set metadata; yaml format. Stored inside trial-set directory. Fields in metadata:
    - Optional, perimeter_set. Example: perimeter_set: A
- Project metadata; .xlsx or .odt, xlsx has best support (apologies to FOSS):
  The table must be in the sheet that is on index 0! The metadata file is stored on the root/base folder.
    - Animal ID column name must be "Animal"
    - Optional, any number of generic columns that should be categorized for the animal
        - Gene
        - Cohort
        - Sex
        - Whatever...
    - Optional, store the usage of a perimeter "Perimeter_{label_of_perimeter}". Row must be empty if there is no
      perimeter. Row must define the label to apply to the perimeter.
    - Make sure your dataset has no junk/invisible characters that might lead to problems with recognizing tags and
      performing comparisons.
- Only MakeSense perimeters are supported. These are stored in the "Perimeter" directory/folder.
    - The file-stem is the perimeter set ID (PID).
      If a perimeter set is stored in several files you must also include a unique identifier (UID)
      after the perimeter set ID (delimit!).
      Opinion: The unique identifier could be a sequence of numbers, letters, or random.
      Example: A-1.csv; where A is the perimeter set ID and 1 is the unique identifier.
    - Make sure that you don't use the same label for the different perimeters when they are defined in MakeSense.
      You can change the label in the file if you have to ensure this later.
    - Optionally, for inspection, you can include an image with the perimeter set as the file-stem.
      Optionally, include the uid if it is specific to the subset (delimit!).
"""
import json
import pickle
from logging import getLogger
from typing import Optional

import pandas as pd
import yaml
from pydantic import DirectoryPath, validate_arguments

from bikipy.behaviour.mapping import NAME_TO_CLASS
from bikipy.ingress.core import init_settings
from bikipy.ingress.utils.constant import get_project_settings_path, load_settings
from bikipy.ingress.utils.meter_pixel_ratio import get_meter_pixel_ratio

logger = getLogger(__name__)


@validate_arguments
def sequence_generate_configuration(
    root_directory: DirectoryPath,
    experiment_name: str,
    meter_pixel_ratio: Optional[float] = None,
    kinematic_data_file_extension: str = ".h5",
    animals_have_plural_trial_sets: bool = False,
    dry_run: bool = False,
) -> dict:
    experiment_class = NAME_TO_CLASS[experiment_name.strip().lower()]

    meter_pixel_ratio = meter_pixel_ratio or get_meter_pixel_ratio(root_directory)

    logger.info(f"Generating experiment configuration at {root_directory}")

    animal_ids = set()
    trial_set_stage_ids = []
    for trial_set_dir in root_directory.iterdir():
        if trial_set_dir.is_file() or trial_set_dir.name == "Perimeter":
            continue
        animal_id = trial_set_dir.name.split("-")[0]
        if animals_have_plural_trial_sets and animal_id in animal_ids:
            msg = (
                f"Animal ID {animal_id} is repeated across trial sets. "
                f"Set animals_have_plural_trial_sets to true if this behaviour is expected"
            )
            raise ValueError(msg)
        animal_ids.add(animal_id)

        stage_ids = set()
        for filename in trial_set_dir.glob(f"*{kinematic_data_file_extension}"):
            stage_ids.add(int(filename.stem.split("-")[0]))
        trial_set_stage_ids.append(stage_ids)

    if any(trial_set_stage_ids[0] != stage_ids for stage_ids in trial_set_stage_ids[1:]):
        msg = f"The trial sets do not have identical trial stage sequence:\n{trial_set_stage_ids}"
        raise ValueError(msg)

    stages = trial_set_stage_ids.pop()
    # if experiment_class.stage_index_to_trial_class and max(stages) != len(
    #     experiment_class.stage_index_to_trial_class
    # ):
    #     msg = (
    #         f"The trials stage length are incorrect, {number_of_stages}."
    #         f"experiment_class.stage_index_to_trial_class:\n{experiment_class.stage_index_to_trial_class}"
    #     )
    #     raise ValueError(msg)

    with open(root_directory / "metadata.xlsx", "rb") as in_file:
        metadata_df = pd.read_excel(in_file)

    try:
        metadata_animal_id_column_set = set(metadata_df.loc[:, "Animal"])
    except KeyError:
        msg = f"Animal ID column, Animal, is not defined in the metadata sheet. Defined columns:\n{metadata_df.columns}"
        raise KeyError(msg)

    if metadata_animal_id_column_set != animal_ids:
        msg = (
            "The animal ID sets in the metadata and the trial_set directory names do not match:\n"
            f"- metadata: {metadata_animal_id_column_set}\n- trial_sets: {animal_ids}"
        )
        raise ValueError(msg)

    settings = init_settings(
        experiment_class,
        root_directory,
        meter_pixel_ratio,
        kinematic_data_file_extension,
        animal_ids,
        animals_have_plural_trial_sets,
        dry_run,
    )

    if dry_run:
        print(json.dumps(settings, indent=2))
    else:
        settings_path = get_project_settings_path(root_directory)
        with open(settings_path, "w") as out_file:
            yaml.dump(settings, out_file, sort_keys=False)

    return settings


@validate_arguments
def analyse_sequence(root_directory: DirectoryPath):
    settings = load_settings(root_directory)
    experiment_class = NAME_TO_CLASS[settings["immutable"]["experiment_class"]]
    metadata = pd.read_excel(root_directory / settings["immutable"]["metadata_filename"])
    metadata.set_index("Animal", inplace=True)

    with open(root_directory / "Perimeter" / settings["perimeter"]["perimeter_pickle_file"], "rb") as in_file:
        perimeters = pickle.load(in_file)

    trial_id_vs_keyword_arguments = {}
    for i, trial_data_path in enumerate(root_directory.glob(f"**/*{settings['immutable']['kinematic_data_file_extension']}")):
        animal_id = trial_data_path.parent.name
        animal_metadata = dict(metadata.loc[animal_id, :])
        trial_id_vs_keyword_arguments[i] = {
            "animal_id": trial_data_path.parent.name
        }
