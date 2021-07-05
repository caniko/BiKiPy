import os
import pickle
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from bikipy.behaviour.nort.experiment import NortExperiment
from bikipy.plugins.belhaj import (
    get_animal_id_vs_apparatus,
    get_animal_id_vs_trial_ids,
    get_trial_id_vs_animal_id,
    get_trial_id_vs_stage,
)

DEEPLABCUT_DIR = Path("/mnt/md0/Projects/Neuroscience/Imen/data/nort")

EXPERIMENT_DIR = DEEPLABCUT_DIR / "Experiment_2"

WORKING_DIR = Path(".").resolve()
DATA_DIR = WORKING_DIR / "data"
IMAGE_DIR = DATA_DIR / "area_images"

# PICKLE_PATHS = (
#     IMAGE_DIR / "A1" / "a1_labels_repickled.pickle",
#     IMAGE_DIR / "A2" / "a2_labels_repickled.pickle",
# )
# META_DATA = DATA_DIR / "nort_round_1.xlsx"

PICKLE_PATHS = (
    IMAGE_DIR / "B1" / "b1_labels_repickled.pickle",
    IMAGE_DIR / "B2" / "b2_labels_repickled.pickle",
)
META_DATA = DATA_DIR / "nort_round_2.xlsx"

EXP_ID_REGEX_PATTERN = re.compile(r"\d+")

with pd.ExcelWriter(
    WORKING_DIR / "nort_analysis.ods",
    engine_kwargs={
        "strings_to_formulas": False,
        "strings_to_urls": False,
    }
) as writer:
    for round_number, (experiment_dir, pickle_path) in enumerate(
        zip(os.listdir(EXPERIMENT_DIR), PICKLE_PATHS)
    ):
        round_designation = f"round_{round_number + 1}"
        experiment_dir = EXPERIMENT_DIR / experiment_dir
        date = experiment_dir.name.split("_")[1]

        with open(pickle_path, "rb") as infile:
            nort_field_vs_nort_field_object = pickle.load(infile)

        exp_metadata_df = pd.read_excel(
            META_DATA, sheet_name=round_number, engine="openpyxl"
        )

        animal_id_vs_app = get_animal_id_vs_apparatus(
            exp_metadata_df, EXP_ID_REGEX_PATTERN
        )
        trial_id_vs_stage = get_trial_id_vs_stage(exp_metadata_df, EXP_ID_REGEX_PATTERN)
        animal_id_vs_trial_ids = get_animal_id_vs_trial_ids(exp_metadata_df)
        exp_vs_animal = get_trial_id_vs_animal_id(animal_id_vs_trial_ids)

        trial_id_vs_paths = {}
        for video_path in glob(str(experiment_dir / "**" / "*.mp4")):
            trial_id = int(EXP_ID_REGEX_PATTERN.findall(Path(video_path).stem)[0])
            trial_id_vs_paths[trial_id] = {"video": video_path}
        for data_path in glob(str(experiment_dir / "**" / "*.parquet")):
            trial_id = int(EXP_ID_REGEX_PATTERN.findall(Path(data_path).stem)[0])
            trial_id_vs_paths[trial_id]["data"] = data_path

        trial_id_range_vs_exp_meta = {}
        for trial_id, paths in trial_id_vs_paths.items():
            trial_data = {
                "coordinate_data_path": paths["data"],
                "stage": (stage := trial_id_vs_stage[trial_id]),
                "video_path": paths["video"],
                "animal_id": (animal_id := exp_vs_animal[trial_id]),
                "inspect": False,
            }

            if stage != "habituation":
                trial_data["field"] = animal_id_vs_app[animal_id]

            trial_id_range_vs_exp_meta[trial_id] = trial_data

        NortExperiment(
            trial_id_vs_data=trial_id_range_vs_exp_meta,
            metric_resolution=0.4,
            nose_label="nose",
            eye_center_label="mid-left_ear-right_ear",
            torso_label="mid-mid-left_ear-right_ear-tail",
            nort_field_vs_nort_field_object=nort_field_vs_nort_field_object,
            perimeter_border_normal_metric_magnitude=0.06,
            center_metric_length=0.2,
            maximum_radians_inter_gaze_perimeter=0.25 * np.pi,
            # func_inspect=True,
            init_from="parquet",
            midpoint_groups=(
                ("left_ear", "right_ear"),
                ("mid-left_ear-right_ear", "tail"),
            ),
        ).df.to_excel(writer, sheet_name=date)
