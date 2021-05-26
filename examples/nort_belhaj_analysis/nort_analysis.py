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
    get_animal_id_vs_exp_ids,
    get_exp_id_vs_animal_id,
    get_exp_id_vs_stage,
    round_vs_apparatus_to_general_nort_fields,
)
from bikipy.utils.video import get_video_data


DEEPLABCUT_DIR = Path("/mnt/md0/Projects/Neuroscience/Imen/data/nort")
ROUND_DIR_NAME = "Round_1"

DLC_HABITUATION_DIR = DEEPLABCUT_DIR / "Open-Field" / ROUND_DIR_NAME
DLC_NOVELTY_DIR = DEEPLABCUT_DIR / "Novelty" / ROUND_DIR_NAME

WORKING_DIR = Path(".").resolve()
DATA_DIR = WORKING_DIR / "data"

EXP_ID_REGEX_PATTERN = re.compile("\d+")


# Set sheet name to 0 for Round_1 and 1 for Round_2
EXP_INFO = pd.read_excel(
    str(DATA_DIR / "nort_round_1.xlsx"), sheet_name=0, engine="openpyxl"
)

with open(DATA_DIR / "area_images" / "A_annotations.pickle", "rb") as infile:
    round_vs_field_apparatus = pickle.load(infile)
round_keys = [f"round_{num}" for num in range(len(round_vs_field_apparatus))]
round_vs_field_vs_apparatus = {
    rem_round: round_vs_apparatus_to_general_nort_fields(field_apparatus)
    for rem_round, field_apparatus in zip(round_keys, round_vs_field_apparatus.values())
}
field_vs_apparatus = round_vs_field_vs_apparatus[
    f"round_{1 if '1' in ROUND_DIR_NAME else 2}"
]

animal_id_vs_app = get_animal_id_vs_apparatus(EXP_INFO, EXP_ID_REGEX_PATTERN)
exp_id_vs_stage = get_exp_id_vs_stage(EXP_INFO, EXP_ID_REGEX_PATTERN)
exp_vs_animal = get_exp_id_vs_animal_id(get_animal_id_vs_exp_ids(EXP_INFO))

exp_id_range_vs_exp_meta = {}
exp_id_vs_coordinate_data_path = {}
for exp_class, root in zip(
    ("novelty", "habituation"), (DLC_NOVELTY_DIR, DLC_HABITUATION_DIR)
):
    for exp_dir in os.listdir(root):
        exp_designation = exp_dir.split("_")[0]
        exp_path = root / exp_dir
        glob_exp_data_path = exp_path / "**" if exp_class == "novelty" else exp_path

        exp_id_vs_coordinate_data_path[exp_designation] = {}
        for data_path in glob(str(glob_exp_data_path / "*.parquet")):
            exp_id = int(EXP_ID_REGEX_PATTERN.findall(Path(data_path).stem)[0])

            exp_id_vs_coordinate_data_path[exp_designation][exp_id] = data_path

        exp_id_range_vs_exp_meta[exp_designation] = {}
        for data_path in glob(str(glob_exp_data_path / "*.mp4")):
            exp_id = int(EXP_ID_REGEX_PATTERN.findall(Path(data_path).stem)[0])

            exp_id_range_vs_exp_meta[exp_designation][exp_id] = {
                "stage": exp_class,
                "recording_resolution": (video_data := get_video_data(data_path))[1:3],
                "fps": video_data[3],
                "animal_id": (animal_id := exp_vs_animal[exp_id]),
                "field": animal_id_vs_app[animal_id],
            }

with pd.ExcelWriter(
    WORKING_DIR / "nort_analysis.ods", strings_to_formulas=False, strings_to_urls=False
) as writer:
    for exp_designation in exp_id_range_vs_exp_meta:
        NortExperiment(
            exp_id_range_vs_exp_meta=exp_id_range_vs_exp_meta[exp_designation],
            experiment_box_real_length=40,
            nose_label="nose",
            eye_center_label="mid-left_ear-right_ear",
            torso_label="mid-mid-left_ear-right_ear-tail",
            nort_field_vs_apparatus=field_vs_apparatus,
            center_size_real_length=20,
            max_radians_gaze_and_object=0.33 * np.pi,
            exp_id_vs_coordinate_data_path=exp_id_vs_coordinate_data_path[
                exp_designation
            ],
            # func_inspect=True,
            init_from="parquet",
            midpoint_groups=[
                ("left_ear", "right_ear"),
                ("mid-left_ear-right_ear", "tail"),
            ],
        ).df.to_excel(writer, sheet_name=exp_designation)
