import pickle
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from bikipy.behaviour.nort.trial import NortTrial
from bikipy.utils.video import get_video_data

WORKING_DIR = Path("C:/Users/Can/Projects/Neuroscience/Imen/data")
NORT_DIR = WORKING_DIR / "nort"

HABIT_DIR = NORT_DIR / "Habituation"
H_BEFORE_DIR = HABIT_DIR / "0_before"
H_AFTER_DIR = HABIT_DIR / "1_after"

NOVELTY_DIR = NORT_DIR / "Novelty"
N_1A_DIR = NOVELTY_DIR / "NORT_02.06.2020 (1A)"
N_2A_DIR = NOVELTY_DIR / "NORT (after)_24.08.2020 (2A)"
N_1B_DIR = NOVELTY_DIR / "NORT2_30.08.2020 (1B)"
N_2B_DIR = NOVELTY_DIR / "NORT2 (after)_23.11.2020 (2B)"

OPEN_FIELD_DIR = NORT_DIR / "Open-Field"
N_1C_DIR = OPEN_FIELD_DIR / "NORT_02.06.2020 (1C)"
N_2C_DIR = OPEN_FIELD_DIR / "NORT (after)_24.08.2020 (2C)"
N_1D_DIR = OPEN_FIELD_DIR / "NORT2_30.08.2020 (1D)"
N_2D_DIR = OPEN_FIELD_DIR / "NORT2 (after)_23.11.2020 (2D)"

ANNOTATIONS_PATH = str(WORKING_DIR / "python_data" / "annotations" / "all.pickle")
BEFORE_ANNOTATIONS_PATH = str(
    WORKING_DIR / "python_data" / "annotations" / "before.pickle"
)
AFTER_ANNOTATIONS_PATH = str(
    WORKING_DIR / "python_data" / "annotations" / "after.pickle"
)
IMPORTED_DLC_FILES = str(WORKING_DIR / "python_data" / "dlc.pickle")
RESULTS = str(WORKING_DIR / "python_data" / "results.pickle")

EXP_ID_FINDER = re.compile("\d+")
BORDER_DISTANCE = 3 * 224 / 40

with open(ANNOTATIONS_PATH, "rb") as infile:
    (
        a1_1,
        a1_2,
        a1_3,
        a1_4,
        a2_1,
        a2_2,
        a2_3,
        a2_4,
        b1_1,
        b1_2,
        b1_3,
        b1_4,
        b2_1,
        b2_2,
        b2_3,
        b2_4,
    ) = pickle.load(infile)


app_to_obj = {
    "before": {
        1: {1: b1_1, 2: b1_2, 3: b1_3, 4: b1_4},
        2: {1: b2_1, 2: b2_2, 3: b2_3, 4: b2_4},
    },
    "after": {
        1: {1: a1_1, 2: a1_2, 3: a1_3, 4: a1_4},
        2: {1: a2_1, 2: a2_2, 3: a2_3, 4: a2_4},
    },
}


exp_info_df_0 = pd.read_excel(str(WORKING_DIR / "NORT_Round1.ods"), sheet_name=0)
exp_info_df_1 = pd.read_excel(str(WORKING_DIR / "NORT_Round1.ods"), sheet_name=1)


def get_animal_id_vs_exp_ids(info_df):
    exp_ids = np.array(
        [int(EXP_ID_FINDER.findall(info)[-1]) for info in info_df["Video_file_name"]]
    )
    animal_id = np.array(info_df["Animal"])

    result = {}
    for i in range(int(animal_id.min()), int(animal_id.max() + 1)):
        loc = np.where(animal_id == i)[0][:2]
        result[i] = tuple(exp_ids[loc])

    return result


def get_exp_id_vs_animal_id(id_exp):
    result = {}
    for animal, exps in id_exp.items():
        for exp in exps:
            result[exp] = animal
    return result


exp_animal = {
    "before": get_exp_id_vs_animal_id(get_animal_id_vs_exp_ids(exp_info_df_0)),
    "after": get_exp_id_vs_animal_id(get_animal_id_vs_exp_ids(exp_info_df_1)),
}


def get_animal_id_vs_apparatus(info_df):
    animal_id = np.unique(info_df["Animal"])
    apparatus = info_df["Apparatus"]

    return {
        int(i): int(EXP_ID_FINDER.findall(app)[0])
        for i, app in zip(animal_id, apparatus)
    }


id_app = {
    "before": get_animal_id_vs_apparatus(exp_info_df_0),
    "after": get_animal_id_vs_apparatus(exp_info_df_1),
}


def get_exp_id_vs_stage(exp_info_df):
    result = {}
    for row in exp_info_df[["Video_file_name", "Stage"]].iterrows():
        exp_idx = int(EXP_ID_FINDER.findall(Path(row[1][0]).stem)[0])
        stage = Path(row[1][1]).stem[-1:]
        result[exp_idx] = stage

    return result


exp_id_vs_stage = {
    "before": get_exp_id_vs_stage(exp_info_df_0),
    "after": get_exp_id_vs_stage(exp_info_df_1),
}

exp_ids_range_vs_exp_meta = {"before": {}, "after": {}}
exp_id_vs_coordinate_data_path = {"before": {}, "after": {}}
for time, paths in zip(
    exp_ids_range_vs_exp_meta,
    ((H_BEFORE_DIR, N_1A_DIR), (H_AFTER_DIR, N_2A_DIR)),
):
    for exp_category, root in zip(("habituation", "novelty_observation"), paths):
        for data_path in glob(str(root / "*.h5")):
            exp_id = int(EXP_ID_FINDER.findall(Path(data_path).stem)[0])

            exp_id_vs_coordinate_data_path[time][exp_id] = data_path

        for data_path in glob(str(root / "*.mp4")):
            exp_id = int(EXP_ID_FINDER.findall(Path(data_path).stem)[0])

            _, x, y, fps = get_video_data(data_path)

            exp_ids_range_vs_exp_meta[time][exp_id] = {
                "exp_category": exp_category,
                "recording_resolution": (x, y),
                "fps": fps,
            }

            if exp_category == "novelty_observation":
                animal_id = exp_animal[time][exp_id]
                apparatus = id_app[time][animal_id]

                stage = int(exp_id_vs_stage[time][exp_id])
                novelty_objs = app_to_obj[time][stage][apparatus]

                exp_ids_range_vs_exp_meta[time][exp_id] = {
                    **exp_ids_range_vs_exp_meta[time][exp_id],
                    **novelty_objs,
                }


result = {}
for time in exp_ids_range_vs_exp_meta:
    result[time] = NortTrial(
        exp_ids_range_vs_exp_meta=exp_ids_range_vs_exp_meta[time],
        nose_label="nose",
        eye_center_label="mid-left_ear-right_ear",
        torso_label="mid-mid-left_ear-right_ear-tail",
        experiment_box_real_length=40,
        center_size_real_length=20,
        max_radians_gaze_and_object=1 / 4 * np.pi,
        exp_id_vs_coordinate_data_path=exp_id_vs_coordinate_data_path[time],
        midpoint_groups=[("left_ear", "right_ear"), ("mid-left_ear-right_ear", "tail")],
    )

with pd.ExcelWriter(
    "C:/Users/Can/Projects/Neuroscience/bikipy/examples/data/results/nort.ods"
) as writer:
    for time, trial in result.items():
        trial.export_to_dataframe().to_excel(writer, sheet_name=time)
