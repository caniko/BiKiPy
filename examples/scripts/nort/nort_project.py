from pathlib import Path
from glob import glob
import pickle
import re

import pandas as pd
import numpy as np

from bikipy.border.base import GenericPolygonalBorder
from bikipy.readers.deeplabcut import DeepLabCutReader
from bikipy.behaviour.nort_observation import nort_observation


WORKING_DIR = Path("C:/Users/Can/Projects/Neuroscience/Imen/data")
NORT_DIR = WORKING_DIR / "nort_observation"
BEFORE_DIR = NORT_DIR / "0_before_02.06.2020"
AFTER_DIR = NORT_DIR / "1_after_24.08.2020"

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


def get_animal_id_vs_exp_ids(info_df):
    exp_ids = np.array(
        [int(EXP_ID_FINDER.findall(info)[1]) for info in info_df["Video_file_name"]]
    )
    animal_id = np.array(info_df["Animal"])
    return {
        i: tuple(exp_ids[np.where(animal_id == i)[0][:2]])
        for i in range(int(animal_id.min()), int(animal_id.max()))
    }


def get_animal_id_vs_apparatus(info_df):
    animal_id = np.unique(info_df["Animal"])
    apparatus = info_df["Apparatus"]

    return {i: EXP_ID_FINDER.findall(app)[0] for i, app in zip(animal_id, apparatus)}


def dlc_objectifier(dir_path):
    exp_id_vs_dlc = {}
    for h_file in glob(str(dir_path / "*.h5")):
        exp_id = int(EXP_ID_FINDER.findall(Path(h_file).stem)[0])
        exp_id_vs_dlc[exp_id] = DeepLabCutReader.from_video(
            str(dir_path / f"Test {exp_id}.mp4"),
            hdf_path=h_file,
            midpoint_groups=[
                ["left_ear", "right_ear"],
                ["mid-left_ear-right_ear", "tail"],
            ],
        )
    return exp_id_vs_dlc


with open(IMPORTED_DLC_FILES, "rb") as infile:
    id_dlc_0, id_dlc_1 = pickle.load(infile)


def border_analysis(id_exp, app_to_obj_t1, app_to_obj_t2, id_app):
    result = {}
    for animal, exp_ids in id_exp.items():
        try:
            app_a_t1 = app_to_obj_t1[int(id_app[animal])]["A"]
            app_b_t1 = app_to_obj_t1[int(id_app[animal])]["B"]
            app_a_t2 = app_to_obj_t2[int(id_app[animal])]["A"]
            app_b_t2 = app_to_obj_t2[int(id_app[animal])]["B"]

        except KeyError:
            print(f"Skipping {animal} because it has no data")
            continue

        t1 = id_dlc_0[exp_ids[0]]
        t2 = id_dlc_0[exp_ids[1]]

        t1_a_at = nort_observation(
            app_a_t1,
            t1["mid-left_ear-right_ear"],
            t1["nose"],
            t1["mid-mid-left_ear-right_ear-tail"],
            BORDER_DISTANCE,
            14.99,
        )
        t1_B_at = nort_observation(
            app_b_t1,
            t1["mid-left_ear-right_ear"],
            t1["nose"],
            t1["mid-mid-left_ear-right_ear-tail"],
            14.99,
        )
        t2_a_at = nort_observation(
            app_a_t2,
            t2["mid-left_ear-right_ear"],
            t2["nose"],
            t2["mid-mid-left_ear-right_ear-tail"],
            14.99,
        )
        t2_B_at = nort_observation(
            app_b_t2,
            t2["mid-left_ear-right_ear"],
            t2["nose"],
            t2["mid-mid-left_ear-right_ear-tail"],
            14.99,
        )

        result[animal] = {
            "t1_a": t1_a_at,
            "t1_b": t1_B_at,
            "t2_a": t2_a_at,
            "t2_b": t2_B_at,
        }

    return result


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

app_to_obj_0_1 = {1: b1_1, 2: b1_2, 3: b1_3, 4: b1_4}
app_to_obj_1_1 = {1: a1_1, 2: a1_2, 3: a1_3, 4: a1_4}
app_to_obj_0_2 = {1: b2_1, 2: b2_2, 3: b2_3, 4: b2_4}
app_to_obj_1_2 = {1: a2_1, 2: a2_2, 3: a2_3, 4: a2_4}

exp_info_df_0 = pd.read_excel(str(WORKING_DIR / "NORT_Round1.xlsx"), sheet_name=0)
exp_info_df_1 = pd.read_excel(str(WORKING_DIR / "NORT_Round1.xlsx"), sheet_name=1)

id_exp_0 = get_animal_id_vs_exp_ids(exp_info_df_0)
id_exp_1 = get_animal_id_vs_exp_ids(exp_info_df_1)

id_app_0 = get_animal_id_vs_apparatus(exp_info_df_0)
id_app_1 = get_animal_id_vs_apparatus(exp_info_df_1)

results_0 = border_analysis(id_exp_0, app_to_obj_0_1, app_to_obj_0_2, id_app_0)
results_1 = border_analysis(id_exp_0, app_to_obj_0_2, app_to_obj_1_2, id_app_1)
