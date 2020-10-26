from pathlib import Path
from glob import glob
import pickle
import re

import pandas as pd
import numpy as np

from bikipy.behaviour.nort.experiment import attention
from bikipy.behaviour.nort import NortObject
from bikipy.readers.deeplabcut import DeepLabCutReader


WORKING_DIR = Path("C:/Users/Can/Projects/Neuroscience/Imen/data")
NORT_DIR = WORKING_DIR / "nort"
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

        t1_a_at = attention(
            app_a_t1,
            t1["mid-left_ear-right_ear"],
            t1["nose"],
            t1["mid-mid-left_ear-right_ear-tail"],
            BORDER_DISTANCE,
            14.99,
        )
        t1_B_at = attention(
            app_b_t1,
            t1["mid-left_ear-right_ear"],
            t1["nose"],
            t1["mid-mid-left_ear-right_ear-tail"],
            BORDER_DISTANCE,
            14.99,
        )
        t2_a_at = attention(
            app_a_t2,
            t2["mid-left_ear-right_ear"],
            t2["nose"],
            t2["mid-mid-left_ear-right_ear-tail"],
            BORDER_DISTANCE,
            14.99,
        )
        t2_B_at = attention(
            app_b_t2,
            t2["mid-left_ear-right_ear"],
            t2["nose"],
            t2["mid-mid-left_ear-right_ear-tail"],
            BORDER_DISTANCE,
            14.99,
        )

        result[animal] = {
            "t1_a": t1_a_at,
            "t1_b": t1_B_at,
            "t2_a": t2_a_at,
            "t2_b": t2_B_at,
        }

    return result


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "bikipy.preferance.gaze":
            renamed_module = "bikipy.behaviour.nort"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


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
    ) = renamed_load(infile)

with open(IMPORTED_DLC_FILES, "rb") as infile:
    id_dlc_0, id_dlc_1 = pickle.load(infile)

b1_1["A"] = NortObject(b1_1["A"].sides, border_distance=BORDER_DISTANCE)
b1_2["A"] = NortObject(b1_2["A"].sides, border_distance=BORDER_DISTANCE)
b1_3["A"] = NortObject(b1_3["A"].sides, border_distance=BORDER_DISTANCE)
b1_4["A"] = NortObject(b1_4["A"].sides, border_distance=BORDER_DISTANCE)
b2_1["A"] = NortObject(b2_1["A"].sides, border_distance=BORDER_DISTANCE)
b2_2["A"] = NortObject(b2_2["A"].sides, border_distance=BORDER_DISTANCE)
b2_3["A"] = NortObject(b2_3["A"].sides, border_distance=BORDER_DISTANCE)
b2_4["A"] = NortObject(b2_4["A"].sides, border_distance=BORDER_DISTANCE)
a1_1["A"] = NortObject(a1_1["A"].sides, border_distance=BORDER_DISTANCE)
a1_2["A"] = NortObject(a1_2["A"].sides, border_distance=BORDER_DISTANCE)
a1_3["A"] = NortObject(a1_3["A"].sides, border_distance=BORDER_DISTANCE)
a1_4["A"] = NortObject(a1_4["A"].sides, border_distance=BORDER_DISTANCE)
a2_1["A"] = NortObject(a2_1["A"].sides, border_distance=BORDER_DISTANCE)
a2_2["A"] = NortObject(a2_2["A"].sides, border_distance=BORDER_DISTANCE)
a2_3["A"] = NortObject(a2_3["A"].sides, border_distance=BORDER_DISTANCE)
a2_4["A"] = NortObject(a2_4["A"].sides, border_distance=BORDER_DISTANCE)

b1_1["B"] = NortObject(b1_1["B"].sides, border_distance=BORDER_DISTANCE)
b1_2["B"] = NortObject(b1_2["B"].sides, border_distance=BORDER_DISTANCE)
b1_3["B"] = NortObject(b1_3["B"].sides, border_distance=BORDER_DISTANCE)
b1_4["B"] = NortObject(b1_4["B"].sides, border_distance=BORDER_DISTANCE)
b2_1["B"] = NortObject(b2_1["B"].sides, border_distance=BORDER_DISTANCE)
b2_2["B"] = NortObject(b2_2["B"].sides, border_distance=BORDER_DISTANCE)
b2_3["B"] = NortObject(b2_3["B"].sides, border_distance=BORDER_DISTANCE)
b2_4["B"] = NortObject(b2_4["B"].sides, border_distance=BORDER_DISTANCE)
a1_1["B"] = NortObject(a1_1["B"].sides, border_distance=BORDER_DISTANCE)
a1_2["B"] = NortObject(a1_2["B"].sides, border_distance=BORDER_DISTANCE)
a1_3["B"] = NortObject(a1_3["B"].sides, border_distance=BORDER_DISTANCE)
a1_4["B"] = NortObject(a1_4["B"].sides, border_distance=BORDER_DISTANCE)
a2_1["B"] = NortObject(a2_1["B"].sides, border_distance=BORDER_DISTANCE)
a2_2["B"] = NortObject(a2_2["B"].sides, border_distance=BORDER_DISTANCE)
a2_3["B"] = NortObject(a2_3["B"].sides, border_distance=BORDER_DISTANCE)
a2_4["B"] = NortObject(a2_4["B"].sides, border_distance=BORDER_DISTANCE)

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
