import pickle
import re
from pathlib import Path

import pandas as pd

from bikipy.behaviour.nort.workflow import deeplabcut_workflow

DATA_DIR = Path("C:/Users/Can/Projects/Neuroscience/Imen/data")
NORT_DIR = DATA_DIR / "nort"

NOVELTY_DIR = NORT_DIR / "Novelty"
N_1B_DIR = NOVELTY_DIR / "NORT2_30.08.2020 (1B)"
N_2B_DIR = NOVELTY_DIR / "NORT2 (after)_23.11.2020 (2B)"

WORKING_DIR = Path("C:/Users/Can/Projects/Neuroscience/bikipy/examples/data")
ANNOTATION_B1 = str(WORKING_DIR / "b1_labels.pickle")
ANNOTATION_B2 = str(WORKING_DIR / "b2_labels.pickle")

EXP_ID_FINDER = re.compile("\d+")
BORDER_DISTANCE = 0.03 * 224 / 0.4


def get_test_id_vs_meta(info_df):
    test_id_vs_meta = {}
    for row in info_df[["Test", "Animal", "Stage", "Apparatus"]].iterrows():
        test_id, animal_id, stage, apparatus = row[1]

        test_id_vs_meta[test_id] = {
            "animal_id": animal_id,
            "stage": stage.split(" ")[0].lower(),
            "field": int(apparatus[-1]),
        }

    return test_id_vs_meta


with open(ANNOTATION_B1, "rb") as infile:
    before_1, before_2, before_3, before_4 = pickle.load(infile)
with open(ANNOTATION_B2, "rb") as infile:
    after_1, after_2, after_3, after_4 = pickle.load(infile)

app_to_obj = {
    "1b": {1: before_1, 2: before_2, 3: before_3, 4: before_4},
    "2b": {1: after_1, 2: after_2, 3: after_3, 4: after_4},
}
for time in app_to_obj.keys():
    for field in app_to_obj[time].keys():
        obj = app_to_obj[time][field]

        obj.constant_object.border_distance = BORDER_DISTANCE
        obj.variable_object.border_distance = BORDER_DISTANCE
        obj.novel_object.border_distance = BORDER_DISTANCE

        obj.constant_object.semantic_label = f"{field} constant"
        obj.variable_object.semantic_label = f"{field} variable"
        obj.novel_object.semantic_label = f"{field} novel"

        obj.constant_object.sides = obj.constant_object.sides
        obj.variable_object.sides = obj.variable_object.sides
        obj.novel_object.sides = obj.novel_object.sides

info_df_before = pd.read_excel(
    str(WORKING_DIR / "NORT2_Round2.xlsx"),
    sheet_name="NORT2_30.08.20",
    engine="openpyxl",
)
info_df_after = pd.read_excel(
    str(WORKING_DIR / "NORT2_Round2.xlsx"),
    sheet_name="NORT2_23.11.2020 (after)",
    engine="openpyxl",
)

before_meta = get_test_id_vs_meta(info_df_before)
after_meta = get_test_id_vs_meta(info_df_after)

trial_base = {"train": {}, "novel": {}}

deeplabcut_workflow()
