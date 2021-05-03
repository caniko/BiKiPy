import pickle
import re
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from bikipy.utils.video import get_video_data
from bikipy.behaviour.nort.experiment import NortExperiment
from bikipy.utils.misc import resolve_stem_in_filepath


DATA_DIR = Path("/mnt/md0/Projects/Neuroscience/Imen/data")
NORT_DIR = DATA_DIR / "nort"

NOVELTY_DIR = NORT_DIR / "Novelty"
N_1A_DIR = NOVELTY_DIR / "NORT_02.06.2020 (1A)"
N_2A_DIR = NOVELTY_DIR / "NORT (after)_24.08.2020 (2A)"
N_1B_DIR = NOVELTY_DIR / "NORT2_30.08.2020 (1B)"
N_2B_DIR = NOVELTY_DIR / "NORT2 (after)_23.11.2020 (2B)"

EXAMPLE_DATA_DIR = Path("C:/Users/Can/Projects/Neuroscience/bikipy/examples/data")

IMAGE_DIR = EXAMPLE_DATA_DIR / "images" / "nort"
ANNOTATION_B1_T1 = str(IMAGE_DIR / "B1" / "b1_labels.pickle")
ANNOTATION_B1_T2 = str(IMAGE_DIR / "B2" / "b2_labels.pickle")

EXP_ID_FINDER = re.compile(r"\d+")
BORDER_DISTANCE = 0.03 * 224 / 0.4


def get_exp():
    return int(EXP_ID_FINDER.findall(Path(data_path).stem)[0])


def get_test_id_vs_meta(info_df):
    # animal_vs_tests = {
    #     animal_id: tuple([info_df["Test"].values[i] for i in test_ids[:2]])
    #     for animal_id, test_ids in zip(
    #         info_df["Animal"], [
    #             np.where(info_df["Animal"].values == i)[0]
    #             for i in range(1, info_df["Animal"].values.max() + 1)
    #         ]
    #     )
    # }

    test_id_vs_meta = {}
    for i, (test_id, animal_id, stage, apparatus) in info_df[
        ["Test", "Animal", "Stage", "Apparatus"]
    ].iterrows():
        test_id_vs_meta[test_id] = {
            "animal_id": animal_id,
            "stage": stage.split(" ")[0].lower(),
            "field": int(apparatus[-1]),
            # "other_test_id": animal_vs_tests[animal_id][animal_vs_tests[i] != test_id]
        }
    return test_id_vs_meta


a1_meta = get_test_id_vs_meta(
    pd.read_excel(
        str(EXAMPLE_DATA_DIR / "NORT_Round1.xlsx"),
        sheet_name="NORT_02.06.2020",
        engine="openpyxl",
    )
)
a2_meta = get_test_id_vs_meta(
    pd.read_excel(
        str(EXAMPLE_DATA_DIR / "NORT_Round1.xlsx"),
        sheet_name="NORT_ 24.08.2020 (after)",
        engine="openpyxl",
    )
)
b1_meta = get_test_id_vs_meta(
    pd.read_excel(
        str(EXAMPLE_DATA_DIR / "NORT2_Round2.xlsx"),
        sheet_name="NORT2_30.08.20",
        engine="openpyxl",
    )
)
b2_meta = get_test_id_vs_meta(
    pd.read_excel(
        str(EXAMPLE_DATA_DIR / "NORT2_Round2.xlsx"),
        sheet_name="NORT2_23.11.2020 (after)",
        engine="openpyxl",
    )
)

with open(ANNOTATION_B1_T1, "rb") as infile:
    before_1, before_2, before_3, before_4 = pickle.load(infile)
with open(ANNOTATION_B1_T2, "rb") as infile:
    after_1, after_2, after_3, after_4 = pickle.load(infile)

app_to_obj = {
    "t1": {1: before_1, 2: before_2, 3: before_3, 4: before_4},
    "t2": {1: after_1, 2: after_2, 3: after_3, 4: after_4},
}

result_dfs = []
for trial_name, trial_dir, trial_meta in zip(
    ("A1", "A2", "B1", "B2"),
    (N_1A_DIR, N_2A_DIR, N_1B_DIR, N_2B_DIR),
    (a1_meta, a2_meta, b1_meta, b2_meta),
):
    exp_ids_range_vs_exp_meta, exp_id_vs_coordinate_data_path = {}, {}

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

    for time_dir in os.listdir(trial_dir):
        for data_path in glob(str(trial_dir / time_dir / "*.h5")):
            exp_id = get_exp()
            exp_id_vs_coordinate_data_path[exp_id] = data_path

        for data_path in glob(str(trial_dir / time_dir / "*.mp4")):
            exp_id = get_exp()
            frame, x, y, fps = get_video_data(data_path)

            exp_ids_range_vs_exp_meta[exp_id] = {
                **trial_meta[exp_id],
                "recording_resolution": (x, y),
                "fps": fps,
                "guiding_image": frame,
            }

    result_dfs.append(
        NortExperiment(
            exp_ids_range_vs_exp_meta=exp_ids_range_vs_exp_meta,
            nose_label="nose",
            eye_center_label="mid-left_ear-right_ear",
            torso_label="mid-mid-left_ear-right_ear-tail",
            experiment_box_real_length=0.4,
            center_size_real_length=0.2,
            max_radians_gaze_and_object=1 / 2 * np.pi,
            exp_id_vs_coordinate_data_path=exp_id_vs_coordinate_data_path,
            nort_fields=app_to_obj["t1"],
            midpoint_groups=[
                ("left_ear", "right_ear"),
                ("mid-left_ear-right_ear", "tail"),
            ],
            label=trial_name,
        )
    )

result_path = resolve_stem_in_filepath(
    str(EXAMPLE_DATA_DIR / "results" / "nort" / "nort.xlsx")
)
with pd.ExcelWriter(result_path) as writer:
    for trial in result_dfs:
        trial.export_to_dataframe().to_excel(writer, sheet_name=str(trial.label))
