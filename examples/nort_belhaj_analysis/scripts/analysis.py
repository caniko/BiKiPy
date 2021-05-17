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

NOVELTY_DIR = DATA_DIR / "nort" / "Novelty"
N_1A_DIR = NOVELTY_DIR / "NORT_02.06.2020 (1A)"
N_2A_DIR = NOVELTY_DIR / "NORT (after)_24.08.2020 (2A)"
N_1B_DIR = NOVELTY_DIR / "NORT2_30.08.2020 (1B)"
N_2B_DIR = NOVELTY_DIR / "NORT2 (after)_23.11.2020 (2B)"

NORT_EXAMPLE_DIR = Path(".").resolve().parent
IMAGE_DIR = NORT_EXAMPLE_DIR / "area_images"
ANNOTATION_B1_T1 = str(IMAGE_DIR / "B1" / "b1_labels.pickle")
ANNOTATION_B1_T2 = str(IMAGE_DIR / "B2" / "b2_labels.pickle")

EXP_ID_REGEX_PATTERN = re.compile(r"\d+")
BORDER_DISTANCE = 0.03 * 224 / 0.4


def get_exp():
    return int(EXP_ID_REGEX_PATTERN.findall(Path(data_path).stem)[0])


def get_test_id_vs_meta(info_df):
    test_id_vs_meta = {}
    for i, (test_id, animal_id, stage, apparatus) in info_df[
        ["Test", "Animal", "Stage", "Apparatus"]
    ].iterrows():
        test_id_vs_meta[test_id] = {
            "animal_id": animal_id,
            "stage": stage.split(" ")[0].lower(),
            "field": int(apparatus[-1]),
        }
    return test_id_vs_meta


def standardize_stage_vs_section(stage_vs_section):
    for experiment_type, section in stage_vs_section.items():
        standard_experiment_type_name = (
            NortExperiment.trial_label_to_experiment_class_name[experiment_type]
        )
        stage_vs_section[standard_experiment_type_name] = stage_vs_section[
            experiment_type
        ]


a1_meta = get_test_id_vs_meta(
    pd.read_excel(
        str(NORT_EXAMPLE_DIR / "nort_round_1.xlsx"),
        sheet_name="NORT_02.06.2020",
        engine="openpyxl",
    )
)
a2_meta = get_test_id_vs_meta(
    pd.read_excel(
        str(NORT_EXAMPLE_DIR / "nort_round_1.xlsx"),
        sheet_name="NORT_ 24.08.2020 (after)",
        engine="openpyxl",
    )
)
b1_meta = get_test_id_vs_meta(
    pd.read_excel(
        str(NORT_EXAMPLE_DIR / "nort_round_2.xlsx"),
        sheet_name="NORT2_30.08.20",
        engine="openpyxl",
    )
)
b2_meta = get_test_id_vs_meta(
    pd.read_excel(
        str(NORT_EXAMPLE_DIR / "nort_round_2.xlsx"),
        sheet_name="NORT2_23.11.2020 (after)",
        engine="openpyxl",
    )
)

with open(ANNOTATION_B1_T1, "rb") as infile:
    before_1, before_2, before_3, before_4 = pickle.load(infile)
with open(ANNOTATION_B1_T2, "rb") as infile:
    after_1, after_2, after_3, after_4 = pickle.load(infile)


with open(IMAGE_DIR / "A_annotations.pickle", "rb") as infile:
    stage_a = pickle.load(infile)

stage_experiment_type_section_obj = {
    "before": {
        "T1": {1: before_1, 2: before_2, 3: before_3, 4: before_4},
        "T2": {1: after_1, 2: after_2, 3: after_3, 4: after_4},
    },
    "after": stage_a,
}


result_dfs = []
for trial_name, experiment_stage, trial_dir, trial_meta in zip(
    ("A1", "A2", "B1", "B2"),
    ("after", "after", "before", "before"),
    (N_1A_DIR, N_2A_DIR, N_1B_DIR, N_2B_DIR),
    (a1_meta, a2_meta, b1_meta, b2_meta),
):
    exp_ids_range_vs_exp_meta, exp_id_vs_coordinate_data_path = {}, {}
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
            nort_fields=stage_experiment_type_section_obj[experiment_stage],
            midpoint_groups=[
                ("left_ear", "right_ear"),
                ("mid-left_ear-right_ear", "tail"),
            ],
            label=trial_name,
        )
    )

result_path = resolve_stem_in_filepath(
    str(
        NORT_EXAMPLE_DIR
        / "results"
        / "nort_belhaj_analysis"
        / "nort_belhaj_analysis.xlsx"
    )
)
with pd.ExcelWriter(result_path) as writer:
    for trial in result_dfs:
        trial.export_to_dataframe().to_excel(writer, sheet_name=str(trial.label))
