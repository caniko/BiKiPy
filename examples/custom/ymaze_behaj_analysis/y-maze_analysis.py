import os
import re
from datetime import date
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

# User defined
from bikipy.behaviour.radial_arm.y_maze import YMazeExperiment
from bikipy.perimeter.polygon.radial_maze import generate_radial_maze_perimeters
from bikipy.utils.ranged_dict import RangeDict

DATASET_LABEL = "phd"

DATA_DIR = Path("/mnt/md0/Projects/Neuroscience/Imen/data/y_maze/phd/")
EXP_ID_REGEX_PATTERN = re.compile("\d+")

ROOT_DIR = Path(__file__).parent

IMAGE_PATH = ROOT_DIR / "perimeter_images" / "phd"
RESULT_DIR = ROOT_DIR / "results"
ANALYSIS_DIR = RESULT_DIR / "for_analysis"
ANNOTATION_PATH = IMAGE_PATH / "annotation"

os.makedirs(ANALYSIS_DIR, exist_ok=True)

# 07.06.2020
first_annotation = generate_radial_maze_perimeters(
    line_csv_path=ANNOTATION_PATH / "lines.csv",
    center_coco_path=ANNOTATION_PATH / "center.json",
    reference_point_coco_path=ANNOTATION_PATH / "reference.csv",
    inspect_image_path=IMAGE_PATH / "a_p1_1_before_1_phd.png",
    # inspect=True,
)

re_referenced = first_annotation.change_reference_with_coco_with_plural_references(
    IMAGE_PATH / "references.csv", image_root=IMAGE_PATH, map_to_image_names=False
)

exp_period_to_perimeter_set = {
    "07.06.2020 (1A)": {1: first_annotation.group, 32: re_referenced[0].group},
    "26.08.2020 (2A)": {1: re_referenced[1].group},
    "31.08.2020 (1B)": {1: re_referenced[2].group, 23: re_referenced[3].group},
    "25.11.2020 (2B)": {1: re_referenced[4].group},
}
round_dirs = (
    [
        DATA_DIR / "Y-maze_07.06.2020 (1A)",
        DATA_DIR / "Y-maze (after)_26.08.2020 (2A)",
    ],
    [
        DATA_DIR / "Y-maze2_31.08.2020 (1B)",
        DATA_DIR / "Y-maze2 (after)_25.11.2020 (2B)",
    ],
)
common_trial_keyword_arguments = {
    "corridor_meter_width": 0.08,
}

experiment_obj_sets = []
for round_id, data_dirs in enumerate(round_dirs):
    experiment_objs = []
    metadata_df = pd.read_excel(ROOT_DIR / "y-maze_metadata.xlsx", sheet_name=round_id)
    for metadata_animal_id_cidx, data_dir in enumerate(data_dirs, start=8):
        print(f"Reading {data_dir}")
        stage = data_dir.name

        trial_set_date_n_id = stage.split("_")[1]
        trial_id_range_to_area_set = RangeDict(exp_period_to_perimeter_set[trial_set_date_n_id])

        trial_set_date = trial_set_date_n_id.split(" ")[0]
        day, month, year = map(int, trial_set_date.split("."))
        date_obj = date(year, month, day)

        trial_id_to_paths = {}
        for video_path in glob(str(data_dir / "*.mp4")):
            trial_id = int(EXP_ID_REGEX_PATTERN.findall(Path(video_path).stem)[0])
            trial_id_to_paths[trial_id] = {"video": video_path}
        for data_path in glob(str(data_dir / "*.h5")):
            trial_id = int(EXP_ID_REGEX_PATTERN.findall(Path(data_path).stem)[0])
            trial_id_to_paths[trial_id]["data"] = data_path

        trial_id_to_exp_meta = {}
        for trial_id, paths in trial_id_to_paths.items():
            if not np.any(trial_id == metadata_df.iloc[:, metadata_animal_id_cidx]):
                continue
            trial_id_to_exp_meta[trial_id] = {
                "animal_id": int(
                    metadata_df.loc[metadata_df.iloc[:, metadata_animal_id_cidx] == trial_id]["Animal ID"].iloc[0]
                ),
                "coordinate_data_path": paths["data"],
                "video_path": paths["video"],
                "inspect": False,
                "timestamp": date_obj,
            }

        experiment_objs.append(
            (
                trial := YMazeExperiment(
                    object_tracking_label_for_kinematics="center_eye",
                    common_trial_keyword_arguments=common_trial_keyword_arguments,
                    trial_id_to_keyword_arguments=trial_id_to_exp_meta,
                    trial_id_range_to_keyword_arguments=trial_id_range_to_area_set,
                    stage=stage,
                    int_id=round_id,
                    data_reader_kwargs={
                        "init_from": "hdf",
                        "x_axis_crop_end_point": 95.0,
                        "y_axis_crop_end_point": 75.0,
                        "midpoint_groups": {
                            "center_eye": ("left_ear", "right_ear"),
                            "torso": ("center_eye", "base_tail"),
                        },
                    },
                )
            )
        )
        # trial.plot(invalid=False)

    experiment_obj_sets.append(experiment_objs)

with pd.ExcelWriter(
    RESULT_DIR / "ymaze_analysis.xlsx",
    engine_kwargs={
        "strings_to_formulas": False,
        "strings_to_urls": False,
    },
) as writer:
    for i, experiment_objs in enumerate(experiment_obj_sets):
        df = pd.concat(
            [experiment.animal_id_indexed_feature_frame for experiment in experiment_objs],
            axis=1,
        )

        label = " & ".join([experiment.stage for experiment in experiment_objs])

        df.to_parquet(RESULT_DIR / "for_analysis" / f"{label}.parquet")
        df.to_excel(writer, sheet_name=f"{label}")
