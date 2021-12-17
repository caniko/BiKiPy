import os
import re
from glob import glob
from pathlib import Path

import pandas as pd

# User defined
from bikipy.behaviour.radial_arm.y_maze import YMazeExperiment
from bikipy.perimeter.radial_arm_maze import generate_radial_arm_maze_arm_perimeters

DATASET_LABEL = "phd"

DATA_DIR = Path("/mnt/md0/Projects/Neuroscience/Imen/data/y_maze/phd/")
EXP_ID_REGEX_PATTERN = re.compile("\d+")

ROOT_DIR = Path(__file__).parent

IMAGE_PATH = ROOT_DIR / "perimeter_images" / "phd"
RESULT_DIR = ROOT_DIR / "results"
ANNOTATION_PATH = IMAGE_PATH / "annotation"

# 07.06.2020
first_annotation = generate_radial_arm_maze_arm_perimeters(
    line_csv_path=ANNOTATION_PATH / "lines.csv",
    center_coco_path=ANNOTATION_PATH / "center.json",
    reference_point_coco_path=ANNOTATION_PATH / "reference.csv",
    inspect_image=IMAGE_PATH / "a_p1_1_before_1_phd.png",
    label="a_p1_1",
    # inspect=True,
)

re_referenced = first_annotation.change_reference_with_coco(
    IMAGE_PATH / "references.csv", image_root=IMAGE_PATH
)

exp_period_vs_perimeter_set = {
    "07.06.2020 (1A)": {1: first_annotation.group, 32: re_referenced[0].group},
    "26.08.2020 (2A)": {1: re_referenced[1].group},
    "31.08.2020 (1B)": {1: re_referenced[2].group, 23: re_referenced[3].group},
    "25.11.2020 (2B)": {1: re_referenced[4].group},
}

experiments = []
for i, subdir in enumerate(os.listdir(str(DATA_DIR)), start=1):
    print(f"Reading {subdir}")

    trial_name = subdir.split("_")[1]
    trial_id_range_vs_area_set = exp_period_vs_perimeter_set[trial_name]

    trial_id_vs_paths = {}
    for video_path in glob(str(DATA_DIR / subdir / "*.mp4")):
        trial_id = int(EXP_ID_REGEX_PATTERN.findall(Path(video_path).stem)[0])
        trial_id_vs_paths[trial_id] = {"video": video_path}
    for data_path in glob(str(DATA_DIR / subdir / "*.h5")):
        trial_id = int(EXP_ID_REGEX_PATTERN.findall(Path(data_path).stem)[0])
        trial_id_vs_paths[trial_id]["data"] = data_path

    trial_id_vs_exp_meta = {}
    for trial_id, paths in trial_id_vs_paths.items():
        trial_id_vs_exp_meta[trial_id] = {
            "coordinate_data_path": paths["data"],
            "video_path": paths["video"],
            "inspect": False,
        }

    experiments.append(
        (
            trial := YMazeExperiment(
                trial_id_vs_keyword_arguments=trial_id_vs_exp_meta,
                trial_id_range_vs_keyword_arguments=trial_id_range_vs_area_set,
                point_label_for_motion_features="torso",
                corridor_meter_width=0.08,
                label=subdir,
                int_id=i,
                data_import_kwargs={
                    "init_from": "hdf",
                    "x_axis_crop_end_point": 95.0,
                    "y_axis_crop_end_point": 75.0,
                    "midpoint_groups": {
                        "center_eye": ("left_ear", "right_ear"),
                        "torso": ("center_eye", "tail"),
                    },
                },
            )
        )
    )
    # trial.plot(invalid=False)


with pd.ExcelWriter(
    RESULT_DIR / "ymaze_analysis.xlsx",
    engine_kwargs={
        "strings_to_formulas": False,
        "strings_to_urls": False,
    },
) as writer:
    for experiment in experiments:
        experiment.animal_summary_frame.to_parquet(
            RESULT_DIR / "for_analysis" / f"{experiment.timestamp}.parquet"
        )
        experiment.animal_summary_frame.to_excel(
            writer, sheet_name=f"{experiment.timestamp}"
        )
