import os
import re
from glob import glob
from pathlib import Path

import pandas as pd

from bikipy.behaviour.y_maze.experiment import YMazeExperiment
from bikipy.perimeter.radial_arm_maze import generate_radial_arm_maze_arm_perimeters

# User defined
DATASET_LABEL = "phd"

DATA_DIR = Path("/mnt/md0/Projects/Neuroscience/Imen/data/y_maze/phd/")
EXP_ID_REGEX_PATTERN = re.compile("\d+")

REPO_PATH = Path("/home/can/Software_Projects/BiKiPy")
ROOT_YMAZE = REPO_PATH / "examples" / "ymaze_behaj_analysis"

IMAGE_PATH = ROOT_YMAZE / "perimeter_images" / "phd"
RESULT_PATH = ROOT_YMAZE / "results"
ANNOTATION_PATH = IMAGE_PATH / "annotation"

# 07.06.2020
first_annotation = generate_radial_arm_maze_arm_perimeters(
    line_csv_path=ANNOTATION_PATH / "lines.csv",
    center_coco_path=ANNOTATION_PATH / "center.json",
    reference_point_coco_path=ANNOTATION_PATH / "reference.csv",
    inspect_image=IMAGE_PATH / "a_p1_1_before_1_phd.png",
    semantic_label="a_p1_1",
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

experiment_data = []
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

    trial_id_range_vs_exp_meta = {}
    for trial_id, paths in trial_id_vs_paths.items():
        trial_id_range_vs_exp_meta[trial_id] = {
            "coordinate_data_path": paths["data"],
            "video_path": paths["video"],
            "inspect": False,
        }

    experiment_data.append(
        (
            trial := YMazeExperiment(
                trial_id_vs_data=trial_id_range_vs_exp_meta,
                trial_id_range_vs_perimeter_set=trial_id_range_vs_area_set,
                point_label_for_motion_features="mid-mid-left_ear-right_ear-base_tail",
                center_triangle_meter_width=0.08,
                semantic_label=subdir,
                int_label=i,
                data_import_kwargs={
                    "init_from": "hdf",
                    "x_crop_start": 95.0,
                    "y_crop_start": 75.0,
                    "midpoint_groups": (
                        ("left_ear", "right_ear"),
                        ("mid-left_ear-right_ear", "base_tail"),
                    ),
                },
            )
        )
    )

    # trial.plot(invalid=False)


with pd.ExcelWriter(RESULT_PATH / "phd.ods") as writer:
    for experiment in experiment_data:
        print(f"Analysing {experiment.semantic_label}")
        df = experiment.export_to_dataframe()
        df.to_excel(writer, sheet_name=experiment.semantic_label)
