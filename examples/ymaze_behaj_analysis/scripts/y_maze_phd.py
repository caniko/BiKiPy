import os
import re
from glob import glob
from pathlib import Path

import pandas as pd

from bikipy.behaviour.y_maze.trial import YMazeTrial
from bikipy.perimeter.parallelogram.classes import ParallelogramPerimeter
from bikipy.perimeter.radial_arm_maze import generate_radial_arm_maze_arm_perimeters
from bikipy.perimeter.triangular import TriangularPerimeter
from bikipy.utils.store import RangeDict
from bikipy.utils.video import get_video_data

# User defined
DATASET_LABEL = "phd"

DATA_DIR = Path("/mnt/md0/Projects/Neuroscience/Imen/data/y_maze/phd/")
EXP_ID_REGEX_PATTERN = re.compile("\d+")

REPO_PATH = Path("/home/can/Software_Projects/BiKiPy")
ROOT_YMAZE = REPO_PATH / "examples" / "ymaze_behaj_analysis"

IMAGE_PATH = ROOT_YMAZE / "area_images" / "phd"
RESULT_PATH = ROOT_YMAZE / "results"
ANNOTATION_PATH = IMAGE_PATH / "annotation"

# 07.06.2020
first_annotation = generate_radial_arm_maze_arm_perimeters(
    line_csv_path=ANNOTATION_PATH / "coco_line_labels.csv",
    center_coco_path=ANNOTATION_PATH / "coco_triangle.json",
    reference_path=ANNOTATION_PATH / "reference.csv",
    inspect_image=IMAGE_PATH / "a_p1_1_before_1_phd.png",
    label="a_p1_1",
)

re_referenced = first_annotation.change_reference_with_coco(
    IMAGE_PATH / "reference_points_2021-09-30-10-12-16.csv", IMAGE_PATH
)

exp_id_vs_areas = {
    "07.06.2020 (1A)": {
        1: first_annotation,
        32: re_referenced[0],
    },
    "26.08.2020 (2A)": {1: re_referenced[1]},
    "31.08.2020 (1B)": {},
    "25.11.2020 (2B)": {},
}

inspect_image = IMAGE_PATH / "before_1.png"
exp_id_vs_areas["31.08.2020 (1B)"][1] = {
    "arms": (
        ParallelogramPerimeter(
            base=[
                [306.67993702619594, 197.67705743226765],
                [282.74619802583584, 236.4269205757078],
            ],
            apex=[
                [171.0554160241553, 122.4567348597073],
                [147.1216770237952, 163.48600171746745],
            ],
            inspect_image=inspect_image,
            semantic_label="A",
        ),
        ParallelogramPerimeter(
            base=[
                [305.54023516903595, 196.5373555751076],
                [329.47397416939606, 239.84602614718784],
            ],
            apex=[
                [432.04714131379654, 121.3170330025473],
                [459.3999858856367, 158.92719428882742],
            ],
            inspect_image=inspect_image,
            semantic_label="B",
        ),
        ParallelogramPerimeter(
            base=[
                [281.60649616867585, 237.5666224328678],
                [331.75337788371604, 237.5666224328678],
            ],
            apex=[
                [285.0256017401558, 400.5439880067486],
                [331.75337788371604, 398.26458429242854],
            ],
            inspect_image=inspect_image,
            semantic_label="C",
        ),
    ),
    "center": TriangularPerimeter(
        base_a=[282.41830190781593, 237.8687618387333],
        base_b=[329.9589404875393, 237.8687618387333],
        apex=[305.0001052331845, 198.64773501046147],
        inspect_image=inspect_image,
        semantic_label="X",
    ),
}

inspect_image = IMAGE_PATH / "before_24.png"
exp_id_vs_areas["31.08.2020 (1B)"][23] = {
    "arms": (
        ParallelogramPerimeter(
            base=[
                [308.4602835740226, 194.5632485451174],
                [286.0356445542109, 234.19749425455205],
            ],
            apex=[
                [174.95545592118992, 119.9882862234179],
                [151.4878104353404, 160.14403516587146],
            ],
            inspect_image=inspect_image,
            semantic_label="A",
        ),
        ParallelogramPerimeter(
            base=[
                [308.98178680704143, 194.5632485451174],
                [333.49243875892876, 233.6759910215332],
            ],
            apex=[
                [438.3145885957231, 117.9022732913424],
                [461.7822340815726, 155.97200930172045],
            ],
            inspect_image=inspect_image,
            semantic_label="B",
        ),
        ParallelogramPerimeter(
            base=[
                [286.0356445542109, 234.71899748757096],
                [334.0139419919476, 235.24050072058986],
            ],
            apex=[
                [289.1646639523241, 394.82049002436634],
                [335.57845169100426, 394.2989867913475],
            ],
            inspect_image=inspect_image,
            semantic_label="C",
        ),
    ),
    "center": TriangularPerimeter(
        base_a=[285.38959181904863, 235.4917299097471],
        base_b=[332.930230398772, 234.89747192750053],
        apex=[308.56565312666373, 194.48792913473568],
        inspect_image=inspect_image,
        semantic_label="X",
    ),
}

inspect_image = IMAGE_PATH / "after_1.png"
exp_id_vs_areas["25.11.2020 (2B)"][1] = {
    "arms": (
        ParallelogramPerimeter(
            base=[
                [305.85276740892823, 197.17076471021176],
                [281.3421154570409, 238.89102335172197],
            ],
            apex=[
                [168.6974171249634, 119.9882862234179],
                [145.2297716391139, 160.66553839889036],
            ],
            inspect_image=inspect_image,
            semantic_label="A",
        ),
        ParallelogramPerimeter(
            base=[
                [304.2882577098716, 197.69226794323066],
                [327.7559031957211, 238.36952011870312],
            ],
            apex=[
                [436.7500788966664, 118.42377652436124],
                [461.7822340815726, 158.57952546681486],
            ],
            inspect_image=inspect_image,
            semantic_label="B",
        ),
        ParallelogramPerimeter(
            base=[
                [280.82061222402206, 238.89102335172197],
                [329.32041289477763, 238.36952011870312],
            ],
            apex=[
                [284.4711348551542, 393.77748355832864],
                [331.40642582685325, 393.25598032530974],
            ],
            inspect_image=inspect_image,
            semantic_label="C",
        ),
    ),
    "center": TriangularPerimeter(
        base_a=[281.2297859433228, 239.05727780322633],
        base_b=[328.77042452304624, 237.8687618387333],
        apex=[304.405847250938, 198.0534770282149],
        inspect_image=inspect_image,
        semantic_label="X",
    ),
}

trial_datas = []
for subdir in os.listdir(str(DATA_DIR)):
    print(f"Reading {subdir}")

    trial_name = subdir.split("_")[1]
    exp_id_range_vs_area_sets = exp_id_vs_areas[trial_name]

    exp_id_vs_dlc_path, exp_id_vs_fps = {}, {}
    for file_path in glob(str(DATA_DIR / subdir / "*.h5")):
        exp_id = int(EXP_ID_REGEX_PATTERN.findall(Path(file_path).stem)[0])
        exp_id_vs_dlc_path[exp_id] = file_path

    for file_path in glob(str(DATA_DIR / subdir / "*.mp4")):
        exp_id = int(EXP_ID_REGEX_PATTERN.findall(Path(file_path).stem)[0])

        _, _x, _y, fps = get_video_data(file_path)
        exp_id_vs_fps[exp_id] = fps

    trial_datas.append(
        (
            trial := YMazeTrial(
                exp_id_range_vs_area_sets=exp_id_range_vs_area_sets,
                feature_tracking_point="mid-mid-left_ear-right_ear-base_tail",
                trial_id_vs_coordinate_data_path=exp_id_vs_dlc_path,
                fps=exp_id_vs_fps,
                center_triangle_meter_width=0.08,
                label=subdir,
                midpoint_groups=(
                    ("left_ear", "right_ear"),
                    ("mid-left_ear-right_ear", "base_tail"),
                ),
                x_crop_start=95.0,
                y_crop_start=75.0,
                # debug=True
            )
        )
    )

    # trial.plot(invalid=False)


with pd.ExcelWriter(RESULT_PATH / "phd.ods") as writer:
    for trial in trial_datas:
        print(f"Analysing {trial.label}")
        df = trial.export_to_dataframe()
        df.to_excel(writer, sheet_name=trial.label)
