import os
import re
from glob import glob
from pathlib import Path

import pandas as pd

from bikipy.behaviour.y_maze.trial import YMazeTrial
from bikipy.perimeter.parallelogram.classes import ParallelogramPerimeter
from bikipy.perimeter.radial_arm_maze import generate_radial_arm_maze_arm_perimeters
from bikipy.perimeter.triangular import TriangularPerimeter
from bikipy.utils.video import get_video_data

# User defined
DATASET_LABEL = "phd"

DATA_DIR = Path("C:/Users/Can/Projects/Neuroscience/Imen/data/results/phd")
EXP_ID_REGEX_PATTERN = re.compile("\d+")

REPO_PATH = Path("C:/Users/Can/Projects/Neuroscience/bikipy")
IMAGE_ROOT = REPO_PATH / "examples/data/images/results/phd"
RESULT_PATH = REPO_PATH / "examples/data/results/results"

IMAGE_A = IMAGE_ROOT / "A"

# 07.06.2020
first_annotation = generate_radial_arm_maze_arm_perimeters(
    line_csv_path=IMAGE_ROOT / "coco_line_labels.csv",
    center_coco_path=IMAGE_ROOT / "coco_triangle.json",
    inspect_image=IMAGE_A / "before_1_phd.png",
    label="before_1"
)
first_annotation.reference_point = 

exp_id_vs_areas = {
    "07.06.2020 (1A)": {
        1: first_annotation,
        32: first_annotation
    },
    "26.08.2020 (2A)": {},
    "31.08.2020 (1B)": {},
    "25.11.2020 (2B)": {},
}

inspect_image = IMAGE_A / "before_32_phd.png"
exp_id_vs_areas["07.06.2020 (1A)"][32] = {
    "arms": (
        ParallelogramPerimeter(
            base=[
                [309.44300412905403, 194.07402857692392],
                [288.38603678988756, 236.187963255257],
            ],
            apex=[
                [175.3803120696939, 121.77844071245221],
                [152.9195469079163, 162.48857756817415],
            ],
            inspect_image=inspect_image,
            semantic_label="A",
        ),
        ParallelogramPerimeter(
            base=[
                [311.8209525619013, 194.7171785444296],
                [336.98296504900634, 234.25748388130893],
            ],
            apex=[
                [438.3499296399152, 117.79331179813704],
                [464.94977141199763, 155.17687320755027],
            ],
            inspect_image=inspect_image,
            semantic_label="B",
        ),
        ParallelogramPerimeter(
            base=[
                [288.431462397926, 235.8676906087083],
                [335.2359574417894, 235.8676906087083],
            ],
            apex=[
                [292.14610486172467, 396.3402450448116],
                [339.6935283983479, 394.11145956653235],
            ],
            inspect_image=inspect_image,
            semantic_label="C",
        ),
    ),
    "center": TriangularPerimeter(
        base_a=[287.1835791862539, 234.81489706597512],
        base_b=[335.3210406873076, 234.81489706597512],
        apex=[309.5923974712272, 194.1470416599126],
        inspect_image=inspect_image,
        semantic_label="X",
    ),
}

inspect_image = IMAGE_A / "after_1_phd.png"
exp_id_vs_areas["26.08.2020 (2A)"][1] = {
    "arms": (
        ParallelogramPerimeter(
            base=[
                [308.19455616608803, 195.86862513168256],
                [283.84982088825086, 237.3978794291695],
            ],
            apex=[
                [174.29851213798355, 120.68635442071474],
                [148.52173360850884, 159.35152221492677],
            ],
            inspect_image=inspect_image,
            semantic_label="A",
        ),
        ParallelogramPerimeter(
            base=[
                [307.06098087637076, 196.75741594708666],
                [331.03489232273137, 236.46420678012123],
            ],
            apex=[
                [438.9174938313537, 118.84220374641495],
                [462.14222054501545, 159.2981793121483],
            ],
            inspect_image=inspect_image,
            semantic_label="B",
        ),
        ParallelogramPerimeter(
            base=[
                [281.60649616867585, 236.4269205757078],
                [330.61367602655605, 238.7063242900278],
            ],
            apex=[
                [285.0256017401558, 392.5660750066285],
                [331.75337788371604, 393.7057768637885],
            ],
            inspect_image=inspect_image,
            semantic_label="C",
        ),
    ),
    "center": TriangularPerimeter(
        base_a=[285.73376623376623, 238.98051948051943],
        base_b=[329.8896103896104, 237.68181818181813],
        apex=[306.51298701298697, 194.82467532467524],
        inspect_image=inspect_image,
        semantic_label="X",
    ),
}

IMAGE_B = IMAGE_ROOT / "B"

inspect_image = IMAGE_B / "before_1.png"
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

inspect_image = IMAGE_B / "before_24.png"
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

inspect_image = IMAGE_B / "after_1.png"
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
