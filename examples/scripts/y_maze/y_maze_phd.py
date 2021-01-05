import os
import re
from glob import glob
from pathlib import Path

import pandas as pd

from bikipy.behaviour.y_maze.trial import YMazeTrial
from bikipy.border.parallelogram.classes import ParallelogramBorder
from bikipy.border.triangular import TriangularBorder
from bikipy.utils.video import get_video_data

# User defined
DATASET_LABEL = "phd"

DATA_DIR = Path("C:/Users/Can/Projects/Neuroscience/Imen/data/y_maze/phd")
EXP_ID_FINDER = re.compile("\d+")

REPO_PATH = Path("C:/Users/Can/Projects/Neuroscience/bikipy")
IMAGE_ROOT = REPO_PATH / "examples/data/images/y_maze/phd"
RESULT_PATH = REPO_PATH / "examples/data/results/y_maze"

exp_id_vs_areas = {
    "07.06.2020 (1A)": {},
    "26.08.2020 (2A)": {},
    "31.08.2020 (1B)": {},
    "25.11.2020 (2B)": {},
}

IMAGE_A = IMAGE_ROOT / "A"
guiding_image = IMAGE_A / "before_1_phd.png"
exp_id_vs_areas["07.06.2020 (1A)"][1] = {
    "arms": [
        ParallelogramBorder(
            base=[
                [308.9593407405159, 193.11825000362762],
                [287.30500545447586, 236.4269205757078],
            ],
            apex=[
                [176.22595780600852, 118.83962548259478],
                [151.93274437947562, 159.54717230543366],
            ],
            guiding_image=guiding_image,
            semantic_label="A",
        ),
        ParallelogramBorder(
            base=[
                [310.16190043086186, 193.75141770204982],
                [332.955937574062, 235.5404857979167],
            ],
            apex=[
                [437.4286078137291, 115.2386230976939],
                [461.4889803537737, 153.8618527014496],
            ],
            guiding_image=guiding_image,
            semantic_label="B",
        ),
        ParallelogramBorder(
            base=[
                [287.52247603020544, 235.89313914308434],
                [334.5558421452106, 235.20147199433427],
            ],
            apex=[
                [291.67247892270586, 395.66825050435193],
                [338.01417788896094, 392.2099147606016],
            ],
            guiding_image=guiding_image,
            semantic_label="C",
        ),
    ],
    "center": TriangularBorder(
        base_a=[287.47034955838177, 235.30709279213306],
        base_b=[334.81540956166964, 235.30709279213306],
        apex=[309.7081807720473, 195.8528761227265],
        guiding_image=guiding_image,
        semantic_label="X",
    ),
}

guiding_image = IMAGE_A / "before_32_phd.png"
exp_id_vs_areas["07.06.2020 (1A)"][32] = {
    "arms": (
        ParallelogramBorder(
            base=[
                [309.44300412905403, 194.07402857692392],
                [288.38603678988756, 236.187963255257],
            ],
            apex=[
                [175.3803120696939, 121.77844071245221],
                [152.9195469079163, 162.48857756817415],
            ],
            guiding_image=guiding_image,
            semantic_label="A",
        ),
        ParallelogramBorder(
            base=[
                [311.8209525619013, 194.7171785444296],
                [336.98296504900634, 234.25748388130893],
            ],
            apex=[
                [438.3499296399152, 117.79331179813704],
                [464.94977141199763, 155.17687320755027],
            ],
            guiding_image=guiding_image,
            semantic_label="B",
        ),
        ParallelogramBorder(
            base=[
                [288.431462397926, 235.8676906087083],
                [335.2359574417894, 235.8676906087083],
            ],
            apex=[
                [292.14610486172467, 396.3402450448116],
                [339.6935283983479, 394.11145956653235],
            ],
            guiding_image=guiding_image,
            semantic_label="C",
        ),
    ),
    "center": TriangularBorder(
        base_a=[287.1835791862539, 234.81489706597512],
        base_b=[335.3210406873076, 234.81489706597512],
        apex=[309.5923974712272, 194.1470416599126],
        guiding_image=guiding_image,
        semantic_label="X",
    ),
}

guiding_image = IMAGE_A / "after_1_phd.png"
exp_id_vs_areas["26.08.2020 (2A)"][1] = {
    "arms": (
        ParallelogramBorder(
            base=[
                [308.19455616608803, 195.86862513168256],
                [283.84982088825086, 237.3978794291695],
            ],
            apex=[
                [174.29851213798355, 120.68635442071474],
                [148.52173360850884, 159.35152221492677],
            ],
            guiding_image=guiding_image,
            semantic_label="A",
        ),
        ParallelogramBorder(
            base=[
                [307.06098087637076, 196.75741594708666],
                [331.03489232273137, 236.46420678012123],
            ],
            apex=[
                [438.9174938313537, 118.84220374641495],
                [462.14222054501545, 159.2981793121483],
            ],
            guiding_image=guiding_image,
            semantic_label="B",
        ),
        ParallelogramBorder(
            base=[
                [281.60649616867585, 236.4269205757078],
                [330.61367602655605, 238.7063242900278],
            ],
            apex=[
                [285.0256017401558, 392.5660750066285],
                [331.75337788371604, 393.7057768637885],
            ],
            guiding_image=guiding_image,
            semantic_label="C",
        ),
    ),
    "center": TriangularBorder(
        base_a=[285.73376623376623, 238.98051948051943],
        base_b=[329.8896103896104, 237.68181818181813],
        apex=[306.51298701298697, 194.82467532467524],
        guiding_image=guiding_image,
        semantic_label="X",
    ),
}

IMAGE_B = IMAGE_ROOT / "B"

guiding_image = IMAGE_B / "before_1.png"
exp_id_vs_areas["31.08.2020 (1B)"][1] = {
    "arms": (
        ParallelogramBorder(
            base=[
                [306.67993702619594, 197.67705743226765],
                [282.74619802583584, 236.4269205757078],
            ],
            apex=[
                [171.0554160241553, 122.4567348597073],
                [147.1216770237952, 163.48600171746745],
            ],
            guiding_image=guiding_image,
            semantic_label="A",
        ),
        ParallelogramBorder(
            base=[
                [305.54023516903595, 196.5373555751076],
                [329.47397416939606, 239.84602614718784],
            ],
            apex=[
                [432.04714131379654, 121.3170330025473],
                [459.3999858856367, 158.92719428882742],
            ],
            guiding_image=guiding_image,
            semantic_label="B",
        ),
        ParallelogramBorder(
            base=[
                [281.60649616867585, 237.5666224328678],
                [331.75337788371604, 237.5666224328678],
            ],
            apex=[
                [285.0256017401558, 400.5439880067486],
                [331.75337788371604, 398.26458429242854],
            ],
            guiding_image=guiding_image,
            semantic_label="C",
        ),
    ),
    "center": TriangularBorder(
        base_a=[282.41830190781593, 237.8687618387333],
        base_b=[329.9589404875393, 237.8687618387333],
        apex=[305.0001052331845, 198.64773501046147],
        guiding_image=guiding_image,
        semantic_label="X",
    ),
}

guiding_image = IMAGE_B / "before_24.png"
exp_id_vs_areas["31.08.2020 (1B)"][23] = {
    "arms": (
        ParallelogramBorder(
            base=[
                [308.4602835740226, 194.5632485451174],
                [286.0356445542109, 234.19749425455205],
            ],
            apex=[
                [174.95545592118992, 119.9882862234179],
                [151.4878104353404, 160.14403516587146],
            ],
            guiding_image=guiding_image,
            semantic_label="A",
        ),
        ParallelogramBorder(
            base=[
                [308.98178680704143, 194.5632485451174],
                [333.49243875892876, 233.6759910215332],
            ],
            apex=[
                [438.3145885957231, 117.9022732913424],
                [461.7822340815726, 155.97200930172045],
            ],
            guiding_image=guiding_image,
            semantic_label="B",
        ),
        ParallelogramBorder(
            base=[
                [286.0356445542109, 234.71899748757096],
                [334.0139419919476, 235.24050072058986],
            ],
            apex=[
                [289.1646639523241, 394.82049002436634],
                [335.57845169100426, 394.2989867913475],
            ],
            guiding_image=guiding_image,
            semantic_label="C",
        ),
    ),
    "center": TriangularBorder(
        base_a=[285.38959181904863, 235.4917299097471],
        base_b=[332.930230398772, 234.89747192750053],
        apex=[308.56565312666373, 194.48792913473568],
        guiding_image=guiding_image,
        semantic_label="X",
    ),
}

guiding_image = IMAGE_B / "after_1.png"
exp_id_vs_areas["25.11.2020 (2B)"][1] = {
    "arms": (
        ParallelogramBorder(
            base=[
                [305.85276740892823, 197.17076471021176],
                [281.3421154570409, 238.89102335172197],
            ],
            apex=[
                [168.6974171249634, 119.9882862234179],
                [145.2297716391139, 160.66553839889036],
            ],
            guiding_image=guiding_image,
            semantic_label="A",
        ),
        ParallelogramBorder(
            base=[
                [304.2882577098716, 197.69226794323066],
                [327.7559031957211, 238.36952011870312],
            ],
            apex=[
                [436.7500788966664, 118.42377652436124],
                [461.7822340815726, 158.57952546681486],
            ],
            guiding_image=guiding_image,
            semantic_label="B",
        ),
        ParallelogramBorder(
            base=[
                [280.82061222402206, 238.89102335172197],
                [329.32041289477763, 238.36952011870312],
            ],
            apex=[
                [284.4711348551542, 393.77748355832864],
                [331.40642582685325, 393.25598032530974],
            ],
            guiding_image=guiding_image,
            semantic_label="C",
        ),
    ),
    "center": TriangularBorder(
        base_a=[281.2297859433228, 239.05727780322633],
        base_b=[328.77042452304624, 237.8687618387333],
        apex=[304.405847250938, 198.0534770282149],
        guiding_image=guiding_image,
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
        exp_id = int(EXP_ID_FINDER.findall(Path(file_path).stem)[0])
        exp_id_vs_dlc_path[exp_id] = file_path

    for file_path in glob(str(DATA_DIR / subdir / "*.mp4")):
        exp_id = int(EXP_ID_FINDER.findall(Path(file_path).stem)[0])

        _, _x, _y, fps = get_video_data(file_path)
        exp_id_vs_fps[exp_id] = fps

    trial_datas.append(
        (
            trial := YMazeTrial(
                exp_id_range_vs_area_sets=exp_id_range_vs_area_sets,
                feature_tracking_point="mid-mid-left_ear-right_ear-base_tail",
                exp_id_vs_coordinate_data_path=exp_id_vs_dlc_path,
                fps=exp_id_vs_fps,
                center_triangle_meter_width=0.08,
                semantic_label=subdir,
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

with pd.ExcelWriter(RESULT_PATH / "phd.xlsx") as writer:
    for trial in trial_datas:
        print(f"Analysing {trial.semantic_label}")
        df = trial.export_to_dataframe()
        df.to_excel(writer, sheet_name=trial.semantic_label)
