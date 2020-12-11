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

exp_id_vs_areas = {"before": {}, "after": {}}


guiding_image = IMAGE_ROOT / "before_1_phd.png"
exp_id_vs_areas["before"][1] = {
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
            label="A",
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
            label="B",
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
            label="C",
        ),
    ],
    "center": TriangularBorder(
        base_a=[287.47034955838177, 235.30709279213306],
        base_b=[334.81540956166964, 235.30709279213306],
        apex=[309.7081807720473, 195.8528761227265],
        guiding_image=guiding_image,
        label="X",
    ),
}

guiding_image = IMAGE_ROOT / "before_32_phd.png"
exp_id_vs_areas["before"][32] = {
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
            label="A",
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
            label="B",
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
            label="C",
        ),
    ),
    "center": TriangularBorder(
        base_a=[287.1835791862539, 234.81489706597512],
        base_b=[335.3210406873076, 234.81489706597512],
        apex=[309.5923974712272, 194.1470416599126],
        guiding_image=guiding_image,
        label="X",
    ),
}

guiding_image = IMAGE_ROOT / "after_1_phd.png"
exp_id_vs_areas["after"][1] = {
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
            label="A",
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
            label="B",
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
            label="C",
        ),
    ),
    "center": TriangularBorder(
        base_a=[285.73376623376623, 238.98051948051943],
        base_b=[329.8896103896104, 237.68181818181813],
        apex=[306.51298701298697, 194.82467532467524],
        guiding_image=guiding_image,
        label="X",
    ),
}

trial_datas = []
for subdir in os.listdir(str(DATA_DIR)):
    print(f"Reading {subdir}")

    trial_name = subdir.split("_")[1].lower()
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
                feature_tracking_point="mid-left_ear-right_ear",
                exp_id_vs_coordinate_data_path=exp_id_vs_dlc_path,
                fps=exp_id_vs_fps,
                center_triangle_cm_width=8,
                label=subdir,
                midpoint_groups=(("left_ear", "right_ear"),)
            )
        )
    )

    # trial.plot()

with pd.ExcelWriter(RESULT_PATH / "phd.xlsx") as writer:
    for trial in trial_datas:
        print(f"Analysing {trial.label}")
        df = trial.export_to_dataframe()
        df.to_excel(writer, sheet_name=trial.label)
