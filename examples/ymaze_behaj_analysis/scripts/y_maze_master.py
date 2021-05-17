import os
import re
from glob import glob
from pathlib import Path

import pandas as pd

from bikipy.behaviour.y_maze.trial import YMazeTrial
from bikipy.border.parallelogram.classes import ParallelogramBorder
from bikipy.border.triangular import TriangularBorder
from bikipy.utils.video import get_video_data

DATA_DIR = Path("C:/Users/Can/Projects/Neuroscience/Imen/data/results/master's")
EXP_ID_REGEX_PATTERN = re.compile("\d+")

REPO_PATH = Path("C:/Users/Can/Projects/Neuroscience/bikipy")
IMAGE_ROOT = REPO_PATH / "examples/data/images/results/master's"
RESULT_PATH = REPO_PATH / "examples/data/results/results"

exp_id_vs_areas = {"before": {}, "after": {}}

guiding_image = IMAGE_ROOT / "before_11_masters.png"
exp_id_vs_areas["before"][11] = {
    "arms": [
        ParallelogramBorder(
            base=[
                [281.5591320745524, 177.520856037089],
                [256.02084967332576, 222.54888027083075],
            ],
            apex=[
                [146.47505937332718, 99.56188870702874],
                [120.93677697210052, 144.58991294077043],
            ],
            guiding_image=guiding_image,
            semantic_label="A",
        ),
        ParallelogramBorder(
            base=[
                [280.27697663670625, 180.21832788412422],
                [305.0748851771108, 222.29962722541677],
            ],
            apex=[
                [417.7926512698586, 103.57024694105576],
                [440.33620448840816, 143.39719096049328],
            ],
            guiding_image=guiding_image,
            semantic_label="B",
        ),
        ParallelogramBorder(
            base=[
                [257.14314435137806, 222.75049828978769],
                [305.53663859386444, 222.75049828978769],
            ],
            apex=[
                [255.74043437333503, 388.27027569887156],
                [304.8352836048429, 388.97163068789314],
            ],
            guiding_image=guiding_image,
            semantic_label="C",
        ),
    ],
    "center": TriangularBorder(
        base_a=[255.97642679900744, 223.20223325062028],
        base_b=[304.4131513647642, 223.99627791563267],
        apex=[281.38585607940445, 182.70595533498755],
        guiding_image=guiding_image,
        semantic_label="X",
    ),
}


guiding_image = IMAGE_ROOT / "before_45_masters.png"
exp_id_vs_areas["before"][45] = {
    "arms": [
        ParallelogramBorder(
            base=[
                [281.60649616867585, 178.30212586054756],
                [255.3933534539957, 220.47109457546776],
            ],
            apex=[
                [145.9819751666352, 99.66269771650718],
                [122.04823616627507, 138.41256085994735],
            ],
            guiding_image=guiding_image,
            semantic_label="A",
        ),
        ParallelogramBorder(
            base=[
                [282.74619802583584, 179.44182771770755],
                [304.4005333118759, 221.61079643262775],
            ],
            apex=[
                [417.2310171707164, 101.94210143082717],
                [443.4441598853966, 141.83166643142738],
            ],
            guiding_image=guiding_image,
            semantic_label="B",
        ),
        ParallelogramBorder(
            base=[
                [256.5330553111557, 221.61079643262775],
                [305.54023516903595, 219.3313927183077],
            ],
            apex=[
                [257.67275716831574, 382.3087582921885],
                [302.1211295975559, 383.4484601493485],
            ],
            guiding_image=guiding_image,
            semantic_label="C",
        ),
    ],
    "center": TriangularBorder(
        base_a=[257.0927345638969, 220.05882526390275],
        base_b=[304.6483163295394, 220.05882526390275],
        apex=[281.24795069882646, 179.2968980362092],
        guiding_image=guiding_image,
        semantic_label="X",
    ),
}


guiding_image = IMAGE_ROOT / "before_63_masters.png"
exp_id_vs_areas["before"][63] = {
    "arms": [
        ParallelogramBorder(
            base=[
                [281.8208727289753, 179.51535686978235],
                [257.3865698856442, 221.59665621107484],
            ],
            apex=[
                [148.11093772519104, 100.78260326349317],
                [122.99790424732295, 141.5064413357117],
            ],
            guiding_image=guiding_image,
            semantic_label="A",
        ),
        ParallelogramBorder(
            base=[
                [282.94480171728196, 178.2204407445053],
                [307.21695048410254, 221.16347317811102],
            ],
            apex=[
                [416.13043854034885, 102.29218049957939],
                [442.8920384627408, 141.50103619982804],
            ],
            guiding_image=guiding_image,
            semantic_label="B",
        ),
        ParallelogramBorder(
            base=[
                [257.9808002542238, 221.87682020764055],
                [307.0411848671066, 222.54888027083075],
            ],
            apex=[
                [257.9808002542238, 385.85947562604326],
                [306.3691248039164, 387.2035957524236],
            ],
            guiding_image=guiding_image,
            semantic_label="C",
        ),
    ],
    "center": TriangularBorder(
        base_a=[258.9542786114356, 222.3857938718662],
        base_b=[306.4937400784737, 223.15255937939912],
        apex=[282.72400934495465, 179.4469254500254],
        guiding_image=guiding_image,
        semantic_label="X",
    ),
}


guiding_image = IMAGE_ROOT / "after_94_masters.png"
exp_id_vs_areas["after"][94] = {
    "arms": [
        ParallelogramBorder(
            base=[
                [284.1194300560733, 177.64115318443794],
                [259.9653668303434, 220.72677947898325],
            ],
            apex=[
                [148.3344259762942, 98.6508383111049],
                [126.13880030940723, 143.04208964487884],
            ],
            guiding_image=guiding_image,
            semantic_label="A",
        ),
        ParallelogramBorder(
            base=[
                [282.893905862418, 179.0045754728493],
                [307.11095326994956, 221.8501208861744],
            ],
            apex=[
                [420.74479110528995, 102.0067837155695],
                [444.34088857929504, 143.61042926184166],
            ],
            guiding_image=guiding_image,
            semantic_label="B",
        ),
        ParallelogramBorder(
            base=[
                [257.7247075137833, 220.93680465623117],
                [305.81426805863657, 222.31079210036984],
            ],
            apex=[
                [257.7247075137833, 383.7543167866629],
                [305.1272743365672, 383.7543167866629],
            ],
            guiding_image=guiding_image,
            semantic_label="C",
        ),
    ],
    "center": TriangularBorder(
        base_a=[255.86363636363635, 222.09740259740255],
        base_b=[306.51298701298697, 223.39610389610385],
        apex=[281.8376623376623, 179.24025974025966],
        guiding_image=guiding_image,
        semantic_label="X",
    ),
}


guiding_image = IMAGE_ROOT / "after_140_masters.png"
exp_id_vs_areas["after"][140] = {
    "arms": [
        ParallelogramBorder(
            base=[
                [285.0670576521427, 175.75754558758268],
                [260.2837591284915, 218.33398151282955],
            ],
            apex=[
                [151.6185271401749, 98.23030405205856],
                [126.83522861652372, 140.17127078439125],
            ],
            guiding_image=guiding_image,
            semantic_label="A",
        ),
        ParallelogramBorder(
            base=[
                [283.7985775610646, 177.70187067917033],
                [307.88986611254813, 220.3778675417983],
            ],
            apex=[
                [418.02147091933, 101.29806984446537],
                [443.48940453089836, 141.22077658692382],
            ],
            guiding_image=guiding_image,
            semantic_label="B",
        ),
        ParallelogramBorder(
            base=[
                [258.47683914420224, 219.85281626878952],
                [305.60397530988723, 221.12652265164587],
            ],
            apex=[
                [259.11369233563045, 384.1609396572587],
                [304.96712211845903, 383.5240864658306],
            ],
            guiding_image=guiding_image,
            semantic_label="C",
        ),
    ],
    "center": TriangularBorder(
        base_a=[257.42688172043006, 219.35766308243723],
        base_b=[307.34229390681, 220.0917132616487],
        apex=[283.1186379928315, 178.25085304659495],
        guiding_image=guiding_image,
        semantic_label="X",
    ),
}

trial_datas = []
for subdir in os.listdir(str(DATA_DIR)):
    print(f"Reading {subdir}")

    trial_name = subdir.split("_")[1].lower()
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
                exp_id_vs_coordinate_data_path=exp_id_vs_dlc_path,
                fps=exp_id_vs_fps,
                center_triangle_meter_width=0.08,
                label=subdir,
                midpoint_groups=(
                    ("left_ear", "right_ear"),
                    ("mid-left_ear-right_ear", "base_tail"),
                ),
                x_crop_start=95.0,
                y_crop_start=75.0,
            )
        )
    )

    #  trial.plot(invalid=False)

with pd.ExcelWriter(RESULT_PATH / "master's.ods") as writer:
    for trial in trial_datas:
        print(f"Analysing {trial.label}")
        df = trial.export_to_dataframe()
        df.to_excel(writer, sheet_name=trial.label)
