from pathlib import Path
from glob import glob
import pandas as pd
import re
import os

from bikipy.behaviour.y_maze import reduce_location_sequence, spontaneous_alterntations
from bikipy.border.parallelogram.classes import ParallelogramBorder
from bikipy.border.triangular import TriangularBorder
from bikipy.readers import DeepLabCutReader


WORKING_DIR = Path("C:/Users/Can/Projects/Neuroscience/bikipy/examples/data/")
DATA_DIR = Path("C:/Users/Can/Projects/Neuroscience/Imen/data/y_maze")

BORDER_IMG_PATH = WORKING_DIR / "images" / "y_maze" / "master's.png"
assert BORDER_IMG_PATH.exists(), f"The image file doesn't exist in {BORDER_IMG_PATH}"
border_img_path_str = str(BORDER_IMG_PATH)
borders = [
    ParallelogramBorder(
        base=[[281.46403344, 180.58152957], [257.53029444, 222.75049829]],
        apex=[[145.83951243, 103.08180329], [121.90577343, 140.69196457]],
        guiding_image=border_img_path_str,
        label="A",
    ),
    ParallelogramBorder(
        base=[[280.32433158, 181.72123143], [306.53747429, 223.89020015]],
        apex=[[420.50766001, 100.80239957], [444.44139901, 144.11107015]],
        guiding_image=border_img_path_str,
        label="B",
    ),
    ParallelogramBorder(
        base=[[256.39059258, 222.75049829], [306.53747429, 223.89020015]],
        apex=[[255.25089072, 384.58816201], [305.39777244, 384.58816201]],
        guiding_image=border_img_path_str,
        label="C",
    ),
]
inferior = [
    TriangularBorder(
        base_a=(257.1623376623377, 223.39610389610385),
        base_b=(280.538961038961, 183.13636363636357),
        apex=(302.6168831168832, 223.39610389610385),
        label="X",
    )
]

exp_id_finder = re.compile("\d+")

DATA_DIR = Path("C:/Users/Can/Projects/Neuroscience/Imen/data/y_maze/master's")
exp_id_finder = re.compile("\d+")
data_dict, dlc_data = {}, {}
for subdir in os.listdir(str(DATA_DIR)):
    print(subdir)
    for file_path in glob(os.path.join(str(DATA_DIR / subdir), "**.h5")):
        exp_id = exp_id_finder.findall(Path(file_path).stem)[0]
        print(exp_id)
        dlc_data[(subdir, exp_id)] = DeepLabCutReader.from_hdf(
            file_path, (640, 480), midpoint_groups=(("left_ear", "right_ear"),)
        )
        data_dict[
            (subdir, exp_id)
        ] = ParallelogramBorder.detect_sequential_border_presence(
            dlc_data[(subdir, exp_id)]["mid-left_ear-right_ear"],
            borders,
            overlap_inferior=inferior,
        )

reduced = {}
for info, location_per_frame in data_dict.items():
    reduced[info] = reduce_location_sequence(location_per_frame)

exp_info_vs_spontaneous_alterntations = {}
for info, arm_location_sequence in reduced.items():
    exp_info_vs_spontaneous_alterntations[info] = spontaneous_alterntations(
        arm_location_sequence, exclude="X"
    )

print(reduced)
