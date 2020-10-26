from pathlib import Path
from glob import glob
import re
import os

import numpy as np

from bikipy.border.parallelogram.classes import ParallelogramBorder
from bikipy.border.triangular import TriangularBorder
from bikipy.behaviour.y_maze.experiment import YMaze
from bikipy.readers import DeepLabCutReader


exp_id_finder = re.compile("\d+")
WORKING_DIR = Path("C:/Users/Can/Projects/Neuroscience/bikipy/examples/data/")
DATA_DIR = Path("C:/Users/Can/Projects/Neuroscience/Imen/data/y_maze/master's")


BORDER_IMG_PATH = WORKING_DIR / "images" / "y_maze" / "master's.png"
assert BORDER_IMG_PATH.exists(), f"The image file doesn't exist in {BORDER_IMG_PATH}"
border_img_path_str = str(BORDER_IMG_PATH)

borders = (
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
)
center = TriangularBorder(
    base_a=(257.1623376623377, 223.39610389610385),
    base_b=(280.538961038961, 183.13636363636357),
    apex=(302.6168831168832, 223.39610389610385),
    label="X",
)
cm_pr_pixel = np.linalg.norm(center.base_a - center.base_b) / 5

data_dict = {}
for subdir in os.listdir(str(DATA_DIR)):
    data_dict[subdir] = {}
    for file_path in glob(os.path.join(str(DATA_DIR / subdir), "**.h5")):
        exp_id = exp_id_finder.findall(Path(file_path).stem)[0]
        print(exp_id)
        dlc_data = DeepLabCutReader.from_hdf(
            file_path, (640, 480), midpoint_groups=(("left_ear", "right_ear"),)
        )
        data_dict[subdir][exp_id] = YMaze(
            dlc_data["mid-left_ear-right_ear"],
            borders,
            center,
            14,
            cm_pr_pixel,
            exp_id,
        )

YMaze.export_to_dataframe(data_dict["0_before"].values())
