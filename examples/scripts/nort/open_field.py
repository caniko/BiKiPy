import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from bikipy.behaviour.nort.trial import NortTrial
from bikipy.utils.video import get_video_data

WORKING_DIR = Path("C:/Users/Can/Projects/Neuroscience/Imen/data")
NORT_DIR = WORKING_DIR / "nort"

OPEN_FIELD_DIR = NORT_DIR / "Open-Field"
OF_1C_DIR = OPEN_FIELD_DIR / "NORT_02.06.2020 (1C)"
OF_2C_DIR = OPEN_FIELD_DIR / "NORT (after)_24.08.2020 (2C)"
OF_1D_DIR = OPEN_FIELD_DIR / "NORT2_30.08.2020 (1D)"
OF_2D_DIR = OPEN_FIELD_DIR / "NORT2 (after)_23.11.2020 (2D)"


EXP_ID_FINDER = re.compile("\d+")


exp_ids_range_vs_exp_meta = {"1C": {}, "2C": {}, "1D": {}, "2D": {}}
exp_id_vs_coordinate_data_path = {"1C": {}, "2C": {}, "1D": {}, "2D": {}}
exp_category = "open field"
for time, root in zip(
    exp_ids_range_vs_exp_meta,
    (OF_1C_DIR, OF_2C_DIR, OF_1D_DIR, OF_2D_DIR),
):
    for data_path in glob(str(root / "*.h5")):
        exp_id = int(EXP_ID_FINDER.findall(Path(data_path).stem)[0])

        exp_id_vs_coordinate_data_path[time][exp_id] = data_path

    for data_path in glob(str(root / "*.mp4")):
        exp_id = int(EXP_ID_FINDER.findall(Path(data_path).stem)[0])

        _, x, y, fps = get_video_data(data_path)

        exp_ids_range_vs_exp_meta[time][exp_id] = {
            "exp_category": exp_category,
            "recording_resolution": (x, y),
            "fps": fps,
        }

result = []
for time in exp_ids_range_vs_exp_meta:
    result.append(
        NortTrial(
            exp_ids_range_vs_exp_meta=exp_ids_range_vs_exp_meta[time],
            experiment_box_real_length=0.4,
            eye_center_label="mid-left_ear-right_ear",
            exp_id_vs_coordinate_data_path=exp_id_vs_coordinate_data_path[time],
            midpoint_groups=[("left_ear", "right_ear")],
            label=time,
        )
    )

with pd.ExcelWriter(NORT_DIR / "open_field.ods") as writer:
    for trial in result:
        trial.export_to_dataframe()["Open-Field"].to_excel(
            writer, sheet_name=trial.label
        )
