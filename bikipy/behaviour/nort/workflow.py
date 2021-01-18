import os
import pickle
import re
from glob import glob
from pathlib import Path
from typing import Dict, Any, AnyStr

import numpy as np
import pandas as pd

from bikipy.behaviour.nort.trial import NortTrial
from bikipy.utils.video import get_video_data
from bikipy.utils.misc import resolve_stem_in_filepath


def deeplabcut_workflow(app_to_obj: Dict, t1_dir: Any, t2_dir: Any, t1_meta: Dict, t2_meta: Dict, exp_id_finder: AnyStr,
             save_dir: Any = None):
    def get_exp():
        return int(exp_id_finder.findall(Path(data_path).stem)[0])

    outfile_path = Path(save_dir) / "nort.xlsx"
    save_dir = resolve_stem_in_filepath(outfile_path)

    result = []
    for root, meta, period in zip((t1_dir, t2_dir), (t1_meta, t2_meta), ("t1", "t2")):
        exp_ids_range_vs_exp_meta, exp_id_vs_coordinate_data_path = {}, {}
        for time_dir in os.listdir(root):
            for data_path in glob(str(root / time_dir / "*.h5")):
                exp_id = get_exp()
                exp_id_vs_coordinate_data_path[exp_id] = data_path

            for data_path in glob(str(root / time_dir / "*.mp4")):
                exp_id = get_exp()
                frame, x, y, fps = get_video_data(data_path)

                exp_ids_range_vs_exp_meta[exp_id] = {
                    **meta[exp_id],
                    "recording_resolution": (x, y),
                    "fps": fps,
                    "guiding_image": frame,
                }

        result.append(
            NortTrial(
                exp_ids_range_vs_exp_meta=exp_ids_range_vs_exp_meta,
                nose_label="nose",
                eye_center_label="mid-left_ear-right_ear",
                torso_label="mid-mid-left_ear-right_ear-tail",
                experiment_box_real_length=0.4,
                center_size_real_length=0.2,
                max_radians_gaze_and_object=1 / 4 * np.pi,
                exp_id_vs_coordinate_data_path=exp_id_vs_coordinate_data_path,
                nort_fields=app_to_obj,
                midpoint_groups=[
                    ("left_ear", "right_ear"),
                    ("mid-left_ear-right_ear", "tail"),
                ],
                label=period
            )
        )

    if save_dir:
        with pd.ExcelWriter() as writer:
            for trial in result:
                trial.export_to_dataframe().to_excel(
                    writer, sheet_name=str(trial.label)
                )

    return result
