import datetime
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from bikipy.behaviour.nort.experiment import NortExperiment
from bikipy.behaviour.nort.trial import NortField
from bikipy.perimeter.base import PolygonalPerimeter
from bikipy.plugins.belhaj import (
    get_animal_id_vs_apparatus,
    get_animal_id_vs_trial_ids,
    get_trial_id_vs_animal_id,
    get_trial_id_vs_stage,
)

DEEPLABCUT_DIR = Path("/mnt/md0/Projects/Neuroscience/Imen/data/nort")

WORKING_DIR = Path(".").resolve()
DATA_DIR = WORKING_DIR / "data"
IMAGE_DIR = DATA_DIR / "area_images"
RESULT_DIR = WORKING_DIR / "results"
if not RESULT_DIR.exists():
    os.mkdir(RESULT_DIR)

EXP_ID_REGEX_PATTERN = re.compile(r"\d+")

TO_SKIP = ([], [103])

nort_field_vs_nort_field_object = {}
for i, list_idx in zip(range(1, 5), range(4)):
    novel = PolygonalPerimeter.from_coco(
        IMAGE_DIR / f"novel_{i}.json",
        IMAGE_DIR / f"references_novel_{i}.csv",
        image_root=IMAGE_DIR,
    )
    training = PolygonalPerimeter.from_coco(
        IMAGE_DIR / f"training_{i}.json",
        IMAGE_DIR / f"references_training_{i}.csv",
        image_root=IMAGE_DIR,
    )
    nort_field_vs_nort_field_object[i] = NortField(
        label=i,
        constant_object_perimeter=training["constant"][list_idx],
        variable_object_perimeter=training["variable"][list_idx],
        novel_object_perimeter=novel["novel"][list_idx],
        novelty_constant_object_perimeter=novel["constant"][list_idx],
    )
    # nort_field_vs_nort_field_object[i].plot()

experiments = []
for round_idx in range(2):
    round_number = round_idx + 1
    experiment_root_data_path = DEEPLABCUT_DIR / f"Experiment_{round_number}"
    meta_data = DATA_DIR / f"nort_round_{round_number}.xlsx"
    for round_part_idx, round_part_dir_name in enumerate(
        os.listdir(experiment_root_data_path)
    ):
        round_dir_path = experiment_root_data_path / round_part_dir_name

        day, month, year = round_dir_path.name.split("_")[1].split(".")
        date = datetime.date(int(year), int(month), int(day))

        exp_metadata_df = pd.read_excel(
            meta_data, sheet_name=round_part_idx, engine="openpyxl"
        )

        animal_id_vs_app = get_animal_id_vs_apparatus(
            exp_metadata_df, EXP_ID_REGEX_PATTERN
        )
        trial_id_vs_stage = get_trial_id_vs_stage(exp_metadata_df, EXP_ID_REGEX_PATTERN)
        animal_id_vs_trial_ids = get_animal_id_vs_trial_ids(exp_metadata_df)
        exp_vs_animal = get_trial_id_vs_animal_id(animal_id_vs_trial_ids)

        trial_id_vs_paths = {}
        for video_path in glob(str(round_dir_path / "**" / "*.mp4")):
            trial_id = int(EXP_ID_REGEX_PATTERN.findall(Path(video_path).stem)[0])
            trial_id_vs_paths[trial_id] = {"video": video_path}
        for data_path in glob(str(round_dir_path / "**" / "*.parquet")):
            trial_id = int(EXP_ID_REGEX_PATTERN.findall(Path(data_path).stem)[0])
            trial_id_vs_paths[trial_id]["data"] = data_path

        trial_id_range_vs_exp_meta = {}
        for trial_id, paths in trial_id_vs_paths.items():
            if trial_id in TO_SKIP[round_idx]:
                continue

            trial_data = {
                "coordinate_data_path": paths["data"],
                "stage": (stage := trial_id_vs_stage[trial_id]),
                "video_path": paths["video"],
                "animal_id": (animal_id := exp_vs_animal[trial_id]),
                "inspect": False,
            }

            if stage != "habituation":
                trial_data["field"] = animal_id_vs_app[animal_id]

            trial_id_range_vs_exp_meta[trial_id] = trial_data

        experiment = NortExperiment(
            trial_id_vs_data=trial_id_range_vs_exp_meta,
            metric_resolution=0.4,
            nose_label="nose",
            center_eye_label="mid-left_ear-right_ear",
            torso_label="mid-mid-left_ear-right_ear-tail",
            nort_field_vs_nort_field_object=nort_field_vs_nort_field_object,
            perimeter_border_normal_metric_magnitude=0.06,
            center_metric_length=0.2,
            maximum_radians_inter_gaze_perimeter=0.33 * np.pi,
            inspection_figure_save=RESULT_DIR / "inspect",
            data_import_kwargs={
                "init_from": "parquet",
                "midpoint_groups": (
                    ("left_ear", "right_ear"),
                    ("mid-left_ear-right_ear", "tail"),
                ),
            },
            timestamp=date,
        )

        experiment.plot_attention_state_distribution()
        experiments.append(experiment)


with pd.ExcelWriter(
    RESULT_DIR / "nort_analysis.ods",
    engine_kwargs={
        "strings_to_formulas": False,
        "strings_to_urls": False,
    },
) as writer:
    for experiment in experiments:
        experiment.df.to_parquet(RESULT_DIR / f"exp_{experiment.timestamp}.parquet")
        experiment.df.to_excel(writer, sheet_name=str(experiment.timestamp))
