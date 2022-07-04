import datetime
import os
import re
from glob import glob
from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd

from bikipy.behaviour.object_recognition.base import ObjectField
from bikipy.behaviour.object_recognition.novel_object_recognition import (
    EXPERIMENT_STAGE_to_TRIAL_CLASS_NAME,
    NortExperiment,
)
from bikipy.perimeter.polygon.base import PolygonPerimeter
from bikipy.perimeter.polygon.makesense import init_polygon_from_makesense_coco_polygon
from bikipy.plugins.belhaj import (
    get_animal_id_to_apparatus,
    get_animal_id_to_trial_ids,
    get_trial_id_to_animal_id,
    get_trial_id_to_stage,
)
from bikipy.utils.io.general import defer_perimeter_set_from_multi_row_reference
from bikipy.utils.io.makesense import read_makesense_point

DEEPLABCUT_DIR = Path("/mnt/soma/Projects/Neuroscience/Imen/data/nort")


WORKING_DIR = Path("").resolve()
DATA_DIR = WORKING_DIR / "data"
IMAGE_DIR = DATA_DIR / "area_images"

RESULT_DIR = WORKING_DIR / "results"
if not RESULT_DIR.exists():
    os.mkdir(RESULT_DIR)

inspect_directory = WORKING_DIR / "inspect"
if inspect_directory.exists():
    rmtree(inspect_directory)
os.mkdir(inspect_directory)

EXP_ID_REGEX_PATTERN = re.compile(r"\d+")

period_to_field_id_to_object_field = {}
for field_idx in range(1, 5):
    training_perimeters = init_polygon_from_makesense_coco_polygon(
        IMAGE_DIR / f"training_{field_idx}.json",
        image_root=IMAGE_DIR,
    )
    nort_training_objects = defer_perimeter_set_from_multi_row_reference(
        reference_perimeters=training_perimeters.values(),
        image_name_to_reference_data=read_makesense_point(
            IMAGE_DIR / f"references_training_{field_idx}.csv", single_row=False
        ),
        image_root=IMAGE_DIR,
    )

    novel_perimeters = init_polygon_from_makesense_coco_polygon(
        IMAGE_DIR / f"novel_{field_idx}.json",
        image_root=IMAGE_DIR,
    )
    nort_novelty_objects = defer_perimeter_set_from_multi_row_reference(
        reference_perimeters=novel_perimeters.values(),
        image_name_to_reference_data=read_makesense_point(
            IMAGE_DIR / f"references_novel_{field_idx}.csv", single_row=False
        ),
        image_root=IMAGE_DIR,
    )
    for experiment_period, novel in nort_novelty_objects.items():
        training = nort_training_objects[experiment_period.replace("novel", "training")]
        period, _, field = experiment_period.split(".")[0].split("_")
        if period not in period_to_field_id_to_object_field:
            period_to_field_id_to_object_field[period] = {}

        period_to_field_id_to_object_field[period][int(field)] = ObjectField.nort_format(
            constant_object_perimeter=training["constant"],
            variable_object_perimeter=training["variable"],
            novel_object_perimeter=novel["novel"],
            novelty_constant_object_perimeter=novel["constant"],
        )
    # period_to_field_id_to_object_field[field_idx].plot()

experiments = []
for period_index, period_letter in enumerate(("A", "B"), start=1):
    period_name = f"{period_letter}{period_index}"
    experiment_root_data_path = DEEPLABCUT_DIR / f"Experiment_{period_index}"
    meta_data = DATA_DIR / f"nort_round_{period_index}.xlsx"
    for round_part_directory_name in os.listdir(experiment_root_data_path):
        round_directory_path = experiment_root_data_path / round_part_directory_name
        round_index = int(round_part_directory_name.split("_")[0][-1]) - 1

        day, month, year = round_directory_path.name.split("_")[1].split(".")
        date = datetime.date(int(year), int(month), int(day))

        exp_metadata_df = pd.read_excel(meta_data, sheet_name=round_index, engine="openpyxl")

        animal_id_to_app = get_animal_id_to_apparatus(exp_metadata_df, EXP_ID_REGEX_PATTERN)
        trial_id_to_stage = get_trial_id_to_stage(exp_metadata_df, EXP_ID_REGEX_PATTERN)
        animal_id_to_trial_ids = get_animal_id_to_trial_ids(exp_metadata_df)
        exp_to_animal = get_trial_id_to_animal_id(animal_id_to_trial_ids)

        trial_id_to_paths = {}
        for video_path in glob(str(round_directory_path / "**" / "*.mp4")):
            trial_id = int(EXP_ID_REGEX_PATTERN.findall(Path(video_path).stem)[0])
            trial_id_to_paths[trial_id] = {"video": video_path}
        for data_path in glob(str(round_directory_path / "**" / "*.parquet")):
            trial_id = int(EXP_ID_REGEX_PATTERN.findall(Path(data_path).stem)[0])
            trial_id_to_paths[trial_id]["data"] = data_path

        trial_id_range_to_exp_meta, trial_id_to_trial_class_name = {}, {}
        for trial_id, paths in trial_id_to_paths.items():
            trial_id_range_to_exp_meta[trial_id] = {
                "coordinate_data_path": paths["data"],
                "video_path": paths["video"],
                "stage": (stage := trial_id_to_stage[trial_id]),
                "animal_id": (animal_id := exp_to_animal[trial_id]),
                "field_id": animal_id_to_app[animal_id],
            }
            trial_id_to_trial_class_name[trial_id] = NortExperiment.trial_class_name_to_trial_class[
                EXPERIMENT_STAGE_to_TRIAL_CLASS_NAME[stage]
            ]

        experiment = NortExperiment(
            stage=str(period_index),
            trial_id_to_trial_class_name=trial_id_to_trial_class_name,
            trial_id_to_keyword_arguments=trial_id_range_to_exp_meta,
            gaze_travel_direction_point_label="nose",
            gaze_start_point_label="center_eye",
            object_tracking_label_for_kinematics="torso",
            id_to_object_field=period_to_field_id_to_object_field[period_name],
            perimeter_border_normal_meters=0.03,
            global_center_length_meters=0.2,
            maximum_radians_inter_gaze_perimeter=np.deg2rad(75.0),
            minimum_seconds_attention=2.0,
            maximum_seconds_distraction=0.5,
            # inspect_directory=inspect_directory,
            reader_kwargs={
                "init_from": "parquet",
                "midpoint_groups": {
                    "center_eye": ("left_ear", "right_ear"),
                    "torso": ("center_eye", "tail"),
                },
            },
            timestamp=date,
        )

        experiments.append(experiment)


with pd.ExcelWriter(
    RESULT_DIR / "nort_analysis.xlsx",
) as writer:
    for experiment in experiments:
        experiment.full_experiment_df.to_parquet(RESULT_DIR / "for_analysis" / f"{experiment.timestamp}.parquet")
        experiment.full_experiment_df.to_excel(writer, sheet_name=f"{experiment.timestamp}")
