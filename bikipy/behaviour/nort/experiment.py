from collections.abc import Mapping, Sequence
from functools import cached_property
from logging import getLogger
from typing import SupportsFloat, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from bikipy.behaviour.base import BaseExperiment
from bikipy.behaviour.nort.trial import NortHabituationTrial, NortObjectField
from bikipy.utils.store import sort_dict_by_key_value

logger = getLogger(__name__)


class NortExperiment(BaseExperiment):
    """
    Class for combining several NORT trials under one class for joint analysis
    """

    period_columns = ("T1", "T2", "Total")

    box_area_names = ("Periphery", "Center")

    trial_label_to_trial_class_name = {
        "habituation": "habituation",
        "open_field": "habituation",
        "1": "training",
        "t1": "training",
        "training": "training",
        "2": "novelty",
        "t2": "novelty",
        "test": "novelty",
        "novelty_observation": "novelty",
        "novelty": "novelty",
    }

    def __init__(
        self,
        trial_id_range_vs_exp_meta: dict,
        experiment_box_metric_length: SupportsFloat,
        nose_label: str,
        eye_center_label: str,
        torso_label: str,
        nort_field_vs_apparatus: Mapping[NortObjectField] = None,
        perimeter_border_normal_metric_magnitude: Union[SupportsFloat, None] = None,
        center_size_metric_length: Union[SupportsFloat, None] = None,
        max_radians_gaze_and_object: SupportsFloat = 0.25 * np.pi,
        *base_trial_args,
        **base_trial_kwargs,
    ):
        """

        Parameters
        ----------
        trial_id_range_vs_exp_meta
        experiment_box_metric_length
        nose_label
        eye_center_label
        torso_label
        nort_field_vs_apparatus
        perimeter_border_normal_pixel_magnitude
            The magnitude of the normal between the perimeter and the border given in meters
        center_size_metric_length
        max_radians_gaze_and_object
        base_trial_args
        base_trial_kwargs
        """
        super().__init__(*base_trial_args, **base_trial_kwargs)

        self.nort_field_vs_apparatus = nort_field_vs_apparatus

        self.trial_id_range_vs_exp_meta = dict(trial_id_range_vs_exp_meta)
        self.torso_label, self.eye_center_label, self.nose_label = (
            str(torso_label),
            str(eye_center_label),
            str(nose_label),
        )
        self.experiment_box_metric_length, self.max_radians_gaze_and_object = (
            float(experiment_box_metric_length),
            float(max_radians_gaze_and_object),
        )

        self.center_size_metric_length = (
            float(center_size_metric_length) if center_size_metric_length else None
        )
        self.perimeter_border_normal_metric_magnitude = (
            perimeter_border_normal_metric_magnitude
        )

        self.experiment_pairs = {}
        (
            self.habituation_trials,
            self.training_object_trials,
            self.novelty_object_trials,
        ) = ([], [], [])
        for exp_id, exp_meta in tqdm(self.trial_id_range_vs_exp_meta.items()):
            logger.info(f"Category {exp_meta['stage']}; ID {exp_id}")

            coordinate_sequence = self.exp_id_vs_coordinate_sequences[exp_id]

            generic_data = {
                "recording_resolution": exp_meta["recording_resolution"],
                "experiment_box_metric_length": experiment_box_metric_length,
                "label": exp_id,
                "func_inspect": self.func_inspect,
            }

            if "inspect" in exp_meta:
                generic_data["func_inspect"] = exp_meta["inspect"]
            if "inspect_image" in exp_meta:
                generic_data["inspect_image"] = exp_meta["inspect_image"]

            if "fps" in exp_meta:
                generic_data["fps"] = exp_meta["fps"]
            elif hasattr(coordinate_sequence, "fps"):
                generic_data["fps"] = coordinate_sequence.fps
            elif isinstance(self.fps, dict):
                generic_data["fps"] = self.fps[exp_id]
            elif self.fps:  # Fallback FPS value
                generic_data["fps"] = self.fps
            else:
                msg = "fps has to be defined"
                raise AttributeError(msg)

            exp_class = self.trial_label_to_trial_class_name[
                exp_meta["stage"].lower().replace(" ", "_")
            ]

            if exp_class == "habituation":
                self.habituation_trials.append(
                    (
                        exp := NortHabituationTrial(
                            coordinate_sequence=coordinate_sequence[
                                self.eye_center_label
                            ],
                            center_size_metric_length=self.center_size_metric_length,
                            **generic_data,
                        )
                    )
                )

            elif exp_class == "training" or exp_class == "novelty":
                try:
                    field = self.nort_field_vs_apparatus[exp_meta["field"] - 1]
                except AttributeError as e:
                    msg = "nort_field_vs_apparatus is not defined, which is required when working with training and/or novelty datasets"
                    raise AttributeError(msg) from e

                analysis_keyword_arguments = {
                    "nose_label": self.nose_label,
                    "eye_center_label": self.eye_center_label,
                    "torso_label": self.torso_label,
                    "perimeter_border_normal_metric_magnitude": self.perimeter_border_normal_metric_magnitude,
                    "max_radians_gaze_and_object": self.max_radians_gaze_and_object,
                    "center_size_metric_length": self.center_size_metric_length,
                    "coordinate_sequence": coordinate_sequence,
                    "movement_feature_point_label": self.eye_center_label,
                    **generic_data,
                }

                if exp_class == "training":
                    self.training_object_trials.append(
                        (exp := field.training(**analysis_keyword_arguments))
                    )
                else:
                    self.novelty_object_trials.append(
                        (exp := field.novelty(**analysis_keyword_arguments))
                    )

            else:
                msg = f"{exp_meta['stage']} has no implementation"
                raise NotImplementedError(msg)

            if (animal_id := exp_meta["animal_id"]) in self.experiment_pairs:
                self.experiment_pairs[animal_id].append(exp)
            else:
                self.experiment_pairs[animal_id] = [exp]

    @cached_property
    def df(self) -> pd.DataFrame:
        """
        Export experimental data to pandas DataFrame

        Useful for exporting to files such as hdf, xlsx, csv, etc

        Returns
        -------
        DataFrame with the combined experiment attributes of all the YMaze objects
        """

        def to_df(data_dict: dict, features: Sequence):
            data_dict = sort_dict_by_key_value(data_dict)
            return pd.DataFrame(
                data_dict.values(),
                index=pd.Series(data_dict.keys(), name="Test", dtype=np.int16),
                columns=pd.MultiIndex.from_tuples(features, names=("Feature", "Area")),
            )

        def feature_area(feature, areas):
            return tuple([(feature, area) for area in areas])

        def movement_feature(category):
            category = str(category)
            return (
                (category, "Displacement"),
                (category, "Median speed"),
                (category, "Median acceleration"),
            )

        habituation_columns = [
            *movement_feature("All"),
            *movement_feature("Periphery"),
            *movement_feature("Center"),
            *feature_area("Entries", ("Periphery", "Center")),
            *feature_area("Time spent", ("Periphery", "Center")),
        ]

        object_columns = [
            *feature_area("Observation instances", ("A", "B", "Total")),
            *feature_area("Observation time", ("A", "B", "Total")),
            ["Object bias score"],
        ]

        novelty_columns = [
            ["Absolute discrimination"],
            ["Discrimination index"],
            ["Novelty preference"],
        ]

        label_vs_data = {}

        if self.habituation_trials:
            habituation_filler = [
                "habituation" for _i in range(len(object_columns + novelty_columns))
            ]
            for nort_habituation in self.habituation_trials:
                label_vs_data[nort_habituation.label] = (
                    nort_habituation.info() + habituation_filler
                )

        if self.training_object_trials:
            training_filler = ["training" for _i in range(len(novelty_columns))]
            for training_trial in self.training_object_trials:
                label_vs_data[training_trial.label] = (
                    training_trial.info() + training_filler
                )

        for novelty_trial in self.novelty_object_trials:
            label_vs_data[novelty_trial.label] = novelty_trial.info()

        return to_df(
            label_vs_data, habituation_columns + object_columns + novelty_columns
        )

    def nort_object_analysis(self):
        if not self.training_object_trials and not self.novelty_object_trials:
            logger.warning(
                "There are neither training or novelty trials in the experiment object, can not analyse"
            )
            return None

    def __repr__(self):
        return self.df
