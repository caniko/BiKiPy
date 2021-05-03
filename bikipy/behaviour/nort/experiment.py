from logging import getLogger
from typing import Any, AnyStr, Dict, Sequence, SupportsFloat, Union

import numpy as np
import pandas as pd

from bikipy.behaviour.base import BaseExperiment
from bikipy.behaviour.nort.trial import (
    NortHabituation,
    NortObjectTraining,
    NortNovelObject,
)
from bikipy.utils.store import sort_dict_by_key_value


logger = getLogger(__name__)


class NortExperiment(BaseExperiment):
    """
    Class for combining several NORT trials under one class for joint analysis
    """
    period_columns = ("T1", "T2", "Total")

    box_area_names = ("Periphery", "Center")

    trial_label_to_experiment_class_name = {
        "habituation": "habituation",
        "open_field": "habituation",

        "t1": "training",
        "training": "training",

        "t2": "novelty",
        "novelty_observation": "novelty",
        "novelty": "novelty"
    }

    def __init__(
        self,
        exp_ids_range_vs_exp_meta: Dict,
        experiment_box_real_length: SupportsFloat,
        eye_center_label: AnyStr,
        nort_fields: Any = None,
        nose_label: Union[AnyStr, None] = None,
        torso_label: Union[AnyStr, None] = None,
        center_size_real_length: Union[SupportsFloat, None] = None,
        max_radians_gaze_and_object: SupportsFloat = 1. / 4. * np.pi,
        *base_trial_args,
        **base_trial_kwargs,
    ):
        """
        Parameters
        ----------
        exp_ids_range_vs_exp_meta
        experiment_box_real_length
        eye_center_label
        nort_fields
        nose_label
        torso_label
        center_size_real_length
        max_radians_gaze_and_object
        base_trial_args
        base_trial_kwargs
        """
        super().__init__(*base_trial_args, **base_trial_kwargs)

        self.nort_fields = nort_fields

        self.exp_ids_range_vs_exp_meta = dict(exp_ids_range_vs_exp_meta)
        self.torso_label, self.eye_center_label, self.nose_label = (
            str(torso_label),
            str(eye_center_label),
            str(nose_label),
        )
        self.experiment_box_real_length, self.max_radians_gaze_and_object = (
            float(experiment_box_real_length),
            float(max_radians_gaze_and_object),
        )

        self.center_size_real_length = (
            float(center_size_real_length) if center_size_real_length else None
        )

        self.experiment_pairs = {}
        (
            self.habituation_trials,
            self.training_object_trials,
            self.novelty_object_trials,
        ) = ([], [], [])
        for exp_id, exp_meta in self.exp_ids_range_vs_exp_meta.items():
            logger.info(f"Category {exp_meta['stage']}; ID {exp_id}")

            coordinate_sequence = self.exp_id_vs_coordinate_sequences[exp_id]

            generic_data = {
                "recording_resolution": exp_meta["recording_resolution"],
                "experiment_box_real_length": experiment_box_real_length,
                "label": exp_id,
            }
            if "guiding_image" in exp_meta:
                generic_data["guiding_image"] = exp_meta["guiding_image"]

            if isinstance(self.fps, dict):
                generic_data["fps"] = self.fps[exp_id]
            elif "fps" in exp_meta:
                generic_data["fps"] = exp_meta["fps"]
            elif isinstance(self.fps, (int, float)):
                generic_data["fps"] = self.fps
            else:
                msg = "fps has to be defined inside exp_meta, or in the class"
                raise AttributeError(msg)

            exp_class = None
            for labels in self.trial_label_to_experiment_class_name:
                if exp_meta["stage"].lower() == labels:
                    exp_class = self.trial_label_to_experiment_class_name[labels]
            assert exp_class

            if exp_class == "habituation":
                self.habituation_trials.append(
                    (
                        exp := NortHabituation(
                            coordinate_sequence=coordinate_sequence[
                                self.eye_center_label
                            ],
                            center_size_real_length=self.center_size_real_length,
                            **generic_data,
                        )
                    )
                )

            elif exp_class == "training" or exp_class == "novelty":
                assert self.nort_fields
                fields = self.nort_fields[exp_meta["field"]]

                with_object_arguments = {
                    "nort_a": fields.constant_object,
                    "nort_b": fields.novel_object
                    if exp_meta["stage"] == "test"
                    or exp_meta["stage"] == "novelty_observation"
                    else fields.variable_object,
                    "nose_label": self.nose_label,
                    "eye_center_label": self.eye_center_label,
                    "torso_label": self.torso_label,
                    "max_radians_gaze_and_object": self.max_radians_gaze_and_object,
                    "center_size_real_length": self.center_size_real_length,
                    "coordinate_sequence": coordinate_sequence,
                    "movement_feature_point_label": self.eye_center_label,
                    **generic_data,
                }

                if exp_class == "training":
                    self.training_object_trials.append(
                        (exp := NortObjectTraining(**with_object_arguments))
                    )
                else:
                    self.novelty_object_trials.append(
                        (exp := NortNovelObject(**with_object_arguments))
                    )

            else:
                msg = f"{exp_meta['stage']} has no implementation"
                raise NotImplementedError(msg)

            if (animal_id := exp_meta["animal_id"]) in self.experiment_pairs:
                self.experiment_pairs[animal_id].append(exp)
            else:
                self.experiment_pairs[animal_id] = [exp]

    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export experimental data to pandas DataFrame

        Useful for exporting to files such as hdf, xlsx, csv, etc

        Returns
        -------
        DataFrame with the combined experiment attributes of all the YMaze objects
        """

        def to_df(data_dict: Dict, features: Sequence):
            feature_order = pd.MultiIndex.from_tuples(
                features, names=("Feature", "Area")
            )

            data_dict = sort_dict_by_key_value(data_dict)
            return pd.DataFrame(
                tuple(data_dict.values()),
                index=tuple(data_dict.keys()),
                columns=feature_order,
            )

        def feature_area(feature, areas):
            return tuple([(feature, area) for area in areas])

        def movement_feature(category):
            category = str(category)
            return (
                ("Displacement", category),
                ("Mean speed", category),
                ("Mean acceleration", category),
            )

        habituation_rows = [
            *movement_feature("All"),
            *movement_feature("Periphery"),
            *movement_feature("Center"),
            *feature_area("Entries", ("Periphery", "Center")),
            *feature_area("Time spent", ("Periphery", "Center")),
        ]

        object_rows = [
            *feature_area("Observation instances", ("A", "B")),
            *feature_area("Observation time", ("A", "B", "Total")),
            ["Object bias score"],
        ]

        novelty_rows = [
            ["Absolute discrimination"],
            ["Discrimination index"],
            ["Novelty preference"],
        ]

        label_vs_data = {}

        habituation_filler = [
            "habituation" for _i in range(len(object_rows + novelty_rows))
        ]
        for nort_habituation in self.habituation_trials:
            label_vs_data[nort_habituation.label] = (
                *nort_habituation.get_info(),
                *habituation_filler,
            )

        training_filler = ["training" for _i in range(len(novelty_rows))]
        for training_experiment in self.training_object_trials:
            label_vs_data[training_experiment.label] = (
                *training_experiment.get_info(),
                *training_filler,
            )

        for novelty_experiment in self.novelty_object_trials:
            label_vs_data[novelty_experiment.label] = novelty_experiment.get_info()

        return to_df(label_vs_data, habituation_rows + object_rows + novelty_rows)
