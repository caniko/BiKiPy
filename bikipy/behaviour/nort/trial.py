from logging import getLogger
from typing import Any, AnyStr, Dict, Sequence, SupportsFloat, Union

import numpy as np
import pandas as pd

from bikipy.behaviour.base import BaseTrial
from bikipy.behaviour.nort.experiment import NortHabituation, NortWithObjects
from bikipy.utils.store import sort_dict_by_key_value

logger = getLogger(__name__)

TRIAL_LABELS_TO_EXPERIMENT_CLASS_NAME = {
    ("t1", "habituation", "open_field", "open field", ): "habitation",
    ("t2", "novelty_observation", "novelty", "test", "training"): "with_objects"
}


class NortTrial(BaseTrial):
    def __init__(
        self,
        exp_ids_range_vs_exp_meta: Dict,
        experiment_box_real_length: SupportsFloat,
        eye_center_label: AnyStr,
        nort_fields: Any = None,
        nose_label: Union[AnyStr, None] = None,
        torso_label: Union[AnyStr, None] = None,
        center_size_real_length: Union[SupportsFloat, None] = None,
        max_radians_gaze_and_object: SupportsFloat = 1 / 4 * np.pi,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

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

        self.habituation_experiments, self.novelty_object_experiments = [], []
        for exp_id, exp_meta in self.exp_ids_range_vs_exp_meta.items():
            logger.info(f"Category {exp_meta['stage']}; ID {exp_id}")

            coordinate_sequence = self.exp_id_vs_coordinate_sequences[exp_id]

            generic_data = {
                "recording_resolution": exp_meta["recording_resolution"],
                "experiment_box_real_length": experiment_box_real_length,
                "fps": self._get_fps(exp_id, exp_meta),
                "label": exp_id,
            }
            if "guiding_image" in exp_meta:
                generic_data["guiding_image"] = exp_meta["guiding_image"]

            exp_class = None
            for labels in TRIAL_LABELS_TO_EXPERIMENT_CLASS_NAME:
                if (exp_stage_label := exp_meta["stage"].lower()) in labels:
                    exp_class = TRIAL_LABELS_TO_EXPERIMENT_CLASS_NAME[exp_stage_label]
            assert exp_class

            if exp_class == "habituation":
                self.habituation_experiments.append(
                    NortHabituation(
                        coordinate_sequence=coordinate_sequence[self.eye_center_label],
                        center_size_real_length=self.center_size_real_length,
                        **generic_data,
                    )
                )

            elif exp_class == "with_objects":
                assert self.nort_fields
                fields = self.nort_fields[exp_meta["field"]]

                self.novelty_object_experiments.append(
                    NortWithObjects(
                        nort_a=fields.constant_object,
                        nort_b=fields.novel_object
                        if exp_meta["stage"] == "test"
                        or exp_meta["stage"] == "novelty_observation"
                        else fields.variable_object,
                        nose_label=self.nose_label,
                        eye_center_label=self.eye_center_label,
                        torso_label=self.torso_label,
                        max_radians_gaze_and_object=self.max_radians_gaze_and_object,
                        center_size_real_length=self.center_size_real_length,
                        coordinate_sequence=coordinate_sequence,
                        movement_feature_point_label=self.eye_center_label,
                        **generic_data,
                    )
                )

            else:
                msg = f"{exp_meta['stage']} has no implementation"
                raise NotImplementedError(msg)

        self.experiments = tuple(
            self.habituation_experiments + self.novelty_object_experiments
        )

    def _get_fps(self, exp_id, exp_meta):
        if isinstance(self.fps, dict):
            return self.fps[exp_id]
        elif "fps" in exp_meta:
            return exp_meta["fps"]
        elif isinstance(self.fps, (int, float)):
            return self.fps
        else:
            msg = (
                "fps has to be defined inside exp_meta, "
                "or in the fps class/trial variable"
            )
            raise AttributeError(msg)

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

        novelty_rows = [
            *feature_area("Observation instances", ("T1", "T2")),
            *feature_area("Observation time", ("T1", "T2", "Total")),
        ]

        habituation_idx_vs_data = {}
        for nort_habituation in self.habituation_experiments:
            habituation_idx_vs_data[
                nort_habituation.label
            ] = nort_habituation.get_info()

        novelty_idx_vs_data = {}
        for novelty_experiment in self.novelty_object_experiments:
            novelty_idx_vs_data[
                novelty_experiment.label
            ] = novelty_experiment.get_info()

        return {
            "Habituation": to_df(habituation_idx_vs_data, habituation_rows),
            "Novelty": to_df(
                novelty_idx_vs_data, habituation_rows + novelty_rows
            ),
        }
