from typing import Any, AnyStr, SupportsFloat, Dict

import pandas as pd
import numpy as np

from bikipy.behaviour.nort.experiment import NortHabituation, NortWithObjects
from bikipy.behaviour.base import BaseTrial


class NortTrial(BaseTrial):
    def __init__(
        self,
        exp_ids_range_vs_exp_meta: Dict,
        torso_label: AnyStr,
        eye_center_label: AnyStr,
        nose_label: AnyStr,
        experiment_box_size_cm: SupportsFloat,
        center_size_cm: SupportsFloat,
        cm_per_pixel: SupportsFloat,
        max_radians_gaze_and_object: SupportsFloat = 1 / 4 * np.pi,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(*args, **kwargs)

        self.exp_ids_range_vs_exp_meta = dict(exp_ids_range_vs_exp_meta)
        self.torso_label, self.eye_center_label, self.nose_label = (
            str(torso_label),
            str(eye_center_label),
            str(nose_label),
        )
        (
            self.experiment_box_size_cm,
            self.center_size_cm,
            self.cm_per_pixel,
            self.max_radians_gaze_and_object,
        ) = (
            float(experiment_box_size_cm),
            float(center_size_cm),
            float(cm_per_pixel),
            float(max_radians_gaze_and_object),
        )

        self.habituation_experiments, self.novelty_object_experiments = [], []
        for exp_id, exp_meta in self.exp_ids_range_vs_exp_meta.items():
            if exp_meta["type"] == "habituation":
                self.habituation_experiments.append(
                    NortHabituation(
                        coordinate_sequence=self.exp_id_vs_coordinate_sequences[exp_id][
                            self.eye_center_label
                        ],
                        recording_resolution=exp_meta["recording_resolution"],
                        experiment_box_size_cm=self.experiment_box_size_cm,
                        center_size_cm=self.center_size_cm,
                        fps=self.fps,
                        cm_per_pixel=self.cm_per_pixel,
                        eye_center_label=self.eye_center_label,
                        label=exp_id,
                    )
                )

            elif exp_meta["type"] == "novelty_observation":
                self.novelty_object_experiments.append(
                    NortWithObjects(
                        nort_a=exp_meta["A"],
                        nort_b=exp_meta["B"],
                        nose_label=self.nose_label,
                        eye_center_label=self.eye_center_label,
                        torso_label=self.torso_label,
                        max_radians_gaze_and_object=self.max_radians_gaze_and_object,
                        recording_resolution=exp_meta["recording_resolution"],
                        experiment_box_size_cm=self.experiment_box_size_cm,
                        center_size_cm=self.center_size_cm,
                        exp_id_vs_coordinate_data_path=self.exp_id_vs_coordinate_sequences[
                            exp_id
                        ],
                        fps=self.fps,
                        cm_per_pixel=self.cm_per_pixel,
                        label=exp_id,
                    )
                )

        self.experiments = tuple(
            self.habituation_experiments + self.novelty_object_experiments
        )

    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export experimental data to pandas DataFrame

        Useful for exporting to files such as hdf, xlsx, csv, etc

        Returns
        -------
        DataFrame with the combined experiment attributes of all the YMaze objects
        """

        def general_data(nort_obj):
            return (
                nort_obj.total_displacement,
                nort_obj.mean_speed,
                nort_obj.mean_acceleration,
                nort_obj.periphery_displacement,
                nort_obj.periphery_mean_speed,
                nort_obj.periphery_mean_acceleration,
                nort_obj.center_displacement,
                nort_obj.center_mean_speed,
                nort_obj.center_mean_acceleration,
            )

        def feature_area(feature, areas):
            return tuple([(feature, area) for area in areas])

        def movement_feature(category):
            return (
                ("Displacement", category),
                ("Mean speed", category),
                ("Mean acceleration", category),
            )

        feature_order = pd.MultiIndex.from_tuples(
            (
                *movement_feature("All"),
                *movement_feature("Periphery"),
                *movement_feature("Center"),
                *feature_area("Entries", ("Periphery", "Center")),
                *feature_area("Time spent", ("Periphery", "Center")),
                *feature_area("Observations", ("A", "B")),
            ),
            names=("Feature", "Area"),
        )

        habituation_filler = ("Habituation", "Habituation")

        index_vs_data = {}
        for nort_habituation in self.habituation_experiments:
            index_vs_data[nort_habituation.label] = (
                *general_data(nort_habituation),
                *habituation_filler
            )
        for novelty_experiment in self.novelty_object_experiments:
            index_vs_data[novelty_experiment.label] = (
                *general_data(novelty_experiment),
                novelty_experiment.observe_times_a,
                novelty_experiment.observe_times_b
            )

        index_vs_data = dict(sorted(index_vs_data.items(), key=lambda item: item[0]))

        return pd.DataFrame(
            tuple(index_vs_data.values()),
            index=tuple(index_vs_data.keys()),
            columns=feature_order,
        )
