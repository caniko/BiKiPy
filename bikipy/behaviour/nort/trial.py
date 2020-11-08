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
        *args, **kwargs
    ):
        """"""
        super().__init__(*args, **kwargs)

        self.exp_ids_range_vs_exp_meta = dict(exp_ids_range_vs_exp_meta)
        self.torso_label, self.eye_center_label, self.nose_label = (
            str(torso_label), str(eye_center_label), str(nose_label)
        )
        self.experiment_box_size_cm, self.center_size_cm, self.cm_per_pixel, self.max_radians_gaze_and_object = float(experiment_box_size_cm), float(center_size_cm), float(cm_per_pixel), float(max_radians_gaze_and_object)

        self.habituation_experiments, self.novelty_object_experiments = [], []
        for exp_id, exp_meta in self.exp_ids_range_vs_exp_meta.items():
            if exp_meta["type"] == "habituation":
                self.habituation_experiments.append(
                    NortHabituation(
                        coordinate_sequence=self.exp_id_vs_coordinate_sequences[exp_id][self.eye_center_label],
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
                        exp_id_vs_coordinate_data_path=self.exp_id_vs_coordinate_sequences[exp_id],
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

        def feature_area(feature, arm_center_labels):
            return tuple([(feature, area) for area in arm_center_labels])

        def feature_triplet(feature, triplets):
            return tuple([(feature, area) for area in triplets])

        first = self.y_maze_experiments[0]
        feature_order = pd.MultiIndex.from_tuples(
            (
                ("Displacement", ""),
                ("Mean speed", ""),
                ("Mean acceleration", ""),
                ("Spontaneous alternations", ""),
                *feature_area("Seconds in area", first.arm_center_labels),
                *feature_area("Area alternations", first.arm_center_labels),
                *feature_triplet("Triplet alternation", first.arm_triplets),
            ),
            names=("Feature", "Area/Triplet"),
        )

        unit_length = None
        index_vs_data = {}
        for y_maze in self.y_maze_experiments:
            index_vs_data[y_maze.label] = (
                y_maze.displacement,
                y_maze.mean_speed,
                y_maze.mean_acceleration,
                y_maze.spontaneous_alternations,
                *tuple(y_maze.seconds_spent_in_areas.values()),
                *tuple(y_maze.area_alternations.values()),
                *tuple(y_maze.triplet_alternation_distribution.values()),
            )
            if not unit_length:
                unit_length = len(index_vs_data[y_maze.label])

        index_vs_data = dict(sorted(index_vs_data.items(), key=lambda item: item[0]))

        return pd.DataFrame(
            tuple(index_vs_data.values()),
            index=tuple(index_vs_data.keys()),
            columns=feature_order,
        )
