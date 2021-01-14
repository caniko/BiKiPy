from logging import getLogger
from typing import AnyStr, Dict, Sequence, SupportsFloat, Union

import numpy as np
import pandas as pd

from bikipy.behaviour.base import BaseTrial
from bikipy.behaviour.nort.experiment import (
    NortHabituation,
    NortOpenField,
    NortWithObjects,
)
from bikipy.utils.store import sort_dict_by_key_value

logger = getLogger(__name__)


class NortTrial(BaseTrial):
    def __init__(
        self,
        exp_ids_range_vs_exp_meta: Dict,
        experiment_box_real_length: SupportsFloat,
        eye_center_label: AnyStr,
        nose_label: Union[AnyStr, None] = None,
        torso_label: Union[AnyStr, None] = None,
        center_size_real_length: Union[SupportsFloat, None] = None,
        max_radians_gaze_and_object: SupportsFloat = 1 / 4 * np.pi,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

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

        (
            self.open_field_experiements,
            self.habituation_experiments,
            self.novelty_object_experiments,
        ) = ([], [], [])
        for exp_id, exp_meta in self.exp_ids_range_vs_exp_meta.items():
            logger.info(f"Category {exp_meta['exp_category']}; ID {exp_id}")

            coordinate_sequences = self.exp_id_vs_coordinate_sequences[exp_id]

            generic_data = {
                "recording_resolution": exp_meta["recording_resolution"],
                "experiment_box_real_length": experiment_box_real_length,
                "fps": self._get_fps(exp_id, exp_meta),
                "label": exp_id,
            }
            if exp_meta["exp_category"] == "open field":
                self.open_field_experiements.append(
                    NortOpenField(
                        coordinate_sequence=coordinate_sequences[self.eye_center_label],
                        **generic_data,
                    )
                )

            elif exp_meta["exp_category"] == "habituation":
                self.habituation_experiments.append(
                    NortHabituation(
                        coordinate_sequence=coordinate_sequences[self.eye_center_label],
                        center_size_real_length=self.center_size_real_length,
                        **generic_data,
                    )
                )

            elif exp_meta["exp_category"] == "novelty_observation":
                self.novelty_object_experiments.append(
                    NortWithObjects(
                        nort_a=exp_meta["A"],
                        nort_b=exp_meta["B"],
                        nose_label=self.nose_label,
                        eye_center_label=self.eye_center_label,
                        torso_label=self.torso_label,
                        max_radians_gaze_and_object=self.max_radians_gaze_and_object,
                        center_size_real_length=self.center_size_real_length,
                        coordinate_sequences=coordinate_sequences,
                        movement_feature_point_label=self.eye_center_label,
                        **generic_data,
                    )
                )

            else:
                msg = f"{exp_meta['exp_category']} has no implementation"
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

        base_rows = list(movement_feature("All"))

        habituation_rows = [
            *movement_feature("Periphery"),
            *movement_feature("Center"),
            *feature_area("Entries", ("Periphery", "Center")),
            *feature_area("Time spent", ("Periphery", "Center")),
        ]

        novelty_rows = [
            *feature_area("Observation instances", ("A", "B")),
            *feature_area("Observation time", ("A", "B", "Total")),
        ]

        open_field_idx_vs_data = {}
        for open_field in self.open_field_experiements:
            open_field_idx_vs_data[open_field.semantic_label] = open_field.get_info()

        habituation_idx_vs_data = {}
        for nort_habituation in self.habituation_experiments:
            habituation_idx_vs_data[
                nort_habituation.semantic_label
            ] = nort_habituation.get_info()

        novelty_idx_vs_data = {}
        for novelty_experiment in self.novelty_object_experiments:
            novelty_idx_vs_data[
                novelty_experiment.semantic_label
            ] = novelty_experiment.get_info()

        return {
            "Open-Field": to_df(open_field_idx_vs_data, base_rows),
            "Habituation": to_df(habituation_idx_vs_data, base_rows + habituation_rows),
            "Novelty": to_df(
                novelty_idx_vs_data, base_rows + habituation_rows + novelty_rows
            ),
        }
