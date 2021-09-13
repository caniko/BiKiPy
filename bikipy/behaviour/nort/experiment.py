from collections.abc import Mapping, Sequence
from functools import cached_property
from logging import getLogger
from typing import Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from bikipy.behaviour.base import BaseExperiment
from bikipy.behaviour.nort.trial import NortField, NortHabituationTrial
from bikipy.utils.store import sort_dict_by_key_value

logger = getLogger(__name__)


class NortExperiment(BaseExperiment):
    """
    Class for combining several NORT trials under one class for joint analysis
    """

    trials_are_sequential = True

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
        nose_label: str,
        eye_center_label: str,
        torso_label: str,
        nort_field_vs_nort_field_object: Union[Mapping[NortField], None] = None,
        perimeter_border_normal_metric_magnitude: Union[float, None] = None,
        center_metric_length: Union[float, None] = None,
        maximum_radians_inter_gaze_perimeter: float = 0.5 * np.pi,
        *base_trial_args,
        **base_trial_kwargs,
    ):
        """

        :param nose_label: Label of the nose in the df
        :param eye_center_label: Label of the eye center in the df
        :param torso_label: Label of the torso in the df
        :param nort_field_vs_nort_field_object:
        :param perimeter_border_normal_metric_magnitude: The magnitude of the normal between the perimeter
            and the border given in meters
        :param center_metric_length:
        :param maximum_radians_inter_gaze_perimeter:
        :param base_trial_args:
        :param base_trial_kwargs:
        :type nose_label: str
        :type eye_center_label: str
        :type torso_label: str
        :type nort_field_vs_nort_field_object: dict
        :type perimeter_border_normal_metric_magnitude: float
        :type center_metric_length: float
        :type maximum_radians_inter_gaze_perimeter: float
        """
        super().__init__(*base_trial_args, **base_trial_kwargs)

        self.nort_field_vs_nort_field_object = nort_field_vs_nort_field_object

        self.torso_label, self.eye_center_label, self.nose_label = (
            str(torso_label),
            str(eye_center_label),
            str(nose_label),
        )
        self.maximum_radians_inter_gaze_perimeter = float(
            maximum_radians_inter_gaze_perimeter
        )

        self.center_metric_length = (
            float(center_metric_length) if center_metric_length else None
        )
        self.perimeter_border_normal_metric_magnitude = (
            perimeter_border_normal_metric_magnitude
        )

        self.animal_vs_trials = {}
        (
            self.habituation_trials,
            self.training_object_trials,
            self.novelty_object_trials,
        ) = ([], [], [])
        for trial_id, trial_meta in self.trial_id_data_tqdm():
            logger.info(f"Category {trial_meta['stage']}; ID {trial_id}")

            coordinate_sequence = self.trial_id_vs_coordinate_sequences[trial_id]

            generic_kwargs = {
                "video_path": trial_meta["video_path"],
                "coordinate_sequence": coordinate_sequence,
                "movement_feature_point_label": self.eye_center_label,
                "metric_resolution": self.metric_resolution,
                "label": trial_id,
                "inspection_figure_save_root": self.inspection_figure_save_root,
                "rigid_nodes_freezing": (self.eye_center_label, self.torso_label),
            }

            if "animal_id" in trial_meta:
                generic_kwargs["animal_id"] = trial_meta["animal_id"]

            if "inspect" in trial_meta:
                generic_kwargs["inspection_figure_save_root"] = trial_meta["inspect"]
                if "inspect_image" in trial_meta:
                    generic_kwargs["inspect_image"] = trial_meta["inspect_image"]

            exp_class = self.trial_label_to_trial_class_name[
                trial_meta["stage"].lower().replace(" ", "_")
            ]

            if exp_class == "habituation":
                self.habituation_trials.append(
                    (
                        exp := NortHabituationTrial(
                            center_metric_length=self.center_metric_length,
                            **generic_kwargs,
                        )
                    )
                )

            elif exp_class == "training" or exp_class == "novelty":
                try:
                    field = self.nort_field_vs_nort_field_object[
                        trial_meta["field"] - 1
                    ]
                except AttributeError as e:
                    msg = "nort_field_vs_nort_field_object is not defined, which is required when working with training and/or novelty datasets"
                    raise AttributeError(msg) from e

                analysis_keyword_arguments = {
                    "nose_label": self.nose_label,
                    "eye_center_label": self.eye_center_label,
                    "torso_label": self.torso_label,
                    "perimeter_border_normal_metric_magnitude": self.perimeter_border_normal_metric_magnitude,
                    "maximum_radians_inter_gaze_perimeter": self.maximum_radians_inter_gaze_perimeter,
                    "center_metric_length": self.center_metric_length,
                    **generic_kwargs,
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
                msg = f"{trial_meta['stage']} has no implementation"
                raise NotImplementedError(msg)

            if "animal_id" in trial_meta:
                if trial_meta["animal_id"] in self.animal_vs_trials:
                    self.animal_vs_trials[trial_meta["animal_id"]].append(exp)
                else:
                    self.animal_vs_trials[trial_meta["animal_id"]] = [exp]

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
                (category, "Median_speed"),
                (category, "Median_acceleration"),
                (category, "Freezing time"),
            )

        habituation_columns = [
            *movement_feature("All"),
            *movement_feature("Periphery"),
            *movement_feature("Center"),
            *feature_area("Time_spent", ("Periphery", "Center")),
            *feature_area("Entries", ("Periphery", "Center")),
        ]

        object_columns = [
            *feature_area("Observation_instances", ("A", "B", "Total")),
            *feature_area("Observation_time", ("A", "B", "Total")),
            ("Object_bias_score", "Total"),
        ]

        novelty_columns = [
            ("Absolute_discrimination", "Total"),
            ("Discrimination_index", "Total"),
            ("Novelty_preference", "Total"),
        ]

        label_vs_data = {}

        if self.habituation_trials:
            habituation_filler = [
                np.nan for _i in range(len(object_columns + novelty_columns))
            ]
            for nort_habituation in self.habituation_trials:
                label_vs_data[nort_habituation.label] = (
                    nort_habituation.info + habituation_filler
                )

        if self.training_object_trials:
            training_filler = [np.nan for _i in range(len(novelty_columns))]
            for training_trial in self.training_object_trials:
                label_vs_data[training_trial.label] = (
                    training_trial.info + training_filler
                )

        for novelty_trial in self.novelty_object_trials:
            label_vs_data[novelty_trial.label] = novelty_trial.info

        return to_df(
            label_vs_data, habituation_columns + object_columns + novelty_columns
        )

    def nort_object_analysis(self):
        if not self.training_object_trials and not self.novelty_object_trials:
            logger.warning(
                "There are neither training or novelty trials in the experiment object, can not analyse"
            )
            return None

    @property
    def attention_state_distribution(self):
        attention_state_analysis = {
            "proximity&gaze true observation false": [],
            "observation&gaze true proximity false": [],
            "observation&proximity true gaze false": [],
            "proximity true gaze false": [],
            "gaze true proximity false": [],
            "all false": [],
        }
        for novelty_trial in self.novelty_object_trials:
            attention_state_analysis["proximity gaze true observation false"].extend(
                (
                    novelty_trial.a_proximity_filtered
                    & novelty_trial.a_gaze_filtered
                    & (
                        not_a_observance_per_frame := ~novelty_trial.a_observance_per_frame
                    ),
                    #
                    novelty_trial.b_proximity_filtered
                    & novelty_trial.b_gaze_filtered
                    & (
                        not_b_observance_per_frame := ~novelty_trial.b_observance_per_frame
                    ),
                )
            )
            attention_state_analysis["observation gaze true proximity false"].extend(
                (
                    novelty_trial.a_observance_per_frame
                    & novelty_trial.a_gaze_filtered
                    & (not_a_proximity_filtered := ~novelty_trial.a_proximity_filtered),
                    #
                    novelty_trial.b_observance_per_frame
                    & novelty_trial.b_gaze_filtered
                    & (not_b_proximity_filtered := ~novelty_trial.b_proximity_filtered),
                ),
            )
            attention_state_analysis["observation proximity true gaze false"].extend(
                (
                    novelty_trial.a_observance_per_frame
                    & novelty_trial.a_proximity_filtered
                    & (not_a_gaze_filtered := ~novelty_trial.a_gaze_filtered),
                    #
                    novelty_trial.b_observance_per_frame
                    & novelty_trial.b_proximity_filtered
                    & (not_b_gaze_filtered := ~novelty_trial.b_gaze_filtered),
                )
            )
            attention_state_analysis["proximity true gaze false"].extend(
                (
                    novelty_trial.a_proximity_filtered & not_a_gaze_filtered,
                    novelty_trial.b_proximity_filtered & not_b_gaze_filtered,
                )
            )
            attention_state_analysis["gaze true proximity false"].extend(
                (
                    novelty_trial.a_gaze_filtered & not_a_proximity_filtered,
                    novelty_trial.b_gaze_filtered & not_b_proximity_filtered,
                ),
            )

        result = []
        for label, data_set in attention_state_analysis.items():
            for idx, data in enumerate(data_set):
                analysis = np.sum(data) / data.size
                attention_state_analysis[label][idx] = analysis
                result.append((analysis, label))

        return pd.DataFrame(result, columns=("Ratio", "Comparison"))

    def plot_attention_state_distribution(self, bins=13, **sns_displot_kwargs):
        sns.set_theme(style="whitegrid")
        sns.displot(
            self.attention_state_distribution,
            x="Ratio",
            hue="Comparison",
            multiple="stack",
            bins=bins,
            **sns_displot_kwargs,
        )
        plt.show()
        sns.displot(
            self.attention_state_distribution,
            x="Ratio",
            hue="Comparison",
            multiple="stack",
            bins=bins,
            **sns_displot_kwargs,
        )
        plt.show()

    def __repr__(self):
        return self.df
