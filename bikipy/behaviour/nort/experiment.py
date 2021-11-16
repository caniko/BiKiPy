from functools import cached_property
from logging import getLogger
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pydantic import Field

from bikipy.behaviour.nort.trial import NortHabituationTrial
from bikipy.behaviour.square import SquareEnclosedExperiment
from bikipy.utils.store import sort_dict_by_key_value

logger = getLogger(__name__)


class NortExperiment(SquareEnclosedExperiment):
    """
    Class for combining several NORT trials under one class for joint analysis
    """

    gaze_travel_direction_point_label: str = Field(
        description="Label signifying the area where the gaze vector"
    )
    gaze_start_point_label: str = Field(description="Label of the eye center in the df")
    torso_label: str = Field(description="Label of the torso in the df")
    nort_field_vs_nort_field_object: Optional[dict] = Field(
        None, description="Label of the nose in the df"
    )
    perimeter_border_normal_metric_magnitude: Optional[float] = Field(
        None,
        description="The magnitude of the normal between the perimeter and the border given in meters",
    )
    maximum_radians_inter_gaze_perimeter: float = Field(0.5 * np.pi)
    trial_label_to_trial_class_name: dict = {
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

    _trials_are_sequential: bool = True
    _period_columns = ("T1", "T2", "Total")

    def trial_keyword_arguments(self, trial_id: int) -> dict:
        generic = super().trial_keyword_arguments(trial_id)
        if self.trial_label_to_trial_class_name[generic["stage"]] != "habituation":
            return {
                **generic,
                "gaze_travel_direction_point_label": self.gaze_travel_direction_point_label,
                "gaze_start_point_label": self.gaze_start_point_label,
                "perimeter_border_normal_metric_magnitude": self.perimeter_border_normal_metric_magnitude,
                "maximum_radians_inter_gaze_perimeter": self.maximum_radians_inter_gaze_perimeter,
            }
        return generic

    @cached_property
    def df(self) -> pd.DataFrame:
        """
        Export experimental data to pandas DataFrame

        Useful for exporting to files such as hdf, xlsx, csv, etc

        Returns
        -------
        DataFrame with the combined experiment attributes of all the YMazeTrial objects
        """
        base_columns = [
            ("All", "Stage"),
            *self._motion_2d_multi_indexer("All"),
            *self._motion_2d_multi_indexer("Periphery"),
            *self._motion_2d_multi_indexer("Center"),
            *self._feature_2d_multi_indexer("Time_spent", ("Periphery", "Center")),
            *self._feature_2d_multi_indexer("Entries", ("Periphery", "Center")),
        ]

        object_columns = [
            *self._feature_2d_multi_indexer(
                "Observation_instances", ("A", "B", "Total")
            ),
            *self._feature_2d_multi_indexer("Observation_time", ("A", "B", "Total")),
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
                label_vs_data[nort_habituation.int_id] = (
                    nort_habituation.info + habituation_filler
                )

        if self.training_object_trials:
            training_filler = [np.nan for _i in range(len(novelty_columns))]
            for training_trial in self.training_object_trials:
                label_vs_data[training_trial.int_id] = (
                    training_trial.info + training_filler
                )

        for novelty_trial in self.novelty_object_trials:
            label_vs_data[novelty_trial.int_id] = novelty_trial.info

        data_dict = sort_dict_by_key_value(label_vs_data)
        df = pd.DataFrame(
            data_dict.values(),
            index=self._frame_index,
            columns=pd.MultiIndex.from_tuples(
                base_columns + object_columns + novelty_columns,
                names=("Feature", "Area"),
            ),
        )
        # df[("All", "Stage")] = FletcherContinuousArray(df[("All", "Stage")])
        return df

    def nort_object_analysis(self):
        if not self.training_object_trials and not self.novelty_object_trials:
            logger.warning(
                "There are neither training or novelty trials in the experiment object, can not analyse"
            )
            return None

    @cached_property
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
            attention_state_analysis["proximity&gaze true observation false"].extend(
                (
                    novelty_trial.a_proximity_filtered
                    & novelty_trial.a_gaze_filtered
                    & ~novelty_trial.a_observance_per_frame,
                    #
                    novelty_trial.b_proximity_filtered
                    & novelty_trial.b_gaze_filtered
                    & ~novelty_trial.b_observance_per_frame,
                )
            )
            attention_state_analysis["observation&gaze true proximity false"].extend(
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
            attention_state_analysis["observation&proximity true gaze false"].extend(
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
