from logging import getLogger

import matplotlib.pyplot as plt
import pandas as pd

from bikipy.behaviour.base import BaseExperiment
from bikipy.behaviour.y_maze.trial import YMazeTrial


logger = getLogger(__name__)


class YMazeExperiment(BaseExperiment):
    def __init__(
        self,
        trial_id_range_vs_perimeter_set: dict[int, dict],
        center_triangle_meter_width: float,
        *args,
        **kwargs,
    ):
        """

        Parameters
        ----------
        trial_id_range_vs_perimeter_set
            Key value pair of experiment ID and perimeter sets
            each depicting the parameters of the experiments within their range.
            The experiment ID range is defined as key : next_key (trial_id:next_trial_id)

        """
        super().__init__(
            *args,
            trial_id_range_vs_common_data=trial_id_range_vs_perimeter_set,
            **kwargs,
        )

        for area_set in trial_id_range_vs_perimeter_set.values():
            for i, arm in enumerate(area_set["arm"], start=1):
                arm.int_label = i
            area_set["center"][0].int_label = 4

        self.trial_id_range_vs_perimeter_sets = self.trial_id_range_vs_common_data

        self.center_triangle_meter_width = float(center_triangle_meter_width)

        if self.inspection_figure_save:
            self.plot()

        self.y_maze_experiments = []
        for trial_id, trial_meta in self.trial_id_data_tqdm():
            logger.info(f"Trial ID {trial_id}")

            experiment_area_set = self.trial_id_range_vs_perimeter_sets[trial_id]

            self.y_maze_experiments.append(
                YMazeTrial(
                    **self.generic_trial_kwargs(trial_id),
                    arms=experiment_area_set["arm"],
                    center=experiment_area_set["center"],
                )
            )

        self.y_maze_experiments = sorted(
            self.y_maze_experiments, key=lambda item: item.semantic_label
        )
        self.trial_id_vs_y_maze = {
            y_maze.semantic_label: y_maze for y_maze in self.y_maze_experiments
        }

    def plot(self, *args, **kwargs):
        previous_id = 0
        for next_trial_id, y_maze in self.trial_id_vs_y_maze.items():
            if not ("invalid" in kwargs and kwargs["invalid"]):
                across_trial_coordinate_sequence = []
                for trial_id in range(previous_id, next_trial_id + 1):
                    try:
                        across_trial_coordinate_sequence.extend(
                            self.trial_id_vs_coordinate_sequence[trial_id][
                                self.feature_tracking_point
                            ]
                        )
                    except KeyError:
                        pass

                y_maze.plot(points=across_trial_coordinate_sequence, *args, **kwargs)
            else:
                y_maze.plot(*args, **kwargs)

            plt.show()

            previous_id = next_trial_id + 1

    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export experimental data to pandas DataFrame

        Useful for exporting to files such as hdf, xlsx, csv, etc

        Returns
        -------
        DataFrame with the combined experiment attributes of all the YMazeTrial objects
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
                *feature_area("Seconds in area", first.arm_center_semantic_labels),
                *feature_area("Area alternations", first.arm_center_semantic_labels),
                *feature_triplet("Triplet alternation", first.arm_semantic_triplets),
            ),
            names=("Feature", "Area/Triplet"),
        )

        unit_length = None
        index_vs_data = {}
        for y_maze in self.y_maze_experiments:
            index_vs_data[y_maze.semantic_label] = (
                y_maze.motion.total_displacement,
                y_maze.motion.median_speed,
                y_maze.motion.median_acceleration,
                y_maze.spontaneous_alternations,
                *tuple(y_maze.seconds_spent_in_areas.values()),
                *tuple(y_maze.area_alternations.values()),
                *tuple(y_maze.triplet_alternation_distribution.values()),
            )
            if not unit_length:
                unit_length = len(index_vs_data[y_maze.semantic_label])

        index_vs_data = dict(sorted(index_vs_data.items(), key=lambda item: item[0]))

        return pd.DataFrame(
            tuple(index_vs_data.values()),
            index=tuple(index_vs_data.keys()),
            columns=feature_order,
        )
