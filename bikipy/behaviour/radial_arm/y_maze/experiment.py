from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bikipy.behaviour.base import BaseExperiment
from bikipy.behaviour.radial_arm.y_maze.trial import YMazeTrial

logger = getLogger(__name__)


class YMazeExperiment(BaseExperiment):
    center_triangle_meter_width: float

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(**kwargs)

        for area_set in trial_id_range_vs_perimeter_set.values():
            for i, arm in enumerate(area_set["arm"], start=1):
                arm.int_id = i
            area_set["center"][0].int_id = 4

        self.trial_id_range_vs_perimeter_sets = self.trial_id_range_vs_common_data

        self.center_triangle_meter_width = float(center_triangle_meter_width)
        self.units_per_pixel = (
            area_set["center"][0].mean_length / self.center_triangle_meter_width
        )

        if self.inspection_figure_save:
            self.plot()

        self.y_maze_experiments = []
        for trial_id, trial_meta in self.trial_id_data_tqdm():
            logger.info(f"Trial ID {trial_id}")

            experiment_area_set = self.trial_id_range_vs_perimeter_sets[trial_id]

            self.y_maze_experiments.append(
                YMazeTrial(
                    **self.trial_keyword_arguments(trial_id),
                    arms=experiment_area_set["arm"],
                    center=experiment_area_set["center"][0],
                    point_label_for_motion_features=self.point_label_for_motion_features,
                )
            )

        self.y_maze_experiments = sorted(
            self.y_maze_experiments, key=lambda item: item.int_id
        )
        self.trial_id_vs_y_maze = {
            y_maze.int_id: y_maze for y_maze in self.y_maze_experiments
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
