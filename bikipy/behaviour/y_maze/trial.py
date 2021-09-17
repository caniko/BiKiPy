from logging import getLogger
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bikipy.behaviour.base import BaseExperiment
from bikipy.behaviour.y_maze.experiment import YMaze
from bikipy.perimeter.base import PolygonalPerimeter
from bikipy.perimeter.triangular import TriangularPerimeter
from bikipy.utils.store import RangeDict

logger = getLogger(__name__)


class YMazeTrial(BaseExperiment):
    def __init__(
        self,
        exp_id_range_vs_area_sets: dict[
            int, dict[str, Union[PolygonalPerimeter, TriangularPerimeter]]
        ],
        feature_tracking_point: str,
        center_triangle_meter_width: float,
        *args,
        **kwargs,
    ):
        """

        Parameters
        ----------
        exp_id_range_vs_area_sets
            Key value pair of experiment ID and perimeter sets
            each depicting the parameters of the experiments within their range.
            The experiment ID range is defined as key : next_key (exp_id:next_exp_id)

        """
        super().__init__(*args, **kwargs)

        # for key, value in exp_id_range_vs_area_sets.items():
        #     arms, center = mean_intersecting_points_on_borders(
        #         value["arms"], value["center"]
        #     )
        #     exp_id_range_vs_area_sets[key] = {"arms": arms, "center": center}

        for area_set in exp_id_range_vs_area_sets.values():
            for i, arm in enumerate(area_set["arms"], start=1):
                arm.int_label = i
            area_set["center"].int_label = 4

        self.exp_id_range_vs_area_sets = RangeDict(exp_id_range_vs_area_sets)

        self.feature_tracking_point = str(feature_tracking_point)
        self.center_triangle_meter_width = float(center_triangle_meter_width)

        if self.inspection_figure_save:
            self.plot()

        self.y_maze_experiments = []
        for (
            exp_id,
            coordinate_sequence,
        ) in self.trial_id_vs_coordinate_sequences.items():
            experiment_area_set = self.exp_id_range_vs_area_sets[exp_id]
            unit_per_pixel = self.center_triangle_meter_width / np.linalg.norm(
                experiment_area_set["center"][0] - experiment_area_set["center"][1]
            )

            if isinstance(self.fps, dict):
                exp_fps = self.fps[exp_id]
            elif isinstance(self.fps, (float, int)):
                exp_fps = self.fps
            else:
                msg = "fps has to be defined"
                raise AttributeError(msg)

            self.y_maze_experiments.append(
                YMaze(
                    arms=experiment_area_set["arms"],
                    center=experiment_area_set["center"],
                    average_intersections=False,
                    coordinate_sequence=coordinate_sequence[
                        self.feature_tracking_point
                    ],
                    fps=exp_fps,
                    unit_per_pixel=unit_per_pixel,
                    label=exp_id,
                )
            )

        self.y_maze_experiments = sorted(
            self.y_maze_experiments, key=lambda item: item.semantic_label
        )
        self.exp_id_vs_y_maze = {
            y_maze.semantic_label: y_maze for y_maze in self.y_maze_experiments
        }

    def plot(self, *args, **kwargs):
        previous_id = 0
        for next_exp_id, y_maze in self.exp_id_vs_y_maze.items():
            if not ("invalid" in kwargs and kwargs["invalid"]):
                across_trial_coordinate_sequence = []
                for exp_id in range(previous_id, next_exp_id + 1):
                    try:
                        across_trial_coordinate_sequence.extend(
                            self.trial_id_vs_coordinate_sequences[exp_id][
                                self.feature_tracking_point
                            ]
                        )
                    except KeyError:
                        pass

                y_maze.plot(points=across_trial_coordinate_sequence, *args, **kwargs)
            else:
                y_maze.plot(*args, **kwargs)

            plt.show()

            previous_id = next_exp_id + 1

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
