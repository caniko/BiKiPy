from typing import Any, AnyStr, SupportsFloat, Dict

import pandas as pd

from bikipy.behaviour.nort.experiment import NortHabituation, NortWithObjects


class YMazeTrial:
    def __init__(
        self,
        exp_ids_range_vs_exp_meta: Dict,
        exp_id_vs_coordinate_sequence: Dict,
        region_of_interest: AnyStr,
        fps: SupportsFloat,
        center_triangle_cm_width: SupportsFloat,
        label: Any,
    ):
        """

        Parameters
        ----------
        exp_id_range_vs_area_sets
            Key value pair of experiment ID and border sets
            each depicting the parameters of the experiments within their range.
            The experiment ID range is defined as key : next_key (exp_id:next_exp_id)

        exp_id_vs_coordinate_sequence
        region_of_interest
        fps
        center_triangle_cm_width
        label
        """
        self.exp_id_range_vs_area_sets = dict(exp_id_range_vs_area_sets)
        self.exp_id_vs_coordinate_sequence = dict(exp_id_vs_coordinate_sequence)
        self.region_of_interest = str(region_of_interest)
        self.fps = float(fps)
        self.center_triangle_cm_width = float(center_triangle_cm_width)
        self.label = label

        keys = tuple([int(exp_id) for exp_id in self.exp_id_range_vs_area_sets.keys()])
        self.exp_id_ranges = tuple(
            [
                tuple([exp_id for exp_id in range(keys[i], keys[i + 1])])
                for i in range(len(keys) - 1)
            ]
        )
        self.last_exp_area_info_id = keys[-1]

        self.area_sets = tuple(self.exp_id_range_vs_area_sets.values())

        y_maze_experiments = []
        for exp_id, coordinate_sequence in self.exp_id_vs_coordinate_sequence.items():
            exp_id = int(exp_id)
            experiment_area_set = None
            for i, exp_range in enumerate(self.exp_id_ranges):
                if exp_id in exp_range:
                    experiment_area_set = self.area_sets[i]
                    break
            if not experiment_area_set:
                if exp_id >= self.last_exp_area_info_id:
                    experiment_area_set = self.area_sets[-1]
                else:
                    msg = f"exp ID {exp_id} is not in {self.exp_id_ranges}, last area info exp ID key {self.last_exp_area_info_id}"
                    raise ValueError(msg)

            y_maze_experiments.append(
                YMaze(
                    coordinate_sequence[self.region_of_interest],
                    experiment_area_set["arms"],
                    experiment_area_set["center"],
                    self.fps,
                    self.center_triangle_cm_width,
                    exp_id,
                )
            )
        self.y_maze_experiments = sorted(
            y_maze_experiments, key=lambda item: item.label
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
