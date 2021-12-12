from functools import cached_property, lru_cache
from typing import Optional

from bikipy._base_class import BikipyBase
from bikipy.behaviour.base import BaseExperiment, BaseTrial
from bikipy.perimeter.base import Perimeter2D, PerimeterSet, Perimeter
from bikipy.utils.typing import NDArray


class RadialMazeBase(BikipyBase):
    corridor_meter_width: float


class BaseRadialMazeExperiment(BaseExperiment, RadialMazeBase):
    pass


class BaseRadialMazeTrial(BaseTrial, RadialMazeBase):
    center: Perimeter2D
    arms: tuple
    reference_point: Optional[NDArray] = None

    def __init__(self, **data):
        if data["reference_point"]:
            data["center"] = data["center"].change_reference(data["reference_point"])
            data["arms"] = data["arms"].change_reference(data["reference_point"])

        data["center"].int_label = 1
        for i, arm_idx in enumerate(range(len(data["arms"])), start=2):
            data["arms"][arm_idx].int_label = i

        super().__init__(**data)

    @property
    def feature_summary_row(self) -> list:
        pass

    @cached_property
    def perimeter_set(self):
        return PerimeterSet(perimeters=(self.center, *self.arms))

    @cached_property
    def meters_per_pixel(self):
        return compute_meter_per_pixel(
            self.center.mean_length, self.corridor_meter_width
        )

    @cached_property
    def _border_presence_data(self):
        return Perimeter.detect_sequential_border_presence(
            self.coordinates_per_frame,
            self.arms,
            inferior_poly_border_instances=[self.center],
        )

    @property
    def alternation_sequence(self):
        return self._border_presence_data[0]

    @property
    def valid_indices(self):
        return self._border_presence_data[1]

    @property
    def valid_boolean_index(self):
        return self._border_presence_data[2]

    @cached_property
    def seconds_spent_in_areas(self) -> dict:
        """
        The time spent in each area; arms and center

        Returns
        -------
        dict, area vs time
        """

        result = copy(self._arm_center_int_id_to_seconds)
        for label, counts in unique_with_counts_zipped(self.alternation_sequence):
            assert label in result, f"{label} is not in {tuple(result.keys())})"
            result[label] = (counts / self.fps) if self.fps else counts

        return generic_int_to_semantic_key_translator(result)

    @cached_property
    def area_alternations(self) -> dict:
        """
        The number of alternations to every arm and center

        Returns
        -------
        dict, arm label vs alternations to arm
        """

        result = copy(self._arm_center_int_id_to_seconds)
        for label, counts in unique_with_counts_zipped(
            self.reduced_alternation_sequence
        ):
            assert label in result
            result[label] = counts

        if not result[self.center.int_id]:
            result[self.center.int_id] = 0

        if result[self.center.int_id] < (
            minimum_center_entries := ceil(self.sum_of_alternations / 2.0)
        ):
            logger.warning(
                f"{self.center.int_id}: The number of alternations to the center, "
                f"{result[self.center.int_id]} can't be less than the "
                f"ceil of half of the total arm alternations, {minimum_center_entries}"
            )

        return result

    @cached_property
    def triplet_alternation_distribution(self) -> dict:
        """
        Define the triplet alternation distribution.

        A y-maze has three arms and one center, compute the number of occurrences
        a given triplet has. There are six possible triplets, six factorial (6!).

        Returns
        -------
        dict, triplet vs number of occurrences.
        """
        distribution = copy(self._arm_triplet_dict)
        for i in range(self.sum_of_alternations):
            current_triplet = self.reduced_without_center[i : i + 3]
            if 1 in current_triplet and 2 in current_triplet and 3 in current_triplet:
                distribution[tuple(current_triplet)] += 1

        result = {}
        for key, value in distribution.items():
            semantic_key = "".join([self.int_to_label[integer] for integer in key])
            result[semantic_key] = value

        return result

    @cached_property
    def spontaneous_alternations(self) -> float:
        """
        Define the number of spontaneous alternations between each y-maze arm

        A y-maze has three arms and one center, compute the number of occurrences
        triplet with unique arms. There are six possible triplets, six factorial (6!).

        Returns
        -------
        float, defining the percentage ratio between triplet consisting of unique arms
        and sum of all triplet alternations.
        """

        if self.sum_of_alternations == 0:
            return 0

        alternations = 0
        for i in range(self.sum_of_alternations):
            current_triplet = self.reduced_without_center[i : i + 3]
            if 1 in current_triplet and 2 in current_triplet and 3 in current_triplet:
                alternations += 1

        assert self.sum_of_alternations > 0, self.sum_of_alternations

        return 100.0 * alternations / self.sum_of_alternations

    @cached_property
    def _arm_center_int_labels(self):
        return

    @cached_property
    def _arm_center_int_id_to_seconds(self):
        return {area: 0 for area in self._arm_center_int_labels}


@lru_cache
def compute_meter_per_pixel(
    corridor_pixel_length: float, corridor_metric_width: float
) -> float:
    return corridor_pixel_length / corridor_metric_width
