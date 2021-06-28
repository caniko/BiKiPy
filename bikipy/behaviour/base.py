from functools import cached_property
from logging import getLogger
from typing import Any, Sequence, Union

from tqdm import tqdm
import numpy as np

from bikipy.feature.motion import Motion, frozen_frames, displacement_by_frame
from bikipy.reader.deeplabcut import DeepLabCutReader
from bikipy.utils.store import RangeDict
from bikipy.utils.video import get_video_data

logger = getLogger(__name__)


class BaseExperiment:
    def __init__(
        self,
        metric_resolution: Union[float, Sequence[float]],
        trial_id_vs_data: dict,
        trial_id_range_vs_data: Union[dict, None] = None,
        fps: Union[dict, float, None] = None,
        coordinate_data_format: str = "deeplabcut",
        label: Any = None,
        func_inspect: bool = False,
        **data_import_kwargs,
    ):
        self.trial_id_vs_data = dict(trial_id_vs_data)
        self._trial_ids_iterable = self.trial_id_vs_data.keys()
        self.trial_ids = tuple(self._trial_ids_iterable)

        self.trial_id_range_vs_data = (
            RangeDict(trial_id_range_vs_data) if trial_id_range_vs_data else None
        )
        self.metric_resolution = metric_resolution

        self.fps = fps
        if fps and not isinstance(fps, (float, int, dict)):
            msg = f"fps has to be float, int or dict, and not {type(fps)}"
            raise ValueError(msg)

        self.coordinate_data_format = str(coordinate_data_format).lower()
        self.label, self.func_inspect = label, func_inspect

        if self.coordinate_data_format == "deeplabcut":
            self.exp_id_vs_coordinate_sequences = {
                exp_id: dlc_obj
                for exp_id, dlc_obj in zip(
                    self._trial_ids_iterable,
                    DeepLabCutReader.init_many(
                        (
                            data["coordinate_data_path"]
                            for data in self.trial_id_vs_data.values()
                        ),
                        labels=self.trial_id_vs_data.keys(),
                        **data_import_kwargs,
                    ),
                )
            }
        else:
            msg = f"{self.coordinate_data_format} as a format for data ingestion has no implementation"
            raise NotImplemented(msg)

    def __getitem__(self, item):
        if self.trial_id_range_vs_data:
            return {**self.trial_id_vs_data[item], **self.trial_id_range_vs_data[item]}
        else:
            return self.trial_id_vs_data[item]

    @cached_property
    def length(self):
        return len(self.trial_ids)

    def exp_id_data_tqdm(self):
        return tqdm(
            ((exp_id, self[exp_id]) for exp_id in self.trial_ids), total=self.length
        )


class BaseTrial:
    second_tolerance = 0.35

    def __init__(
        self,
        coordinate_sequence: dict,
        unit_per_pixel: float,
        rigid_nodes_freezing: Union[Sequence[Union[str, int]], None] = None,
        movement_feature_point_label: Union[str, None] = None,
        video_path: Any = None,
        recording_resolution: Union[Sequence[int], None] = None,
        fps: Union[float, None] = None,
        label: Any = None,
        func_inspect: bool = False,
        inspect_image: Any = None,
    ):
        """
        :param coordinate_sequence: The coordinates of the subject across the frames in the video recording
        :param unit_per_pixel: Number defining the number of pixels that goes into one centimeter
        :param rigid_nodes_freezing: Nodes that should remain during freeze/immobility, most often due to fear.
        :param movement_feature_point_label: Label of the node that will be used to track general animal movement
        :param video_path: Path to trial video recording
        :param recording_resolution: Video resolution
        :param fps: Frames per second of video
        :param label: Experiment label
        :param func_inspect: If True, will generate inspection figures from functions that have support
        :param inspect_image: Image used for inspection
        :type coordinate_sequence: dict
        :type unit_per_pixel: float
        :type rigid_nodes_freezing: Sequence[Union[str, int]] (optional)
        :type movement_feature_point_label: str (optional)
        :type video_path: Any (optional)
        :type recording_resolution: Sequence[int] (optional)
        :type fps: float (optional)
        :type label: Any
        :type func_inspect: bool
        :type inspect_image: Any
        """

        if video_path:
            _frame, x_res, y_res, self.fps = get_video_data(video_path)
            self.recording_resolution = (x_res, y_res)
        else:
            self.recording_resolution = recording_resolution
            self.fps = fps

        self.unit_per_pixel = float(unit_per_pixel)
        self.label = label
        self.func_inspect = func_inspect
        self.inspect_image = inspect_image

        if self.recording_resolution:
            assert len(recording_resolution) == 2, recording_resolution
            self.horizontal_resolution = int(recording_resolution[0])
            self.vertical_resolution = int(recording_resolution[1])
            self.recording_resolution = (
                self.horizontal_resolution,
                self.vertical_resolution,
            )

        # coordinate_sequence must be a reader object, like DeepLabCutReader
        self.movement_feature_point_label = str(movement_feature_point_label)
        self.coordinate_sequence = coordinate_sequence
        self.coordinates_per_frame = self.coordinate_sequence[
            self.movement_feature_point_label
        ]

        self.experiment_seconds = self.coordinates_per_frame.shape[0] / self.fps

        self.motion = Motion(self.coordinates_per_frame, self.unit_per_pixel, self.fps)

        self._rigid_nodes_freezing = None
        self._frozen_boolean_index = None
        if rigid_nodes_freezing:
            self.rigid_nodes_freezing = rigid_nodes_freezing

    @cached_property
    def _frame_tolerance(self):
        return round(self.second_tolerance * self.fps)

    @property
    def rigid_nodes_freezing(self):
        return self._rigid_nodes_freezing

    @rigid_nodes_freezing.setter
    def rigid_nodes_freezing(self, value: Sequence):
        self._rigid_nodes_freezing = value
        if not value:
            return

        self._frozen_boolean_index = frozen_frames(
            self.fps,
            [
                displacement_by_frame(coordinate_sequence, remove_tails=False)
                * self.unit_per_pixel
                for coordinate_sequence in self.coordinate_sequence[
                    self.rigid_nodes_freezing
                ]
            ],
        )

    @property
    def frozen_boolean_index(self):
        if self._frozen_boolean_index is None:
            msg = "rigid_nodes_freezing has to be defined in order to compute frozen time data"
            raise AttributeError(msg)
        return self._frozen_boolean_index

    @cached_property
    def total_frozen_frames(self):
        return np.sum(self.frozen_boolean_index) / self.fps
