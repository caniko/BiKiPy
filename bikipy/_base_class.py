from abc import ABC
from datetime import date, datetime
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Literal, Optional, Union

import compress_pickle
import numpy as np
from pydantic import BaseModel, DirectoryPath, Extra, Field, FilePath

from bikipy.utils.typing import NDArray, OptionalPathTyping, PathTyping
from bikipy.utils.video import get_video_data


class BikipyBase(BaseModel, ABC):
    int_id: Optional[int] = None
    label: Optional[str] = None
    group_label: Optional[str] = None
    timestamp: Union[date, datetime] = Field(default_factory=datetime.utcnow)
    save_root: Optional[DirectoryPath] = None

    category: ClassVar[Optional[str]] = None

    class Config:
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True
        extra = Extra.allow
        keep_untouched = (cached_property,)

    def save(self, save_root: OptionalPathTyping = None):
        save_root = Path(save_root or self.save_root)
        assert save_root
        compress_pickle.dump(
            self, save_root / f"pickle_{self.category}_{self.timestamp}.lzma"
        )

    @property
    def _hash_key(self):
        return (
            self.best_id,
            self.group_label,
            self.timestamp,
            self.category,
        )

    def __hash__(self):
        return sum(hash(key) for key in self._hash_key)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._hash_key == other._hash_key
        return self._hash_key == other

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def _inquire_timestamp_attribute(obj):
        if not hasattr(obj, "timestamp"):
            msg = "Cannot perform inequality operations on object without the timestamp attribute"
            raise AttributeError(msg)

    def __lt__(self, other):
        self._inquire_timestamp_attribute(other)
        return self.timestamp < other.timestamp

    def __le__(self, other):
        self._inquire_timestamp_attribute(other)
        return self.timestamp <= other.timestamp

    def __gt__(self, other):
        self._inquire_timestamp_attribute(other)
        return self.timestamp > other.timestamp

    def __ge__(self, other):
        self._inquire_timestamp_attribute(other)
        return self.timestamp >= other.timestamp

    @property
    def best_id(self):
        return self.label or self.int_id or None


class VideoMetaDataMixin(BaseModel):
    video_path: Optional[FilePath] = None
    manual_recording_resolution: Optional[NDArray[Literal[np.int16]]] = None
    manual_fps: Optional[float] = None

    @cached_property
    def video_metadata_can_be_defined(self):
        return self.video_path or (self.manual_recording_resolution and self.manual_fps)

    @cached_property
    def _video_metadata(self) -> tuple:
        error_msg = (
            "Either video_path or video metadata needs to be exclusively defined."
        )
        if np.any(self.manual_recording_resolution) and self.manual_fps:
            if self.video_path:
                raise ValueError(error_msg)
            fps = self.manual_fps
            recording_resolution = self.manual_recording_resolution
        elif self.video_path:
            _frame, horizontal_resolution, vertical_resolution, fps = get_video_data(
                self.video_path
            )
            recording_resolution = (horizontal_resolution, vertical_resolution)
        else:
            raise ValueError(error_msg)
        return np.array(recording_resolution, dtype=np.int16), fps

    @property
    def _video_metadata_dict_manual_format(self):
        return {
            "manual_recording_resolution": self.recording_resolution,
            "manual_fps": self.fps,
        }

    @property
    def recording_resolution(self) -> np.ndarray:
        return self._video_metadata[0]

    @property
    def horizontal_resolution(self):
        return self.recording_resolution[0]

    @property
    def vertical_resolution(self):
        return self.recording_resolution[1]

    @property
    def fps(self):
        return self._video_metadata[1]
