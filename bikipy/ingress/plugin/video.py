from functools import lru_cache

from pydantic import FilePath

from bikipy.core.video import VideoMetadata


@lru_cache
def video_file_path_to_value(file_path: FilePath, *args, **kwargs):
    return VideoMetadata(video_path=file_path)
