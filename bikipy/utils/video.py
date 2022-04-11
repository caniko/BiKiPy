from pathlib import Path
from typing import Union

import cv2

from bikipy.utils.typing import PathTyping


def get_video_data(video_path: PathTyping, frame_time: Union[str, int, None] = None):
    """
    Get a frame from a given relative location, and resolution info of video

    Parameters
    ----------
    video_path: str
        Path to video to be analysed
    frame_time:
        Relative location of the frame used for reference in analysis

    Returns
    -------
    Tuple: (frame, height, width, fps)
    """
    video_path = Path(video_path).resolve()
    assert video_path.exists()

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    msg = f"frame_time can only be defined as halfway, start, end, or integer; " f"not {type(frame_time)}"
    if frame_time:
        if isinstance(frame_time, str):
            if (frame_time := frame_time.lower()) == "middle":
                target_frame_index = round(frame_count / 2.0)
            elif frame_time == "start" or frame_time == "beginning":
                target_frame_index = 0
            elif frame_time == "end":
                target_frame_index = frame_count
            else:
                raise ValueError(msg)
        elif isinstance(frame_time, int):
            target_frame_index = frame_time
        else:
            raise TypeError(msg)

        cap.set(1, target_frame_index - 1)
        res, frame = cap.read()

        assert res, f"Could not extract frame from media, {video_path}"
    else:
        frame = None

    return (
        frame,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FPS)),
    )
