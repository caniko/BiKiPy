from pathlib import Path

import cv2


def get_video_data(video_path, frame_time="middle"):
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

    frame_time = frame_time.lower()

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_time == "middle":
        target_frame = frame_count / 2.0
    elif frame_time == "start" or frame_time == "beginning":
        target_frame = 0
    elif frame_time == "end":
        target_frame = frame_count
    else:
        msg = f"frame_time can only be halfway; start; end,\n" f"and not {frame_time}"
        raise ValueError(msg)

    cap.set(1, target_frame - 1)

    res, frame = cap.read()
    assert res, f"Could not extract frame from media, {video_path}"

    return (
        frame,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FPS)),
    )
