try:
    import cv2
except ModuleNotFoundError as e:
    msg = (
        "opencv-python is required to get data from video, type: "
        "pip install opencv-python=4.2.0.34"
    )
    raise ModuleNotFoundError(msg) from e


def get_video_data(video_path, frame_loc="middle"):
    """ Get a frame from a given relative location, and resolution info of video

def get_video_data(video, frame_loc="middle"):
    cap = cv2.VideoCapture(video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if frame_loc == "middle":
        target_frame = frame_count / 2
    else:
        msg = f"frame_loc can only be halfway; start; end,\n" "and not {frame_loc}"
        raise ValueError(msg)

    cap.set(1, target_frame - 1)

    res, frame = cap.read()
    assert res, "Could not extract frame from media"

    return frame, height, width
