import os
import cv2
import re


def get_video_data(video, frame_loc="middle"):
    cap = cv2.VideoCapture(video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if frame_loc == "middle":
        target_frame = frame_count / 2
    else:
        msg = "frame_loc can only be halfway; start; end,\n" "and not {}".format(
            frame_loc
        )
        raise ValueError(msg)

    cap.set(1, target_frame - 1)

    res, frame = cap.read()
    assert res, "Could not extract frame from media"

    return frame, height, width


def handle_video_data(video_path, file_format="mp4", frame_loc="middle"):
    filetype = re.search(r"\.\w+$", video_path)
    if filetype is not None:
        if video_path.endswith(".avi") or video_path.endswith(file_format):
            return get_video_data(os.path.abspath(video_path))
        else:
            msg = "Invalid filetype, {}".format(filetype)
            raise ValueError(msg)
    elif os.path.exists(video_path):
        video_list = [
            video
            for video in os.listdir(video_path)
            if video.endswith(".avi") or video.endswith(file_format)
        ]
        data_set = []
        for video_name in video_list:
            video_data = list(get_video_data(os.path.abspath(video_name)))
            video_data.append(video_name)
            # video_data = frame, height, width, video_name

            data_set.append(video_data)

        return data_set

    else:
        msg = "Media file is not defined"
        raise ValueError(msg)
