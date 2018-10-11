from os import listdir
import cv2


def get_video_data(filename, frame_loc='middle', path='.'):
    if isinstance(filename, str):
        user_video = filename
    elif filename is True:
        video_list = [video for video in listdir(path) if
                      video.endswith('.avi') or video.endswith('.mp4')]
        user_video = video_list[0]
    else:
        msg = 'Media file is not defined'
        raise ValueError(msg)

    cap = cv2.VideoCapture(user_video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if frame_loc == 'middle':
        target_frame = frame_count / 2
    else:
        msg = 'frame_loc can only be halfway; start; end,\n' \
              'and not {}'.format(frame_loc)
        raise AttributeError(msg)

    cap.set(1, target_frame - 1)

    res, frame = cap.read()
    assert res, 'Could not extract frame from media'

    return frame, height, width
