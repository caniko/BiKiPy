import os
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from dlca.video_analysis import get_video_data
from settings import DATA_FOLDER_NAME

import cv2

"""
This tool gives the user the ability to choose area that they want to crop
and then crops the video frame by frame.

The video created is mp4v (*".mp4").

The command line to run this in terminal:
python video_crop.py filename
"""


def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    global x1, x2, y1, y2
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))


def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


# choose frame
frame_example = get_video_data(sys.argv[1], path=DATA_FOLDER_NAME)[0]
fig, current_ax = plt.subplots()  # make a new plotting range
plt.imshow(frame_example)

# drawtype is 'box' or 'line' or 'none'
toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],
                                       # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
plt.connect('key_press_event', toggle_selector)
plt.show()


# cropped video path
CROPPED_VIDEOS = os.path.join(DATA_FOLDER_NAME, "cropped_videos")
if not os.path.exists(CROPPED_VIDEOS):
    os.mkdir(CROPPED_VIDEOS)
video_path = os.path.join(DATA_FOLDER_NAME, sys.argv[1])


# cropping video of interest
stream = cv2.VideoCapture(video_path)
outputpath = os.path.join(CROPPED_VIDEOS,
                          "{}_cropped.mp4".format(sys.argv[1][:-4]))
size = (int(round(x2-x1)), int(round(y2-y1)))
codec = cv2.VideoWriter_fourcc(*"mp4v")
cropped_video = cv2.VideoWriter()
cropped_video.open(outputpath, fourcc=codec, fps=30, frameSize=size,
                   isColor=True)

# loading and cropping video frame by frame
while True:
    grabbed, frame = stream.read()
    if not grabbed:
        break
    cropped_frame = frame[int(round(y1)):int(round(y2)), int(round(x1)):int(round(x2))]
    #cv2.imshow('cropped', cropped_frame) # to show each frame
    #cv2.waitKey(int(round(1000/30))) # to show frames with ca. 30fps
    cropped_video.write(cropped_frame)
stream.release()

