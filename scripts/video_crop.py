from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
from settings import DATA_FOLDER_NAME
from dlca.video_analysis import get_video_data
import sys
import os
from moviepy.editor import *


"""
moviepy requires setuptools and ex_setup already installed

Do a mouseclick somewhere, move the mouse to some destination, release
the button.  This class gives click- and release-events and also draws
a line or a box from the click-point to the actual mouse position
(within the same axes) until the button is released.  Within the
method 'self.ignore()' it is checked whether the button from eventpress
and eventrelease are the same.

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

# crop frame
frame = get_video_data(sys.argv[1], path=DATA_FOLDER_NAME)[0]
fig, current_ax = plt.subplots()  # make a new plotting range
plt.imshow(frame)

# drawtype is 'box' or 'line' or 'none'
toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
plt.connect('key_press_event', toggle_selector)
plt.show()


#crop video - WIP
CROPPED_VIDEOS = os.path.join(DATA_FOLDER_NAME, "cropped_videos")
if not os.path.exists(CROPPED_VIDEOS):
    os.mkdir(CROPPED_VIDEOS)

video_path = os.path.join(DATA_FOLDER_NAME, sys.argv[1])
original_video = VideoFileClip(video_path)
cropped_video = vfx.crop(original_video, x1=x1, y1=y1, x2=x2, y2=y2)
cropped_video.release()
