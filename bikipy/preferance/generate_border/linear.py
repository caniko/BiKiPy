from typing import Union, AnyStr, Any
from collections.abc import Sequence
import matplotlib.pyplot as plt
from PIL import Image

from bikipy.preferance.border import LineBorder


def borders_on_image(
    img: Any,
    orientation: AnyStr,
    resolution: Union[Sequence, None] = None,
):
    """
    Initialize class using data from a sample frame/image

    Parameters
    ----------
    img
        object containing the image that will be used to determine the border location
    orientation: str; {"vertical", "horizontal"}
        The orientation of the border used for analysis
    resolution: Sequence
        The respective resolution of the frame.

    Returns
    -------
    tuple(greater_than_borders, less_than_borders)
    """

    if not resolution:
        img = Image.open(img)
        resolution = img.size

    plt.imshow(img)

    plt.title("Greater than borders")
    greater_than_borders = [
        LineBorder(coordinate, orientation, logic=">", resolution=resolution)
        for coordinate in plt.ginput(0, 0)
    ]
    plt.title("Less than borders")
    less_than_borders = [
        LineBorder(coordinate, orientation, logic="<", resolution=resolution)
        for coordinate in plt.ginput(0, 0)
    ]

    return greater_than_borders, less_than_borders


def borders_on_video(video_path: AnyStr, orientation: AnyStr):
    """Initialize class using data from a sample video file

    Parameters
    ----------
    video_path: str
        The name of the video file in local directory to be used for analysis.
        Required if orientation == 'lasso' and frame == None.
        None: No action

        str:  The video file matching the string will be selected.
              File extension must be included.

        True: If there is only one video file, it will be selected.
    orientation: str; {"vertical", "horizontal"}
        The orientation of the border used for analysis

    Returns
    -------
    bikipy.preferance.border.borders_on_image call
    """
    from bikipy.utils.video import get_video_data

    frame, x_res, y_res = get_video_data(video_path)

    return borders_on_image(frame, orientation, resolution=(x_res, y_res))
