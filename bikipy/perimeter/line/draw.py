from typing import Any, Optional

import matplotlib.pyplot as plt
from PIL import Image

from bikipy.core.typing import NDArrayFp64
from bikipy.perimeter import LinePerimeter
from bikipy.utils.video import get_video_data


def borders_on_image(
    img: Any,
    orientation: str,
    resolution: Optional[NDArrayFp64],
):
    """
    Initialize class using data from a sample frame/image

    Parameters
    ----------
    img
        object containing the image that will be used to determine the perimeter location
    orientation: str; {"vertical", "horizontal"}
        The orientation of the perimeter used for analysis
    resolution: NDArrayFp64
        The respective resolution of the frame.

    Returns
    -------
    tuple(greater_than_borders, less_than_borders)
    """

    if not resolution:
        img = Image.open(img)
        resolution = img.size

    plt.imshow(img)

    plt.title("Greater than border_vertices")
    greater_than_borders = [
        LinePerimeter(coordinate, orientation, logic=">", resolution=resolution) for coordinate in plt.ginput(0, 0)
    ]
    plt.title("Less than border_vertices")
    less_than_borders = [
        LinePerimeter(coordinate, orientation, logic="<", resolution=resolution) for coordinate in plt.ginput(0, 0)
    ]

    return greater_than_borders, less_than_borders


def draw_on_video_frame(video_path: str, orientation: str):
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
        The orientation of the perimeter used for analysis

    Returns
    -------
    bikipy.perimeter.linear.draw.borders_on_image call with frame from video
    """
    frame, x_res, y_res, _fps = get_video_data(video_path)

    return borders_on_image(frame, orientation, resolution=(x_res, y_res))
