from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from matplotlib.axes import Axes
from pydantic import FilePath, validate_arguments
from pydantic_numpy import NDArray
from pydantic_numpy.dtype import NDArrayFp64, NDArrayUint8

from bikipy.utils.plot.io import ax_imshow_gray


def save_plt_fig_cv(figure, save_path: Path) -> None:
    figure.canvas.draw()
    b = figure.get_window_extent()

    img = np.array(figure.canvas.buffer_rgba(), dtype=np.uint8)
    img = img[int(b.y0) : int(b.y1), int(b.x0) : int(b.x1), :]
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

    cv2.imwrite(str(save_path.with_suffix(".png")), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def read_image(
    image: FilePath | NDArrayUint8, imread_flagg: Optional[list] = None
) -> np.ndarray[int, np.dtype[np.uint8]]:
    if isinstance(image, (Path, str)):
        image_path = Path(image).resolve()
        assert image_path.exists(), image_path
        image = read_image_from_path(image_path, flags=imread_flagg)
    else:
        assert isinstance(image, NDArrayFp64), f"image must be either path or NDArrayFp64, but got:\n{image}"

    return image


@validate_arguments
@lru_cache
def read_image_from_path(
    image_path: FilePath, imread_flagg: Optional[list] = None
) -> np.ndarray[int, np.dtype[np.uint8]]:
    return cv2.imread(str(image_path))


def axis_frame_imshow(ax: Axes, image: NDArray):
    ax.autoscale(enable=True)

    ax_imshow_gray(ax, image)

    ax.invert_yaxis()
    # ax.set_aspect("equal", adjustable="box")

    return ax
