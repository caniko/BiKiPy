from pathlib import Path

import cv2
import numpy as np


def save_plt_fig_cv(figure, save_path: Path) -> None:
    figure.canvas.draw()
    b = figure.get_window_extent()

    img = np.array(figure.canvas.buffer_rgba(), dtype=np.uint8)
    img = img[int(b.y0) : int(b.y1), int(b.x0) : int(b.x1), :]
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

    cv2.imwrite(str(save_path.with_suffix(".png")), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
