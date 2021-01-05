from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def detect(image, alpha):
    if not isinstance(image, np.ndarray):
        image = Path(image).resolve()
        assert image.exists()
        image = cv2.imread(str(image), 0)

    contrasted = cv2.convertScaleAbs(image, alpha=alpha)

    lowpass = 115

    ret, thresh = cv2.threshold(contrasted, lowpass, 255, cv2.THRESH_BINARY_INV)

    # labeled, objs = ndimage.semantic_label(np.mean(thresh, axis=2))
    # b = ndimage.find_objects(labeled)
    #
    # # labeled = ndimage.binary_erosion(labeled, iterations=1)
    #
    # stack_mask = np.zeros_like(thresh, dtype=np.uint8)
    # stack_mask[labeled == 2] = np.array((255, 255, 255), dtype=np.uint8)
    # stack_mask[labeled == 3] = np.array((255, 255, 255), dtype=np.uint8)
    # plt.imshow(stack_mask)

    plt.imshow(thresh, "gray")
    plt.title("BINARY_INV")
    plt.xticks([]), plt.yticks([])

    plt.show()


if __name__ == "__main__":
    from bikipy.utils.video import get_video_data

    ROOT = Path("C:/Users/Can/Projects/Neuroscience/Imen/data/nort/0_before_02.06.2020")
    video = ROOT / "Test 47.mp4"

    frame, _width, _height, _fps = get_video_data(video, "start")

    alpha = 1

    detect(frame, alpha)
