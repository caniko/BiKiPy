from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

ROOT = Path("C:/Users/Can/Projects/Neuroscience/bikipy/examples/data/images")
img = cv2.imread(str(ROOT / "nort" / "before_47.png"), 0)

img = cv2.convertScaleAbs(img, alpha=1, beta=-2)

lowpass = 125

ret, thresh1 = cv2.threshold(img, lowpass, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, lowpass, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, lowpass, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, lowpass, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, lowpass, 255, cv2.THRESH_TOZERO_INV)

a, found = ndimage.label(thresh1)
b = ndimage.find_objects(a)

areas = []
for o in b:
    areas.append((o[0].stop - o[0].start) * (o[1].stop - o[1].start))
y_maze_obj_idx = np.where(np.array(areas) == np.sort(areas)[-1])[0][0]

stack_mask = np.zeros(
    (
        *thresh1.shape,
        3,
    ),
    dtype=np.uint8,
)

print(found)
print(ndimage.center_of_mass(a, labels=[i for i in range(6)]))

stack_mask[a != 0] = np.array((255, 255, 255), dtype=np.uint8)

titles = ["Original Image", "BINARY", "BINARY_INV", "TRUNC", "TOZERO", "STACK_MASK"]
images = [img, thresh1, thresh2, thresh3, thresh4, stack_mask]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
