import matplotlib.pyplot as plt
import numpy as np

from bikipy.behaviour.rectangle import gaussian_scoring_field

resolution = (1000, 2000)
my_pic = np.empty(resolution, dtype=np.uint8)

model = gaussian_scoring_field(resolution)

r = range(500)
for x in r:
    for y in r:
        my_pic[x, y] = round(model(x, y) * 255)


plt.imshow(my_pic)
plt.show()
