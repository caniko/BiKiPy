import numpy as np

sides = np.array([(1, 1), (0, 1), (1, 0), (0, 0)])
argsorted_x, argsorted_y = np.argsort(sides.T, axis=1)

vertical_side_a = argsorted_x[:2]
vertical_side_b = argsorted_x[2:]

horizontal_side_a = argsorted_y[2:]
horizontal_side_b = argsorted_y[:2]

for corner in vertical_side_a:
    if corner in horizontal_side_b:
        down_left = sides[corner]
        (down_right,) = sides[horizontal_side_b[horizontal_side_b != corner]]

    elif corner in horizontal_side_a:
        up_left = sides[corner]
        (up_right,) = sides[horizontal_side_a[horizontal_side_a != corner]]

print(1)
