import cv2
import matplotlib.pyplot as plt


def onclick(event):
    ix, iy = event.xdata, event.ydata
    string_coords = f"{ix} {iy}"
    print(f"x = {ix}, y = {iy}")

    return string_coords


fig, ax = plt.subplots()
ax.imshow(cv2.imread("maze_example.png"))

cid = fig.canvas.mpl_connect("button_press_event", onclick)
