from datetime import datetime

import zmq
import cv2
import matplotlib.pyplot as plt


def onclick(event):
    ix, iy = event.xdata, event.ydata
    string_coords = f"{ix} {iy}"
    print(f"x = {ix}, y = {iy}")

    socket.send(bytes(string_coords, "utf-8"))
    return (string_coords, str(datetime.now())).join(",")


context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.PUSH)
socket.connect("tcp://localhost:5555")

fig, ax = plt.subplots()
ax.imshow(cv2.imread("maze_example.png"))

cid = fig.canvas.mpl_connect("button_press_event", onclick)
