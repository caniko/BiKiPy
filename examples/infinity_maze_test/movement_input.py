from datetime import datetime

import zmq
from matplotlib import pyplot as plt

from bikipy.perimeter.base import PolygonalPerimeter


def onclick(event):
    ix, iy = event.xdata, event.ydata
    string_coords = f"{ix} {iy},{datetime.now()}"
    print(f"x = {ix}, y = {iy}")

    socket.send(bytes(string_coords, "utf-8"))
    return string_coords


context = zmq.Context()

#  Socket to talk to server
socket = context.socket(zmq.PUSH)
socket.connect("tcp://localhost:5555")

perimeters = PolygonalPerimeter.from_coco(
    "./coco_annotations_2021-09-01-02-19-41.json", inspect_image="./maze_example.png"
)
fig, ax = plt.subplots()
PolygonalPerimeter.plot_perimeters(tuple(perimeters.values()), ax=ax)

cid = fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()
