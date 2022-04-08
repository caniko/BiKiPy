from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from bikipy.perimeter.radial.utils import plot_circle
from bikipy.utils.misc import read_image


def draw_bikipy_circle(radius: float, img: Any = None):
    def on_press(event: Any):
        print(
            "%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f"
            % (
                "double" if event.dblclick else "single",
                event.button,
                event.x,
                event.y,
                event.xdata,
                event.ydata,
            )
        )
        if event.button:
            plot_circle((event.xdata, event.ydata), radius, ax)
            center = np.array((event.xdata, event.ydata))

    def on_pick(event: Any):
        """

        Handles the pick event - if an object has been picked, store a
        reference to it.  We do this by simply adding a reference to it
        named 'stored_pick' to the axes object.  Note that in python we
        can dynamically add an attribute variable (stored_pick) to an
        existing object - even one that is produced by a library as in this
        case

        Parameters
        ----------
        event
            Matplotlib event

        Returns
        -------

        """

        this_artist = event.artist  # the picked object is available as event.artist
        # print(this_artist) #For debug just to show you which object is picked
        plt.gca().picked_object = this_artist

    def on_key(event: Any):
        """
        Function to be bound to the key press event
        If the key pressed is delete and there is a picked object,
        remove that object from the canvas

        :param event:
        :return:
        """
        if event.key == "delete":
            if ax.picked_object:
                ax.picked_object.remove()
                ax.picked_object = None
                ax.figure.canvas.draw()

    fig, ax = plt.subplots()
    if img is not None:
        ax.imshow(read_image(img))

    center = None

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("key_press_event", on_key)

    ax.aspect = 1
    plt.tight_layout()

    plt.show()

    return center


draw_bikipy_circle(2)
