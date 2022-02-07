from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

lines = {1: None, 2: None}


def parallelogram_input(img: Any, reverse_y_axis: bool = False):
    def generate_parallelogram(line_seg_a, line_seg_b):
        # Sort based on magnitude
        a_argsorted_norms = np.argsort(np.linalg.norm(line_seg_a, axis=1))
        line_seg_a = line_seg_a[a_argsorted_norms]
        b_argsorted_norms = np.argsort(np.linalg.norm(line_seg_b, axis=1))
        line_seg_b = line_seg_b[b_argsorted_norms]

        # Visualize generate_parallellogram
        ax.plot(
            (line_seg_a[0][0], line_seg_b[0][0]),
            (line_seg_a[0][1], line_seg_b[0][1]),
            "-b",
            (line_seg_a[1][0], line_seg_b[1][0]),
            (line_seg_a[1][1], line_seg_b[1][1]),
            "-b",
        )

        # Generate gradient data
        a_mid = line_seg_a[0] + (line_seg_a[1] - line_seg_a[0]) / 2.0
        b_mid = line_seg_b[0] + (line_seg_b[1] - line_seg_b[0]) / 2.0
        ax.plot((a_mid[0], b_mid[0]), (a_mid[1], b_mid[1]), "-k")

    def draw_line(start_x: float, start_y: float):
        """
        function to draw lines - from matplotlib examples.  Note you don't need
        to keep a reference to the lines drawn, so I've removed the class as it
        is overkill for your purposes
        """

        end_x, end_y = plt.ginput(1)[0]
        line = ax.plot((start_x, end_x), (start_y, end_y))[0]
        line.set_pickradius(5)
        ax.figure.canvas.draw()

        return np.array(((start_x, start_y), (end_x, end_y)))

    def on_click(event: Any):
        """
        This implements click functionality.  If it's a double click do something,
        else ignore.
        Once in the double click block, if its a left click, wait for a further
        click and draw a line between the double click co-ordinates and that click
        (using ginput(1) - the 1 means wait for one mouse input - a higher number
        is used to get multiple clicks to define a polyline)

        """
        if event.dblclick and event.button == 1:
            global lines
            new_line = draw_line(event.xdata, event.ydata)
            if lines[1] is None:
                assert lines[2] is None, lines
                lines[1] = new_line
            elif lines[2] is None:
                lines[2] = new_line
                generate_parallelogram(lines[1], lines[2])

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

    global lines
    fig, ax = plt.subplots()
    img = Image.open(img)
    ax.imshow(img)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("key_press_event", on_key)

    ax.aspect = 1
    plt.tight_layout()

    plt.show()

    if reverse_y_axis:
        for key in lines:
            for i in range(2):
                lines[key][i][1] = img.size[1] - lines[key][i][1]

    result = np.array(lines.values())
    lines = {1: None, 2: None}

    return result
