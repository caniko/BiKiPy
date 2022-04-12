from typing import Iterable, Sequence

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


class MplWidget(QWidget):
    def __init__(self, overlay_image: np.array, parent=None):

        QWidget.__init__(self, parent)

        self.overlay_image = overlay_image

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.overlay_image)

        self.canvas = FigureCanvasQTAgg(self.fig)

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)

        self.setLayout(vertical_layout)

    def re_draw(self, circle_args_set: Iterable[list[float, float, float]]):
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.overlay_image)

        for circle_args in circle_args_set:
            self.draw_circle(*circle_args)

        self.canvas = FigureCanvasQTAgg(self.fig)

    def draw_circle(self, x_trans: float, y_trans: float, radius: float):
        theta = np.linspace(0, 2.0 * np.pi, 150)

        a = radius * np.cos(theta) + x_trans
        b = radius * np.sin(theta) + y_trans

        self.ax.plot(a, b)
