import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MplWidget(QWidget):
    def __init__(self, parent=None):

        QWidget.__init__(self, parent)

        self.canvas = FigureCanvasQTAgg(Figure())

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)

        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)

    def draw_circle(self):
        theta = np.linspace(0, 2.0 * np.pi, 150)

        radius = 0.4

        a = radius * np.cos(theta)
        b = radius * np.sin(theta)

        figure, axes = plt.subplots(1)

        axes.plot(a, b)
        axes.set_aspect(1)

        plt.title("Parametric Equation Circle")
        plt.show()
