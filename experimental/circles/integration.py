from abc import ABC, abstractmethod
from uuid import uuid4

import cv2
from pydantic import FilePath

from PyQt5 import QtWidgets

from examples.circles.mpl_widget import MplWidget


class CircleAnnotatorQtBase(ABC):
    def __init__(self, overlay_image: FilePath = "/home/can/Pictures/PythonHaikoHeroku.png"):
        self.widget: MplWidget
        self.horizontalSlider: QtWidgets.QSlider

        self.circles = {}
        self.overlay_image = cv2.imread(overlay_image)

        self.app = QtWidgets.QApplication([])
        self.Dialog = QtWidgets.QDialog()

        self.start()

    @abstractmethod
    def setupUi(self, Dialog):
        ...

    def start(self):
        self.setupUi(self.Dialog)

        self.horizontalSlider.setMinimum(50)
        self.horizontalSlider.setMaximum(round(min(self.overlay_image.shape[:2]) / 4.0))

        self.Dialog.show()
        self.app.exec_()

    def setup_signals(self):
        self.widget.canvas.mpl_connect('button_press_event', self.mousePressEvent)

    def mousePressEvent(self, e):
        circle_args = [e.xdata, e.ydata, self.horizontalSlider.value()]
        self.widget.draw_circle(*circle_args)
        self.circles[uuid4()] = circle_args
