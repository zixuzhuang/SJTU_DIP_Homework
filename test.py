import os
import sys

import cv2
import numpy as np
import pyqtgraph as pg
# from imageProcess import (CBR_func, OBR_func, conditional_dilation,
#                           thresholdMaxEntropy, thresholdOTSU)
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt5.uic import loadUi
from scipy import ndimage
from skimage import filters, img_as_ubyte, io, morphology


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = loadUi("./MainWindow.ui", self)

        # ### Projcet 1
        # self.ui.RawImageView_1.getHistogramWidget().setVisible(False)
        # self.ui.RawImageView_1.ui.menuBtn.setVisible(False)
        # self.ui.RawImageView_1.ui.roiBtn.setVisible(False)
        # self.ui.GrayImageView_1.getHistogramWidget().setVisible(False)
        # self.ui.GrayImageView_1.ui.menuBtn.setVisible(False)
        # self.ui.GrayImageView_1.ui.roiBtn.setVisible(False)
        # self.ui.BinaryImageView_1.getHistogramWidget().setVisible(False)
        # self.ui.BinaryImageView_1.ui.menuBtn.setVisible(False)
        # self.ui.BinaryImageView_1.ui.roiBtn.setVisible(False)

        # self.ui.OpenButton_1.clicked.connect(self.openImage_1)
        # self.ui.OTSUButton.clicked.connect(self.binary_otsu)
        # self.ui.EntropyButton.clicked.connect(self.binary_entropy)
        # self.ui.ThresholdSlider.valueChanged.connect(self.binary_thresh)
        # self.ui.ThresholdSpin.valueChanged.connect(self.binary_thresh)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
