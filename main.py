import os
import sys
from functools import partial

import cv2
import numpy as np
import pyqtgraph as pg
# from imageProcess import (CBR_func, OBR_func, conditional_dilation,
#                           thresholdMaxEntropy, thresholdOTSU)
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt5.uic import loadUi
from scipy import ndimage
from skimage import filters, img_as_ubyte, io, morphology

from projects.project1 import thresholdMaxEntropy, thresholdOTSU
from projects.project6 import CBR_func, OBR_func, conditional_dilation


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = loadUi("./MainWindow.ui", self)
        self.init()

        # Main Part
        self.ui.OpenButton.clicked.connect(self.open)
        self.ui.ClearButton.clicked.connect(self.clear)

        # Project 1
        self.ui.OTSUButton.clicked.connect(self.proj1_otsu)
        self.ui.MEButton.clicked.connect(self.proj1_maxEntropy)
        self.ui.P1T.valueChanged.connect(self.proj1_slideSlider)
        self.ui.ThresholdSlider.valueChanged.connect(self.proj1_slideSlider)

        # Project 2
        self.ui.RXButton.clicked.connect(partial(self.proj2_edge, "roberts", "x"))
        self.ui.RYButton.clicked.connect(partial(self.proj2_edge, "roberts", "y"))
        self.ui.PXButton.clicked.connect(partial(self.proj2_edge, "prewitt", "x"))
        self.ui.PYButton.clicked.connect(partial(self.proj2_edge, "prewitt", "y"))
        self.ui.SXButton.clicked.connect(partial(self.proj2_edge, "sobel", "x"))
        self.ui.SYButton.clicked.connect(partial(self.proj2_edge, "sobel", "y"))
        self.ui.GaussianButton.clicked.connect(partial(self.proj2_noise, "gaussian"))
        self.ui.MedianButton.clicked.connect(partial(self.proj2_noise, "median"))

        # Project 3
        self.ui.P3D.clicked.connect(partial(self.proj3_binary_operation, "dilation"))
        self.ui.P3E.clicked.connect(partial(self.proj3_binary_operation, "erosion"))
        self.ui.P3O.clicked.connect(partial(self.proj3_binary_operation, "opening"))
        self.ui.P3C.clicked.connect(partial(self.proj3_binary_operation, "closing"))
        self.ui.P3T.valueChanged.connect(partial(self.updata_binary, "proj3"))

        # Project 4
        self.ui.P4D.clicked.connect(self.proj4_distance_transform)
        self.ui.P4S.clicked.connect(self.proj4_skeleton)
        self.ui.P4R.clicked.connect(self.proj4_skeleton_restore)
        self.ui.P4T.valueChanged.connect(partial(self.updata_binary, "proj4"))

        # Project 5
        self.ui.P5D.clicked.connect(partial(self.proj5_operation, "dilation"))
        self.ui.P5E.clicked.connect(partial(self.proj5_operation, "erosion"))
        self.ui.P5O.clicked.connect(partial(self.proj5_operation, "opening"))
        self.ui.P5C.clicked.connect(partial(self.proj5_operation, "closing"))

        # Project 6
        self.ui.P6T.valueChanged.connect(partial(self.updata_binary, "proj6"))
        self.ui.P6E.clicked.connect(self.proj6_edge_detect)
        self.ui.P6G.clicked.connect(self.proj6_gradient)
        self.ui.P6R.clicked.connect(self.proj6_reconst)
        return

    def init(self):
        # Main Part
        self.ui.RawView.getHistogramWidget().setVisible(False)
        self.ui.RawView.ui.menuBtn.setVisible(False)
        self.ui.RawView.ui.roiBtn.setVisible(False)

        self.ui.GreyView.getHistogramWidget().setVisible(False)
        self.ui.GreyView.ui.menuBtn.setVisible(False)
        self.ui.GreyView.ui.roiBtn.setVisible(False)

        # Project 1
        self.ui.BinaryView.getHistogramWidget().setVisible(False)
        self.ui.BinaryView.ui.menuBtn.setVisible(False)
        self.ui.BinaryView.ui.roiBtn.setVisible(False)
        self.ui.P1T.setValue(0)
        self.ui.ThresholdSlider.setValue(0)
        self.ui.HistView.setMouseEnabled(x=False, y=False)

        # Project 2
        self.ui.P2OView.getHistogramWidget().setVisible(False)
        self.ui.P2OView.ui.menuBtn.setVisible(False)
        self.ui.P2OView.ui.roiBtn.setVisible(False)
        self.ui.KernelBox.setValue(5)
        self.ui.SigmaBox.setValue(2.0)

        # Project 3
        self.ui.P3OView.getHistogramWidget().setVisible(False)
        self.ui.P3OView.ui.menuBtn.setVisible(False)
        self.ui.P3OView.ui.roiBtn.setVisible(False)
        self.ui.P3BView.getHistogramWidget().setVisible(False)
        self.ui.P3BView.ui.menuBtn.setVisible(False)
        self.ui.P3BView.ui.roiBtn.setVisible(False)

        # Project 4
        self.ui.P4OView.getHistogramWidget().setVisible(False)
        self.ui.P4OView.ui.menuBtn.setVisible(False)
        self.ui.P4OView.ui.roiBtn.setVisible(False)
        self.ui.P4BView.getHistogramWidget().setVisible(False)
        self.ui.P4BView.ui.menuBtn.setVisible(False)
        self.ui.P4BView.ui.roiBtn.setVisible(False)

        # Project 5
        self.ui.P5OView.getHistogramWidget().setVisible(False)
        self.ui.P5OView.ui.menuBtn.setVisible(False)
        self.ui.P5OView.ui.roiBtn.setVisible(False)

        # Project 6
        self.ui.P6OView.getHistogramWidget().setVisible(False)
        self.ui.P6OView.ui.menuBtn.setVisible(False)
        self.ui.P6OView.ui.roiBtn.setVisible(False)
        self.ui.P6BView.getHistogramWidget().setVisible(False)
        self.ui.P6BView.ui.menuBtn.setVisible(False)
        self.ui.P6BView.ui.roiBtn.setVisible(False)
        return

    def open(self):
        # Get image name
        img_name, _ = QFileDialog.getOpenFileName(self, "Open An Image File", os.getcwd(), "image file(*.jpg *.png *.bmp);;All Files(*)")
        if img_name == "":
            return 0
        else:
            # Load image
            self.raw_img = img_as_ubyte(io.imread(img_name))
            # Clear last image
            self.clear()
            # Display raw image
            axis = (1, 0, 2) if self.raw_img.ndim == 3 else (1, 0)
            self.ui.RawView.setImage(np.transpose(self.raw_img, axis))
            # Display grey image
            self.grey_img = self.raw_img if self.raw_img.ndim == 2 else cv2.cvtColor(self.raw_img, cv2.COLOR_RGB2GRAY)
            self.ui.GreyView.setImage(np.transpose(self.grey_img, (1, 0)))
            # Cal binary image
            self.init_threshold = int(filters.threshold_otsu(self.grey_img))
            self.binary_img = (self.grey_img >= self.init_threshold).astype(np.int_)
            self.binary_img_3 = self.binary_img.copy()
            self.binary_img_4 = self.binary_img.copy()
            self.binary_img_6 = self.binary_img.copy()
            # Display histogram in P1
            hist, _ = np.histogram(self.grey_img, bins=np.arange(257))
            self.ui.HistView.getPlotItem().plot(np.arange(257), hist, stepMode=True, fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150))
            # Display binary image in P3 P4 P6
            self.ui.P3T.setValue(self.init_threshold)
            self.ui.P4T.setValue(self.init_threshold)
            self.ui.P6T.setValue(self.init_threshold)
            self.ui.P3BView.clear()
            self.ui.P4BView.clear()
            self.ui.P6BView.clear()
            self.ui.P3BView.setImage(np.transpose(self.binary_img_3, (1, 0)), levels=[0, 1])
            self.ui.P4BView.setImage(np.transpose(self.binary_img_4, (1, 0)), levels=[0, 1])
            self.ui.P6BView.setImage(np.transpose(self.binary_img_6, (1, 0)), levels=[0, 1])
        return

    def clear(self):
        # Main Part
        self.ui.RawView.clear()
        self.ui.GreyView.clear()

        # Program 1
        self.ui.BinaryView.clear()
        self.ui.HistView.clear()
        self.ui.P1T.setValue(0)
        self.ui.ThresholdSlider.setValue(0)

        # Program 2
        self.ui.P2OView.clear()

        # Program 3
        self.ui.P3OView.clear()
        self.ui.P3BView.clear()
        # Program 4
        self.ui.P4OView.clear()
        self.ui.P4BView.clear()
        # Program 5
        self.ui.P5OView.clear()
        # Program 6
        self.ui.P6OView.clear()
        self.ui.P6BView.clear()

        return

    def updata_binary(self, proj):
        projs = {
            "proj3": [self.ui.P3BView, self.binary_img_3, self.ui.P3T],
            "proj4": [self.ui.P4BView, self.binary_img_4, self.ui.P4T],
            "proj6": [self.ui.P6BView, self.binary_img_6, self.ui.P6T],
        }
        threshold = projs[proj][2].value()
        if proj == "proj3":
            self.binary_img_3 = (self.grey_img >= threshold).astype(np.int_)
        elif proj == "proj4":
            self.binary_img_4 = (self.grey_img >= threshold).astype(np.int_)
        elif proj == "proj6":
            self.binary_img_6 = (self.grey_img >= threshold).astype(np.int_)
        print(proj, threshold, np.sum(projs[proj][1]))
        projs[proj][0].clear()
        projs[proj][0].setImage(np.transpose(projs[proj][1], (1, 0)), levels=[0, 1])
        return

    def proj1_otsu(self):
        threshold = int(filters.threshold_otsu(self.grey_img))
        self.ui.ThresholdSlider.setValue(int(threshold))
        self.ui.P1T.setValue(int(threshold))
        return

    def proj1_maxEntropy(self):
        threshold = thresholdMaxEntropy(self.grey_img)
        self.ui.ThresholdSlider.setValue(int(threshold))
        self.ui.P1T.setValue(int(threshold))
        return

    def proj1_slideSlider(self, threshold):
        try:
            self.ui.HistView.getPlotItem().removeItem(self.threshLine)
        except:
            pass
        binary_image = self.grey_img >= threshold
        self.ui.BinaryView.setImage(np.transpose(binary_image, (1, 0)), levels=[0, 1])
        self.threshLine = self.ui.HistView.getPlotItem().addLine(x=threshold)
        self.ui.P1T.setValue(int(threshold))
        return

    def proj2_edge(self, method, direction):
        methods = {
            "robertsx": filters.roberts_pos_diag,
            "robertsy": filters.roberts_neg_diag,
            "prewittx": filters.prewitt_v,
            "prewitty": filters.prewitt_h,
            "sobelx": filters.sobel_v,
            "sobely": filters.sobel_h,
        }
        self.ui.P2OView.clear()
        self.ui.P2OView.setImage(np.transpose(methods[method + direction](self.grey_img), (1, 0)))
        return

    def proj2_noise(self, method):
        kernel = self.ui.KernelBox.value()
        sigma = self.ui.SigmaBox.value()
        if method == "gaussian":
            img = cv2.GaussianBlur(self.grey_img, (kernel, kernel), sigma)
        elif method == "median":
            kernel = np.ones((kernel, kernel))
            img = filters.median(self.grey_img, selem=kernel)
        self.ui.P2OView.clear()
        self.ui.P2OView.setImage(np.transpose(img, (1, 0)))
        return

    def proj3_binary_operation(self, method):
        methods = {
            "dilation": morphology.dilation,
            "erosion": morphology.erosion,
            "opening": morphology.opening,
            "closing": morphology.closing,
        }
        kernel_size = self.ui.P3K.value()
        kernel = morphology.disk(kernel_size)
        img = methods[method](self.binary_img_3, selem=kernel)
        self.ui.P3OView.clear()
        self.ui.P3OView.setImage(np.transpose(img, (1, 0)))

    def proj4_distance_transform(self):
        method = self.ui.P4DS.currentText()
        methods = {
            "City Block": "taxicab",
            "Chessboard": "chessboard",
        }
        if method == "Eculidean":
            img = ndimage.distance_transform_edt(self.binary_img_4)
        else:
            img = ndimage.distance_transform_cdt(self.binary_img_4, metric=methods[method])
        self.ui.P4OView.clear()
        self.ui.P4OView.setImage(np.transpose(img, (1, 0)))

    def proj4_skeleton(self):
        img, dist = morphology.medial_axis(self.binary_img_4, return_distance=True)
        self.skeleton = img
        self.dist = dist
        self.ui.P4OView.clear()
        self.ui.P4OView.setImage(np.transpose(img, (1, 0)))

    def proj4_skeleton_restore(self):
        assert self.skeleton is not None
        assert self.dist is not None
        img = morphology.reconstruction(self.skeleton, self.dist)
        self.ui.P4OView.clear()
        self.ui.P4OView.setImage(np.transpose(img, (1, 0)))

    def proj5_operation(self, method):
        methods = {
            "dilation": morphology.dilation,
            "erosion": morphology.erosion,
            "opening": morphology.opening,
            "closing": morphology.closing,
        }
        kernel_size = self.ui.P5K.value()
        kernel = morphology.disk(kernel_size)
        img = methods[method](self.grey_img, selem=kernel)
        self.ui.P5OView.clear()
        self.ui.P5OView.setImage(np.transpose(img, (1, 0)))

    def proj6_edge_detect(self):
        method = self.ui.P6EL.currentText()
        if method == "Standard":
            img = img_as_ubyte(morphology.dilation(self.binary_img_6)) - img_as_ubyte(morphology.erosion(self.binary_img_6))
        elif method == "External":
            img = img_as_ubyte(morphology.dilation(self.binary_img_6)) - img_as_ubyte(self.binary_img_6)
        elif method == "Internal":
            img = img_as_ubyte(self.binary_img_6) - img_as_ubyte(morphology.erosion(self.binary_img_6))
        self.ui.P6OView.clear()
        self.ui.P6OView.setImage(np.transpose(img, (1, 0)))

    def proj6_gradient(self):
        method = self.ui.P6EL.currentText()
        if method == "Standard":
            img = (morphology.dilation(self.grey_img) - morphology.erosion(self.grey_img)) / 2
        elif method == "External":
            img = (morphology.dilation(self.grey_img) - self.grey_img) / 2
        elif method == "Internal":
            img = (self.grey_img - morphology.erosion(self.grey_img)) / 2
        self.ui.P6OView.clear()
        self.ui.P6OView.setImage(np.transpose(img, (1, 0)))

    def proj6_reconst(self):
        method = self.ui.P6RL.currentText()
        if method == "Conditional Dilation":
            img = conditional_dilation(self.binary_img_6)
        elif method == "OBR":
            img = OBR_func(self.grey_img)
        elif method == "CBR":
            img = CBR_func(self.grey_img)
        self.ui.P6OView.clear()
        self.ui.P6OView.setImage(np.transpose(img, (1, 0)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
