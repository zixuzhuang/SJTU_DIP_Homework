import cv2
import numpy as np
from scipy.stats import entropy
from skimage import img_as_ubyte, morphology


def conditional_dilation(binary_image):
    target = binary_image.copy()
    marker = binary_image.copy()
    for i in range(3):
        marker = morphology.erosion(marker)
    while np.sum(marker - morphology.dilation(marker)) > 0:
        marker = morphology.dilation(marker)
        marker = marker * binary_image
    return marker


def grayscale_reconstruction(grey_image):
    count = 0
    mask = grey_image.copy()
    marker = grey_image.copy()
    for i in range(3):
        marker = morphology.erosion(marker)
    while True:
        marker = morphology.dilation(marker)
        marker = np.minimum(marker, mask)
        if count != np.sum(marker):
            count = np.sum(marker)
        else:
            return marker


def OBR_func(grey_image):
    open_image = morphology.opening(grey_image)
    return grayscale_reconstruction(open_image)


def CBR_func(grey_image):
    close_image = morphology.closing(grey_image)
    return grayscale_reconstruction(close_image)
