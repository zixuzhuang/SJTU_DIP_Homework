import numpy as np
from scipy.stats import entropy
from skimage import img_as_ubyte, morphology


def thresholdOTSU(image):

    hist, bin_edges = np.histogram(np.ravel(image), 256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = hist.astype(np.float32)

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]

    return int(threshold)


def thresholdMaxEntropy(image, nbins=256):
    assert image.ndim == 2
    hist, bin_edges = np.histogram(image.ravel(), nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    H = []
    for i in range(nbins - 1):
        e = entropy(hist[0 : i + 1]) + entropy(hist[i + 1 :])
        H.append(e)
    idx = np.argmax(H)
    threshold = bin_centers[:-1][idx]

    return threshold
