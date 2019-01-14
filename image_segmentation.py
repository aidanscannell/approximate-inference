import cv2
from sklearn.cluster import KMeans
import numpy as np
from ising_model import prior
from helpers import plot


def setup_image(img_path, num_clusters=3):
    img = cv2.imread(img_path)  # read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to rgb from bgr

    # Select ROI
    imgF = cv2.bitwise_or(img, img, mask=mask(img))
    imgB = cv2.bitwise_or(img, img, mask=mask(img))

    # cluster the pixel intensities
    clt = KMeans(n_clusters=num_clusters)
    clt.fit(img.reshape((-1, 3)))

    labelsF = clt.predict(imgF.reshape((imgF.shape[0] * imgF.shape[1], 3)))
    labelsB = clt.predict(imgB.reshape((imgB.shape[0] * imgB.shape[1], 3)))

    histF = centroid_histogram(labelsF, clt.labels_)
    histB = centroid_histogram(labelsB, clt.labels_)

    return img, clt, histF, histB


def mask(img):
    r = cv2.selectROI(img)
    mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)
    cv2.rectangle(mask, (int(r[0]), int(r[1])), (int(r[0] + r[2]), int(r[1] + r[3])), (255, 255, 255), -1)
    return mask


def centroid_histogram(labels, labelsAll):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(labelsAll)) + 1)
    (hist, _) = np.histogram(labels, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def gibbs(yImg, consts, max_scans, clt, histF, histB):
    """
    Re-implementation of Gibbs sampling for Ising Model (image de-noising) with
    likelihoods taking the form of histograms
    :param yImg: noisy image (y)
    :param consts: weight parameters for prior and likelihood respectively [w, beta]
    :param max_scans: number of full image passes
    :param li1: likelihood for x_ij = 1
    :param li1: likelihood for x_ij = -1
    :return: returns latent image x
    """
    xImg = np.zeros([yImg.shape[0], yImg.shape[1]])
    # xImg[yImg < np.mean(yImg)] = -1
    bins = clt.labels_.reshape([yImg.shape[0], yImg.shape[1]])
    for scan in range(max_scans):
        for i in range(yImg.shape[0]):
            for j in range(yImg.shape[1]):
                # TODO: add i and j to params
                px1 = histF[bins[i, j]] * prior(1, xImg, i, j, consts[0])
                px2 = histB[bins[i, j]] * prior(-1, xImg, i, j, consts[0])
                cond_dist = px1 / (px1 + px2)

                t = np.random.uniform(0, 1)
                #     t = np.random.uniform(0.4, 0.6)

                if cond_dist > t:
                    xImg[i, j] = 1
                else:
                    xImg[i, j] = -1

    return xImg


# img = 'data/pugRGBsmall.jpg'
img_path = 'data/pugRGBsmall.jpg'
num_clusters = 50
max_scans = 5
consts = [1/4, 1]  # set weight parameters for prior and likelihood respectively [w, beta]

yImg, clt, histF, histB = setup_image(img_path, num_clusters)

xImg = gibbs(yImg, consts, max_scans, clt, histF, histB)

plot(yImg, xImg)
