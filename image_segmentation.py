import cv2
from sklearn.cluster import KMeans
import numpy as np
from ising_model import neighbours
import matplotlib.pyplot as plt


class ImageSegmentation:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, img_path, num_clusters=3):
        self.num_clusters = num_clusters
        self.CLUSTERS = num_clusters
        self.img = cv2.imread(img_path)  # read image
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # convert to rgb from bgr

        self.yImg = np.asarray(self.img)

        self.width = self.img.shape[1]
        self.height = self.img.shape[0]

        # Select ROI
        self.maskF = self.mask()
        self.maskB = self.mask()

        imgF = cv2.bitwise_or(self.img, self.img, mask=self.maskF)
        imgB = cv2.bitwise_or(self.img, self.img, mask=self.maskB)

        plt.imshow(imgF)
        plt.show()

        # image = self.img.reshape((-1, 3))
        #
        # # cluster the pixel intensities
        # clt = KMeans(n_clusters=self.num_clusters)
        # clt.fit(image)

        clt = self.cluster(self.img)

        labelsF = clt.predict(imgF.reshape((imgF.shape[0] * imgF.shape[1], 3)))
        labelsB = clt.predict(imgB.reshape((imgB.shape[0] * imgB.shape[1], 3)))

        histF = self.centroid_histogram(labelsF)
        histB = self.centroid_histogram(labelsB)



    def likelihood(self, val, histF, histB):
        # set histogram values
        nB, binsB, patchesB = histB
        nF, binsF, patchesF = histF

        # find likelihood of foreground and background and normalise
        likelihoodF = nF[int(round(val))]
        likelihoodB = nB[int(round(val))]
        sumLikelihood = likelihoodB + likelihoodF
        likelihoodF = likelihoodF / sumLikelihood
        likelihoodB = likelihoodB / sumLikelihood


    def mask(self):
        r = cv2.selectROI(self.img)
        mask = np.full((self.img.shape[0], self.img.shape[1]), 0, dtype=np.uint8)  # mask is only
        cv2.rectangle(mask, (int(r[0]), int(r[1])), (int(r[0]+r[2]), int(r[1]+r[3])), (255, 255, 255), -1)
        return mask

    def cluster(self, img):
        # reshape the image to be a list of pixels
        # image = img.reshape((img.shape[0] * img.shape[1], 3))
        image = img.reshape((-1, 3))

        # cluster the pixel intensities
        clt = KMeans(n_clusters=self.num_clusters)
        clt.fit(image)
        return clt

    def centroid_histogram(self, labels):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(labels)) + 1)
        (hist, _) = np.histogram(labels, bins=numLabels)

        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()

        # return the histogram
        return hist




# img = 'data/pugRGBsmall.jpg'
img = 'data/pugRGBsmall.jpg'
num_clusters = 20
segmenter = ImageSegmentation(img, num_clusters)

# # set variational distribution
# mu = np.zeros([xImg.shape[0], xImg.shape[1]])


def hist(noisyImg,latentImg, x, y, consts, mu, histF, histB):
    # calculate average value for pixel i
    val = noisyImg[x, y]
    val = sum(val) / 3

    # set histogram values
    nB, binsB, patchesB = histB
    nF, binsF, patchesF = histF

    # find likelihood of foreground and background and normalise
    likelihoodF = nF[int(round(val))]
    likelihoodB = nB[int(round(val))]
    sumLikelihood = likelihoodB + likelihoodF
    likelihoodF = likelihoodF / sumLikelihood
    likelihoodB = likelihoodB / sumLikelihood

    # calculate variational parameter for current pixel
    m_i = 0
    neighbour = neighbours(x, y, latentImg.shape[0], latentImg.shape[1], size=8)
    for ii in range(len(neighbour)):
        neighbour1 = neighbour[ii]
        m_i += consts[1] * mu[neighbour1[0], neighbour1[1]]
    a_i = m_i + (likelihoodF - likelihoodB) # where Li(xi) = h*sumXi - n*sumXiYi
    mu_i = np.tanh(a_i)

    # update variational parameter matrix
    mu[x, y] = mu_i

    # calculate posterior for pixel i
    q_x = 1 / (1 + np.exp(-2 * a_i))

    # set pixel value in latent image (x) based on posterior
    if q_x < 0.5:
        latentImg[x, y] = 1
    else:
        latentImg[x, y] = -1

    return latentImg, mu
