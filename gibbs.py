import numpy as np
from ising_model import prior, likelihood


def gibbs(yImg, consts, max_scans, li1=None, li2=None):
    """
    Implements Gibbs sampling for Ising Model (image de-noising)
    :param yImg: noisy image (y)
    :param consts: weight parameters for prior and likelihood respectively [w, beta]
    :param max_scans: number of full image passes
    :param li1: likelihood for x_ij = 1
    :param li1: likelihood for x_ij = -1
    :return: returns latent image x
    """
    xImg = np.ones(yImg.shape)
    xImg[yImg < np.mean(yImg)] = -1
    for scan in range(max_scans):
        for i in range(yImg.shape[0]):
            for j in range(yImg.shape[1]):
                # TODO: add i and j to params
                if li1 is None:
                    li1 = likelihood(yImg[i, j], 1, consts[1])
                if li2 is None:
                    li2 = likelihood(yImg[i, j], -1, consts[1])
                px1 = li1 * prior(1, xImg, i, j, consts[0])
                px2 = li2 * prior(-1, xImg, i, j, consts[0])
                cond_dist = px1 / (px1 + px2)

                t = np.random.uniform(0, 1)
                #     t = np.random.uniform(0.4, 0.6)

                if cond_dist > t:
                    xImg[i, j] = 1
                else:
                    xImg[i, j] = -1

    return xImg
# def gibbs(yImg, xImg, i, j, consts, li1, li2):
#     """
#     Implements Gibbs sampling for Ising Model (image de-noising)
#     :param yImg: noisy image (y)
#     :param xImg: latent image (x)
#     :param i: row index
#     :param j: col index
#     :param consts: weight parameters for prior and likelihood respectively [w, beta]
#     :return: returns latent value of pixel (i, j) x_ij
#     """
#     # TODO: add i and j to params
#     # px1 = likelihood(yImg[i, j], 1, consts[1]) * prior(1, xImg, i, j, consts[0])
#     # px2 = likelihood(yImg[i, j], -1, consts[1]) * prior(-1, xImg, i, j, consts[0])
#     px1 = li1 * prior(1, xImg, i, j, consts[0])
#     px2 = li2 * prior(-1, xImg, i, j, consts[0])
#     cond_dist = px1 / (px1 + px2)
#
#     t = np.random.uniform(0, 1)
#     #     t = np.random.uniform(0.4, 0.6)
#
#     if cond_dist > t:
#         return 1
#     else:
#         return -1
