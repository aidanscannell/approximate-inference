import numpy as np
from ising_model import prior, likelihood


def gibbs(yImg, consts, max_scans):
    """
    Implements Gibbs sampling for Ising Model (image de-noising)
    :param yImg: noisy image (y)
    :param consts: weight parameters for prior and likelihood respectively [w, beta]
    :param max_scans: number of full image passes
    :return: returns latent image x
    """
    xImg = np.ones(yImg.shape)
    xImg[yImg < np.mean(yImg)] = -1
    for scan in range(max_scans):
        for i in range(yImg.shape[0]):
            for j in range(yImg.shape[1]):
                # TODO: add i and j to params
                px1 = likelihood(yImg[i, j], 1, consts[1]) * prior(1, xImg, i, j, consts[0])
                px2 = likelihood(yImg[i, j], -1, consts[1]) * prior(-1, xImg, i, j, consts[0])
                cond_dist = px1 / (px1 + px2)

                t = np.random.uniform(0, 1)
                #     t = np.random.uniform(0.4, 0.6)

                if cond_dist > t:
                    xImg[i, j] = 1
                else:
                    xImg[i, j] = -1

    return xImg
