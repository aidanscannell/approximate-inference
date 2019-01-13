from ising_model import prior, likelihood
import numpy as np

def icm(yImg, consts, max_scans):
    """
    Implements Iterative Conditional Modes for Ising Model (image de-noising) (a coordinate-wise gradient approach)
    :param yImg: noisy image (y)
    :param consts: weight parameters for prior and likelihood respectively [w, beta]
    :param max_scans: number of full image passes
    :return: returns latent value of pixel (i, j) x_ij
    """
    xImg = np.ones(yImg.shape)
    xImg[yImg < np.mean(yImg)] = -1
    for scan in range(max_scans):
        for i in range(yImg.shape[0]):
            for j in range(yImg.shape[1]):
                joint_dist_pos = likelihood(yImg[i, j], 1, consts[1]) * prior(1, xImg, i, j, consts[0])
                joint_dist_neg = likelihood(yImg[i, j], -1, consts[1]) * prior(-1, xImg, i, j, consts[0])

                if joint_dist_pos > joint_dist_neg:
                    xImg[i, j] = 1
                else:
                    xImg[i, j] = -1
    return xImg
