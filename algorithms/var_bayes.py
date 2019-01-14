import numpy as np
from ising_model import neighbours, li


def var_bayes(yImg, consts, max_scans):
    """
    Implementation of variational Bayes for Ising Model
    :param yImg: noisy image (y)
    :param consts: weight parameters for prior and likelihood respectively [w, beta]
    :param max_scans: number of full image passes
    :return: returns variational parameter mu
    """
    mu = np.zeros(yImg.shape)
    for scan in range(max_scans):
        for i in range(yImg.shape[0]):
            for j in range(yImg.shape[1]):
                neighbour = neighbours(i, j, yImg.shape[0], yImg.shape[1], size=8)  # get neighbours of current pixel
                m_i = consts[0] * np.sum(mu[neighbour[:, 0], neighbour[:, 1]])  # m_i = sum (w_ij * mu_ij)
                a_i = m_i + 0.5 * (li(yImg[i, j], 1, consts[1]) - li(yImg[i, j], -1, consts[1]))
                mu[i, j] = np.tanh(a_i)
    mu[mu > 0] = 1
    mu[mu <= 0] = -1
    return mu

