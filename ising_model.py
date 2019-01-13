import numpy as np


def neighbours(i, j, M, N, size=4):
    """
    Finds the neighbours of a node (i, j)
    :param i: y position
    :param j: x position
    :param M: height of image
    :param N: width of image
    :param size: 4 doesn't include diagonals, 8 does include diagonals
    :return: array containing coordinates of neighbours
    """
    if size == 4:
        if i == 0 and j == 0:
             n = [(0, 1), (1, 0)]
        elif i == 0 and j == N-1:
             n = [(0, N-2), (1, N-1)]
        elif i == M-1 and j == 0:
             n = [(M-1, 1), (M-2, 0)]
        elif i == M-1 and j == N-1:
             n = [(M-1, N-2), (M-2, N-1)]
        elif i == 0:
             n = [(0, j-1), (0, j+1), (1, j)]
        elif i == M-1:
             n = [(M-1, j-1), (M-1, j+1), (M-2, j)]
        elif j == 0:
             n = [(i-1, 0), (i+1, 0), (i, 1)]
        elif j == N-1:
             n = [(i-1, N-1), (i+1, N-1), (i, N-2)]
        else:
             n = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
        return n

    if size == 8:
        if i == 0 and j == 0:
             n = [(0, 1), (1, 0), (1, 1)]
        elif i == 0 and j == N-1:
             n = [(0, N-2), (1, N-1), (1, N-2)]
        elif i == M-1 and j == 0:
             n = [(M-1, 1), (M-2, 0), (M-2, 1)]
        elif i == M-1 and j == N-1:
             n = [(M-1, N-2), (M-2, N-1), (M-2, N-2)]
        elif i == 0:
             n = [(0, j-1), (0, j+1), (1, j), (1, j+1), (1, j-1)]
        elif i == M-1:
             n = [(M-1, j-1), (M-1, j+1), (M-2, j), (M-2, j-1), (M-2, j+1)]
        elif j == 0:
             n = [(i-1, 0), (i+1, 0), (i, 1), (i-1, 1), (i+1, 1)]
        elif j == N-1:
             n = [(i-1, N-1), (i+1, N-1), (i, N-2), (i-1, N-2), (i+1, N-2)]
        else:
             n = [(i-1, j), (i+1, j), (i, j-1), (i, j+1), (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]

        return np.array(n)


def prior(x_i, x, i, j, w):
    """
    Calcualtes Ising prior p(x) (assuming pixel depends on its neighbouring pixels)
    :param x_i: value of x at index i (x[i,j])
    :param x: latent variable x
    :param i: y position
    :param j: x position
    :param w: prior weight parameter
    :retur: prior p(x)
    """
    n = neighbours(i, j, x.shape[0], x.shape[1], size=8)
    # TODO: Add 1/Z
    return np.exp(np.sum(w * x_i * x[n[:, 0], n[:, 1]]))


def li(y, x, beta):
    """
    Calculates the likelihhod p(y|x)
    :param y: noisy image value (y_i)
    :param x: latent image value (x_i)
    :param beta: likelihood weight parameter
    :return: likelihhod p(y|x)
    """
    return beta * ((2 * y - 1) * x)


def likelihood(y, x, beta):
    """
    Calculates the likelihhod p(y|x) = 1/Z prod_over_i exp(Li(xi))
    :param y: noisy image value (y_i)
    :param x: latent image value (x_i)
    :param beta: likelihood weight parameter
    :return: likelihhod p(y|x)
    """
    # TODO: Add 1/Z
    return np.exp(li(y, x, beta))
