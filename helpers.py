import numpy as np
import matplotlib.pyplot as plt

def percent_correct(xImg, orgImg):
    """
    Calculates the percentage of pixels that are correct in xImg relative to orgImg
    :param xImg: latent image (x)
    :param orgImg: original (noise free) image
    :return: percentage of pixels in xImg that are the same as in orgImg
    """
    return float(np.sum(xImg == orgImg)) / orgImg.size * 100.00

def plot_comparison(orgImg, xImg, yImg):
    """
    Plot original image, noisy image and de-noised image
    :param orgImg: original image
    :param xImg: latent image (x)
    :param yImg: noisy image (y)
    """
    if type(xImg) is dict:
        f, axarr = plt.subplots(2, len(xImg))
        axarr[0, 0].imshow(orgImg, cmap='gray')
        axarr[0, 0].set_title('Original Image')
        axarr[0, 1].imshow(yImg, cmap='gray')
        axarr[0, 1].set_title('Noisy Image')
        axarr[0, 2] = None
        for i, (name, x) in enumerate(xImg.items()):
            axarr[1, i].imshow(x, cmap='gray')
            axarr[1, i].set_title(name)

    else:
        f, axarr = plt.subplots(1,3)
        axarr[0].imshow(orgImg, cmap='gray')
        axarr[0].set_title('Original Image')
        axarr[1].imshow(yImg, cmap='gray')
        axarr[1].set_title('Noisy Image')
        axarr[2].imshow(xImg, cmap='gray')
        axarr[2].set_title('De-noised Image')

    plt.show()

def plot_iteration(xImg, i):
    """
    Plots de-noised image at iteration i
    :param xImg: latent image (x)
    :param i: number of iterations
    """
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.imshow(xImg, cmap='gray')
    ax.set_title('De-noised Image (%d iterations)' % (i+1), fontsize=16)
    plt.show()
