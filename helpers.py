import numpy as np
import matplotlib.pyplot as plt
import cv2

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

def plot(yImg, xImg):
    maskF = np.full((xImg.shape[0], xImg.shape[1]), 0, dtype=np.uint8)  # mask is only
    maskB = np.full((xImg.shape[0], xImg.shape[1]), 0, dtype=np.uint8)  # mask is only

    maskF[xImg == 1] = 255
    maskB[xImg == -1] = 255

    maskF = cv2.bitwise_or(yImg, yImg, mask=maskF)
    maskB = cv2.bitwise_or(yImg, yImg, mask=maskB)

    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(yImg, cmap='gray')
    axarr[0, 0].set_title('Original Image')
    axarr[1, 0].imshow(xImg, cmap='gray')
    axarr[1, 0].set_title('Latent Image')
    axarr[0, 1].imshow(maskB, cmap='gray')
    axarr[0, 1].set_title('Background Image')
    axarr[1, 1].imshow(maskF, cmap='gray')
    axarr[1, 1].set_title('Foreground Image')
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False) # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    plt.show()
