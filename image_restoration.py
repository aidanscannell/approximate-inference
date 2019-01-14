import numpy as np
from imageio import imread
from algorithms.icm import icm
from algorithms.gibbs import gibbs
from algorithms.var_bayes import var_bayes
from helpers import plot_comparison


def setup_image(img, noise=None):
    # read image and convert to greyscale
    orgImg = np.asarray(img) / 255

    # if image is already noisy don't add more noise!
    if noise is None:
        yImg = orgImg
    else:
        yImg = add_noise(orgImg, *noise)

    return orgImg, yImg

def add_gaussian_noise(im, prop, varSigma):
    """
     Adds Gaussian noise to an image
    :param im: input image
    :param prop: proportion of pixels to have noise added
    :param varSigma: variance of Gaussian noise
    :return: original image with added Gaussian noise
    """
    N = int(np.round(np.prod(im.shape) * prop))

    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N], im.shape)
    e = varSigma * np.random.randn(np.prod(im.shape)).reshape(im.shape)
    im2 = np.copy(im)
    im2 = np.float64(im2)
    im2[index] += e[index]

    return im2

def add_saltnpeppar_noise(im, prop):
    """
    Adds salt and pepper noise (flipped pixels) to an image
    :param im: input image
    :param prop: proportion of pixels to have noise added
    :return: im2: original image with added salt and pepper noise
    """
    N = int(np.round(np.prod(im.shape) * prop))

    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N], im.shape)
    im2 = np.copy(im)
    im2[index] = 1 - im2[index]

    return im2

def add_noise(orgImg, propSP, propG, varSigma, level):
    """
    Adds noise to self.orgImg and returns noisy image yImg
    :param prop: proportion of pixels to have noise added
    :param varSigma: variance of Gaussian noise
    :param level: 1 = Gaussian noise, 2 = Salt & Pepper noise, 3 = both
    """
    yImg = orgImg
    # add noise to image to create y (noisy image)
    if level == 1:  # gaussian noise only
        yImg = add_gaussian_noise(orgImg, propG, varSigma)  # noisy image
    elif level == 2:  # salt and pepper noise only
        yImg = add_saltnpeppar_noise(orgImg, propSP)  # noisy image
    elif level == 3:  # both gaussian + salt and pepper noise
        yImg = add_gaussian_noise(orgImg, propG, varSigma)
        yImg = add_saltnpeppar_noise(yImg, propSP)

    return yImg


algorithms = {"ICM": icm, "Gibbs": gibbs, "Variational Bayes": var_bayes}
# algorithm = algorithms["ICM"]

if __name__ == "__main__":
    # input image
    img_path = 'data/pug-glasses.jpg'
    img = imread(img_path)


    max_scans = 10
    consts = [1/4, 1]  # set weight parameters for prior and likelihood respectively [w, beta]
    noise = [[0.2, 0.2, 0.1, 3], [0.2, 0.2, 0.1, 3], [1.0, 1.0, 0.1, 1], [1.0, 1.0, 3.2, 1], [0.5, 0.5, 3.2, 1],
             [0.2, 0.2, 0, 2], [0.5, 0.5, 0, 2]]

    noiseLevel = {1: "Gaussian", 2: "Salt & Pepper", 3: "Gaussian + Salt & Pepper"}

    # run de-noising for different noise levels
    for n in noise:
        print("Noise Level:\n\t"
              "Gaussian Noise ~ N(0, %.2f), proportion = %.2f \n\t"
              "Salt & Pepper proportion = %.2f" % (n[2], n[1], n[0]))

        orgImg, yImg = setup_image(img, n)

        x = {}
        for name, alg in algorithms.items():
            print("\n-----Beginning %s De-Noising-----\n" % name)
            x[name] = alg(yImg, consts, max_scans)

        plot_comparison(orgImg, x, yImg)
