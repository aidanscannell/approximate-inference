from imageio import imread
from image_restoration import ImageRestoration
from icm import icm
from gibbs import gibbs
from var_bayes import var_bayes
from helpers import plot_comparison


# input image
img_path = 'data/pug-glasses.jpg'
img = imread(img_path)

algorithms = {"ICM": icm, "Gibbs": gibbs, "Variational Bayes": var_bayes}
consts = [1/4, 1] # set weight parameters for prior and likelihood respectively [w, beta]
noise = [[0.2, 0.2, 0.1, 3], [0.2, 0.2, 0.1, 3], [1.0, 1.0, 0.1, 1], [1.0, 1.0, 3.2, 1], [0.5, 0.5, 3.2, 1], [0.2, 0.2, 0, 2], [0.5, 0.5, 0, 2]]
noiseLevel = {1: "Gaussian", 2: "Salt & Pepper", 3: "Gaussian + Salt & Pepper"}

# run de-noising for different noise levels
for n in noise:
    print("Noise Level:\n\t"
          "Gaussian Noise ~ N(0, %.2f), proportion = %.2f \n\t"
          "Salt & Pepper proportion = %.2f" % (n[2], n[1], n[0]))

    # initialise image restorer with image and noise
    restorer = ImageRestoration(img, n)

    x = {}
    for name, alg in algorithms.items():
        print("\n-----Beginning %s De-Noising-----\n" % name)
        x[name] = restorer.run(alg, consts)

    plot_comparison(restorer.orgImg, x, restorer.yImg)
