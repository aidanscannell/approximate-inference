# Approximate Inference

In machine learning inference is the task of combinging our assumptions with the observed data. More specifically, if we have a set of observed data $\mathbf{Y}$, parameterized by a variable $\theta$, then we wish to obtain the posterior,

<a href="https://www.codecogs.com/eqnedit.php?latex=p(\mathbf{\theta}&space;\mid&space;\mathbf{Y})&space;=&space;\frac{p(\mathbf{Y}&space;\mid&space;\mathbf{\theta})&space;p(\mathbf{\theta})}{p(\mathbf{Y})}." target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(\mathbf{\theta}&space;\mid&space;\mathbf{Y})&space;=&space;\frac{p(\mathbf{Y}&space;\mid&space;\mathbf{\theta})&space;p(\mathbf{\theta})}{p(\mathbf{Y})}." title="p(\mathbf{\theta} \mid \mathbf{Y}) = \frac{p(\mathbf{Y} \mid \mathbf{\theta}) p(\mathbf{\theta})}{p(\mathbf{Y})}." /></a>

The denominator is known as the marginal likelihood (or evidence) and represents the probability of the observed data when all of the assumptions have been propogated through and integrated out,

<a href="https://www.codecogs.com/eqnedit.php?latex=p(\mathbf{Y})&space;=&space;\int&space;p(\mathbf{Y},&space;\mathbf{\theta})&space;d\theta." target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(\mathbf{Y})&space;=&space;\int&space;p(\mathbf{Y},&space;\mathbf{\theta})&space;d\theta." title="p(\mathbf{Y}) = \int p(\mathbf{Y}, \mathbf{\theta}) d\theta." /></a>

Sometimes this integral is intractable (computationally or analytically) so we cannot exploit conjugacy to avoid it's calculation. For this reason we have to make approximations to this integral.

This can be demonstarted through image restoration, the process of cleaning up images that have been corrupted by noise. 


## Installation

- Install the requirements using```pip install -r requirements```.
    - Make sure you use Python 3.
    - You may want to use a virtual environment for this.

## Usage
This work implements and compares a variety of approximate inference techniques for the tasks of image de-noising (restoration) and image segmentation.
### Image Restoration (De-nosing)
- Run ```python image_restoration.py``` to perform image restoration on the [pug image](data/pug-glasses.jpg) using Iterative Conditional Modes, Gibbs Sampling and Variational Bayes.
    - Feel free to use you own image.
- Alternatively, see the [Image Restoration Notebook](notebooks/image_restoration.ipynb) for a walk through of the code and underlying mathematics. 

### Image Segmentation
- Run ```python image_segmentation.py``` to run image segmentation on the [pug image](data/pugRGBsmall.jpg).
    - Again, feel free to use you own image.
- This will prompt the user to select a ROI in the foreground and a ROI in the background.
    - It uses Gibbs sampling (a MCMC method) to infer the *latent* foreground/background segmentation. 