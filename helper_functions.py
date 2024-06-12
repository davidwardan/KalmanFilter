# improt libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
# define gaussian mixture function
def gaussian_mixture(mu1, mu2, cov1, cov2, p1, p2):
    # get the mean
    mean = p1 * mu1 + p2 * mu2
    # get the covariance
    m1 = np.squeeze(mu1-mean)
    m2 = np.squeeze(mu2-mean)
    cov = p1 * (cov1 + m1 @ m1.T) + p2 * (cov2 + m2 @ m2.T)

    return mean, cov

# define a function to generate data
def gen_data(n,R):
    np.random.seed(0)
    t = np.linspace(0, n, n)
    z = t + np.random.normal(0, R[0], n)

    return z, t









