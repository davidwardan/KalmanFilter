# import libraries
import numpy as np


# define gaussian mixture function
def gaussian_mixture(mu1, mu2, cov1, cov2, p1, p2):
    # get the mean
    mean = p1 * mu1 + p2 * mu2
    # get the covariance
    m1 = np.squeeze(mu1 - mean)
    m2 = np.squeeze(mu2 - mean)
    cov = p1 * (cov1 + m1 @ m1.T) + p2 * (cov2 + m2 @ m2.T)

    return mean, cov


# define a function to generate data
def gen_noisy_data(n, R):
    np.random.seed(0)
    t = np.linspace(0, n, n)
    z = t + np.random.normal(0, R[0], n)

    return z, t


# define a function to generate data using ssm
def gen_ssm_data(A, Q, C, R, x, n):
    np.random.seed(0)
    t = range(n)

    z = np.zeros(n)
    # x = np.random.multivariate_normal(x, P)
    for i in range(n):
        x = A @ x + np.random.multivariate_normal([0, 0], Q)
        z[i] = (C @ x + np.random.normal(0, np.sqrt(R))).flatten()[0]

    return z, t
