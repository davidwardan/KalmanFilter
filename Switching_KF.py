# TODO: Update code
"""
Implementation is not ready yet for the switching kalman filter.
The code is not working as expected.
Use SKF.py for the implementation of the switching kalman filter.
"""

import math
import os

import matplotlib.pyplot as plt
# import libraries
import numpy as np
from scipy.stats import norm

# import needed functions
from filters import KalmanFilter
from helper_functions import gaussian_mixture

# set plot parameters
plt.rcParams.update({'font.size': 16})
plt.rcParams['text.usetex'] = True

# create out directory if it does not exist
if not os.path.exists('out'):
    os.makedirs('out')

np.random.seed(235)

# create dataset from given
x = np.array([[10], [0]])
P = np.array([[4 ** 2, 0],
              [0, 0.001 ** 2]])
pi = np.array([[0.95], [0.05]])
probability = [pi]

A_1 = np.array([[1, 0],
                [0, 0]])
A_2 = np.array([[1, 1],
                [0, 1]])

Q_1_1, Q_2_2, Q_2_1 = 0, 0, 0

Q_1_2 = np.array([[0, 0],
                  [0, (2.3 * 10 ** -4) ** 2]])

C = np.array([[1, 0]])
R = np.array([1 ** 2])

pred_mean_s1 = [x]
pred_cov_s1 = [P]
pred_mean_s2 = [x]
pred_cov_s2 = [P]
pred_mean = [x]
pred_cov = [P]
# create time steps with 5-second interval
t = np.arange(0, 200)
t_switch = int(len(t) / 2)

# Create probability transition matrix
Z = np.array([[0.997, 0.003],
              [0.003, 0.997]])

observations = []
obs_crt = []
s = 0.25
for i in range(1, len(t)):
    if i < t_switch:
        obs_crt.append(10)
        observations.append(obs_crt[-1] + np.random.normal(0, R))
    else:
        obs_crt.append(10 + s)
        observations.append(obs_crt[-1] + np.random.normal(0, R))
        s += 0.25

# Apply switching kalman filter
M1 = []
M2 = []

kf_1_1 = KalmanFilter(A_1, Q_1_1, C, R, x, P)
kf_1_2 = KalmanFilter(A_1, Q_1_2, C, R, x, P)
kf_2_1 = KalmanFilter(A_2, Q_2_1, C, R, x, P)
kf_2_2 = KalmanFilter(A_2, Q_2_2, C, R, x, P)

pred_mean_s1 = [x]
pred_cov_s1 = [P]
pred_mean_s2 = [x]
pred_cov_s2 = [P]
pred_mean = [x]
pred_cov = [P]

# Loop through the observations
for z in observations:
    # apply kalman filter for regime 1 to regime 1
    kf_1_1.filter(z)
    xpred_1_1 = kf_1_1.x_pred[-1]
    Ppred_1_1 = kf_1_1.P_pred[-1]
    likelihood_1_1 = norm.pdf(z, (C @ xpred_1_1),
                              np.sqrt((C @ Ppred_1_1 @ C.T + R)))

    # apply kalman filter for regime 1 to regime 2
    kf_1_2.filter(z)
    xpred_1_2 = kf_1_2.x_pred[-1]
    Ppred_1_2 = kf_1_2.P_pred[-1]
    likelihood_1_2 = norm.pdf(z, (C @ xpred_1_2),
                              np.sqrt((C @ Ppred_1_2 @ C.T + R)))

    # apply kalman filter for regime 2 to regime 2
    kf_2_2.filter(z)
    xpred_2_2 = kf_2_2.x_pred[-1]
    Ppred_2_2 = kf_2_2.P_pred[-1]
    likelihood_2_2 = norm.pdf(z, (C @ xpred_2_2),
                              np.sqrt((C @ Ppred_2_2 @ C.T + R)))

    # apply kalman filter for regime 2 to regime 1
    kf_2_1.filter(z)
    xpred_2_1 = kf_2_1.x_pred[-1]
    Ppred_2_1 = kf_2_1.P_pred[-1]
    likelihood_2_1 = norm.pdf(z, (C @ xpred_2_1),
                              np.sqrt((C @ Ppred_2_1 @ C.T + R)))

    # calculate the joint posterior probability for regime 1 to regime 1
    M_sum = (likelihood_1_1 * Z[0][0] * pi[0] + likelihood_1_2 * Z[0][1] * pi[1]
             + likelihood_2_1 * Z[1][0] * pi[0] + likelihood_2_2 * Z[1][1] * pi[1])

    M_1_1 = likelihood_1_1 * Z[0][0] * pi[0] / M_sum

    # calculate the joint posterior probability for regime 1 to regime 2
    M_1_2 = likelihood_1_2 * Z[0][1] * pi[1] / M_sum

    # calculate the joint posterior probability for regime 2 to regime 2
    M_2_2 = likelihood_2_2 * Z[1][1] * pi[1] / M_sum

    # calculate the joint posterior probability for regime 2 to regime 1
    M_2_1 = likelihood_2_1 * Z[1][0] * pi[0] / M_sum

    # print(sum([M_1_1, M_1_2, M_2_1, M_2_2])) # check if the sum of the probabilities is 1

    # get the new knowledge probability vector
    M1.append(np.squeeze(M_1_1 + M_2_1))
    M2.append(np.squeeze(M_1_2 + M_2_2))
    pi = np.array([[M1[-1]], [M2[-1]]])

    # pi = pi.reshape(2, 1)
    probability.append(pi)

    # get the filtered states and covariances
    xfilt_1_1 = kf_1_1.x_filt[-1]
    Pfilt_1_1 = kf_1_1.P_filt[-1]
    xfilt_2_1 = kf_2_1.x_filt[-1]
    Pfilt_2_1 = kf_2_1.P_filt[-1]
    xfilt_1_2 = kf_1_2.x_filt[-1]
    Pfilt_1_2 = kf_1_2.P_filt[-1]
    xfilt_2_2 = kf_2_2.x_filt[-1]
    Pfilt_2_2 = kf_2_2.P_filt[-1]

    # get the posterior mean and covariance for regime 1
    w_1_1 = M_1_1 / (M_1_1 + M_2_1)
    w_2_1 = M_2_1 / (M_1_1 + M_2_1)
    posterior_mean_1, posterior_cov_1 = gaussian_mixture(xfilt_1_1, xfilt_2_1, Pfilt_1_1, Pfilt_2_1, w_1_1, w_2_1)
    pred_mean_s1.append(posterior_mean_1)
    pred_cov_s1.append(posterior_cov_1)

    # get the posterior mean and covariance for regime 2
    w_1_2 = M_1_2 / (M_1_2 + M_2_2)
    w_2_2 = M_2_2 / (M_1_2 + M_2_2)
    posterior_mean_2, posterior_cov_2 = gaussian_mixture(xfilt_1_2, xfilt_2_2, Pfilt_1_2, Pfilt_2_2, w_1_2, w_2_2)
    pred_mean_s2.append(posterior_mean_2)
    pred_cov_s2.append(posterior_cov_2)
    # print(sum([w_1_1, w_2_2, w_1_2, w_2_1])) # check if the probabilities are correct sum to 2

    # get the new knowledge mean and covariance
    # print(pi[0]+pi[1]) # check if the sum of the probabilities is 1
    knowledge_mean, knowledge_cov = gaussian_mixture(posterior_mean_1, posterior_mean_2, posterior_cov_1,
                                                     posterior_cov_2, pi[0], pi[1])
    pred_mean.append(knowledge_mean)
    pred_cov.append(knowledge_cov)

# plot the probabilities
figure, axis = plt.subplots(3, figsize=(12, 10))
axis[0].plot(np.arange(1, len(t)), M1, label='Regime 1', color='red')
axis[0].plot(np.arange(1, len(t)), M2, label='Regime 2', color='steelblue')
axis[0].legend(loc='upper left')
axis[0].set_xlabel('Time')
axis[0].set_ylabel('Probability')
axis[0].set_xlim(-1, len(t) + 1)
axis[0].set_ylim(-0.1, 1.1)
axis[0].set_title('Regime Probabilities')

# plot first hidden state and observations
axis[1].plot(t, [i[0][0] for i in pred_mean], label='State estimate', linestyle='--', color='red')
axis[1].scatter(t[1:], observations, label='Observations', color='steelblue', s=5, marker='o')
axis[1].fill_between(t, [i[0][0] - math.sqrt(j[0][0]) for i, j in zip(pred_mean, pred_cov)],
                     [i[0][0] + math.sqrt(j[0][0]) for i, j in zip(pred_mean, pred_cov)], color='red', alpha=0.2)
axis[1].set_ylim(-20, 60)
axis[1].plot(t[1:], obs_crt, color='black', linestyle='--')
axis[1].legend(loc='upper left')
axis[1].set_xlabel('Time')
axis[1].set_ylabel('Value')
axis[1].set_xlim(-1, len(t) + 1)
axis[1].set_title('First Hidden State')

# plot second hidden state
axis[2].plot(t, [i[1][0] for i in pred_mean], label='State estimate', linestyle='--', color='red')
axis[2].fill_between(t, [i[1][0] - math.sqrt(j[1][1]) for i, j in zip(pred_mean, pred_cov)],
                     [i[1][0] + math.sqrt(j[1][1]) for i, j in zip(pred_mean, pred_cov)], color='red', alpha=0.2)
axis[2].legend(loc='upper left')
axis[2].set_xlabel('Time')
axis[2].set_ylabel('Value')
axis[2].set_xlim(-1, len(t) + 1)
axis[2].set_title('Second Hidden State')

plt.tight_layout()

# save the plot
plt.savefig('out/' + 'SKF.png', dpi=300, bbox_inches='tight')
