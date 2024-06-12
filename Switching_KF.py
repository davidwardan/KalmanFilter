# TODO: Update code
'''
Implementation not ready yet for the switching kalman filter. The code is not working as expected.
use SKF.py for the implementation of the switching kalman filter.
'''

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

# import needed functions
from KalmanFilters import KalmanFilter
from helper_functions import gaussian_mixture

# np.random.seed(235)

# create dataset from given
x = np.array([[10], [0]])
P = np.array([[4**2, 0],
                          [0, 0.001**2]])
pi = np.array([[0.95], [0.05]])
probability = [pi]

A_1 = np.array([[1, 0],
                                [0, 0]])
A_2 = np.array([[1, 1],
                                [0, 1]])

Q_1_1, Q_2_2, Q_2_1 = 0, 0, 0

Q_1_2 = np.array([[0, 0],
                                [0, (2.3*10**-4)**2]])

C = np.array([[1, 0]])
R = np.array([1**2])

# create time steps with 5 seconds interval
t = np.arange(0, 200)
t_switch = int(len(t)/2)

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
    posterior_mean_1, posterior_cov_1 = gaussian_mixture(xfilt_1_1, xfilt_2_1, Pfilt_1_1,  Pfilt_2_1, w_1_1, w_2_1)
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
    knowledge_mean, knowledge_cov = gaussian_mixture(posterior_mean_1, posterior_mean_2, posterior_cov_1, posterior_cov_2, pi[0], pi[1])
    pred_mean.append(knowledge_mean)
    pred_cov.append(knowledge_cov)

# plot the porbabilities
figure1, axis = plt.subplots()
axis.plot(np.arange(1, len(t)), M1, label='Regime 1', color='red', marker='o', markersize=2)
axis.plot(np.arange(1, len(t)), M2, label='Regime 2', color='blue', marker='o', markersize=2)
axis.legend()

# plot first hidden state and observations
figure2, axis = plt.subplots()
axis.plot(t, [i[0][0] for i in pred_mean], label='Predicted Mean', color='steelblue', marker='o', markersize=1)
axis.scatter(t[1:], observations, label='Observations', color='darkgreen', marker='s', s=1)
axis.fill_between(t, [i[0][0] - math.sqrt(j[0][0]) for i, j in zip(pred_mean, pred_cov)], [i[0][0] + math.sqrt(j[0][0]) for i, j in zip(pred_mean, pred_cov)], color='darkorange', alpha=0.5)
axis.set_ylim(-20, 60)
axis.plot(t[1:], obs_crt, color='red', markersize=2, linestyle='--')
axis.legend()

# plot second hidden state
figure3, axis = plt.subplots()
axis.plot(t, [i[1][0] for i in pred_mean], label='Predicted Mean', color='steelblue', marker='o', markersize=2)
axis.fill_between(t, [i[1][0] - math.sqrt(j[1][1]) for i, j in zip(pred_mean, pred_cov)], [i[1][0] + math.sqrt(j[1][1]) for i, j in zip(pred_mean, pred_cov)], color='darkorange', alpha=0.5)
axis.legend()
plt.show()