# import libraries
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

# import needed functions
from filter import kalman_filter

np.random.seed(235)

# create dataset from given
knowledge_mean = np.array([[10], [0]])
knowledge_cov = np.array([[0.01**2, 0], [0, 0.001**2]])
knowledge_probability = np.array([[0.95], [0.05]])

transition_matrix_1 = np.array([[1, 0],
                                [0, 0]])
transition_matrix_2 = np.array([[1, 1],
                                [0, 1]])

model_error_cov_1_1, model_error_cov_2_2, model_error_cov_2_1 = 0, 0, 0

model_error_cov_1_2 = np.array([[0, 0],
                                [0, (2.3*10**-4)**2]])

observation_matrix = np.array([[1, 0]])
observation_error_cov = 0.5**2

# create time steps with 5 seconds interval
t = np.arange(0, 50)

# Create probability transition matrix
Z = np.array([[0.997, 0.003],
              [0.003, 0.997]])

# Calculate hidden states using the transition matrix
hidden_states = [knowledge_mean]
for i in range(1, len(t)):
    if i < 25:
        hidden_states.append(transition_matrix_1 @ hidden_states[i-1] + np.random.normal(0, 0.5**2))
    else:
        hidden_states.append(transition_matrix_2 @ hidden_states[i-1] + np.random.normal(0, 0.5**2))

# Create observations from hidden states
observations = []
for i in range(1,len(hidden_states)):
    y = observation_matrix @ hidden_states[i] + np.random.normal(0, observation_error_cov)
    observations.append(y[0][0])

# Apply switching kalman filter
M1 = []
M2 = []
probability = [knowledge_probability]
pred_mean = [knowledge_mean]
pred_cov = [knowledge_cov]
# Loop through the observations
for i in range(len(observations)):
    # apply kalman filter for regime 1 to regime 1
    states_1_1 = kalman_filter(i, knowledge_mean, knowledge_cov, transition_matrix_1, observation_matrix,
                                        observation_error_cov, model_error_cov_1_1, observations)
    likelihood_1_1 = norm.pdf(observations[i], (observation_matrix @ states_1_1[0]),
                                np.sqrt((observation_matrix @ states_1_1[1] @ np.transpose(observation_matrix) + observation_error_cov)))

    # apply kalman filter for regime 1 to regime 2
    states_1_2 = kalman_filter(i, knowledge_mean, knowledge_cov, transition_matrix_2, observation_matrix,
                                        observation_error_cov, model_error_cov_1_2, observations)
    likelihood_1_2 = norm.pdf(observations[i], (observation_matrix @ states_1_2[0]),
                                np.sqrt((observation_matrix @ states_1_2[1] @ np.transpose(observation_matrix) + observation_error_cov)))

    # apply kalman filter for regime 2 to regime 2
    states_2_2 = kalman_filter(i, knowledge_mean, knowledge_cov, transition_matrix_2, observation_matrix,
                                        observation_error_cov, model_error_cov_2_2, observations)
    likelihood_2_2 = norm.pdf(observations[i], (observation_matrix @ states_2_2[0]),
                                np.sqrt((observation_matrix @ states_2_2[1] @ np.transpose(observation_matrix) + observation_error_cov)))

    # apply kalman filter for regime 2 to regime 1
    states_2_1 = kalman_filter(i, knowledge_mean, knowledge_cov, transition_matrix_1, observation_matrix,
                                        observation_error_cov, model_error_cov_2_1, observations)
    likelihood_2_1 = norm.pdf(observations[i], (observation_matrix @ states_2_1[0]),
                                np.sqrt((observation_matrix @ states_2_1[1] @ np.transpose(observation_matrix) + observation_error_cov)))

    # calculate the joint posterior probability for regime 1 to regime 1
    M_sum = (likelihood_1_1 * Z[0][0] * knowledge_probability[0] + likelihood_1_2 * Z[0][1] * knowledge_probability[1]
             + likelihood_2_1 * Z[1][0] * knowledge_probability[0] + likelihood_2_2 * Z[1][1] * knowledge_probability[1])

    M_1_1 = likelihood_1_1 * Z[0][0] * knowledge_probability[0] / M_sum

    # calculate the joint posterior probability for regime 1 to regime 2
    M_1_2 = likelihood_1_2 * Z[0][1] * knowledge_probability[1] / M_sum

    # calculate the joint posterior probability for regime 2 to regime 2
    M_2_2 = likelihood_2_2 * Z[1][1] * knowledge_probability[1] / M_sum

    # calculate the joint posterior probability for regime 2 to regime 1
    M_2_1 = likelihood_2_1 * Z[1][0] * knowledge_probability[0] / M_sum

    # get the new knowledge probability vector
    M1.append(np.squeeze(M_1_1 + M_2_1))
    M2.append(np.squeeze(M_1_2 + M_2_2))
    knowledge_probability = np.array([[M1[-1]], [M2[-1]]])

    knowledge_probability = knowledge_probability.reshape(2, 1)
    probability.append(knowledge_probability)

    # get the posterior mean and covariance for regime 1
    w_1_1 = M_1_1 / (M_1_1 + M_2_1)
    w_2_1 = M_2_1 / (M_1_1 + M_2_1)
    posterior_mean_1 = w_1_1 * states_1_1[2] + w_2_1 * states_2_1[2]
    mu = np.squeeze(states_1_1[2]-posterior_mean_1)
    posterior_cov_1 = (w_1_1 * (states_1_1[3] + mu@mu.T)
                       + w_2_1 * (states_2_1[3] + mu@mu.T))

    # get the posterior mean and covariance for regime 2
    w_1_2 = M_1_2 / (M_1_2 + M_2_2)
    w_2_2 = M_2_2 / (M_1_2 + M_2_2)
    posterior_mean_2 = w_1_2 * states_1_2[2] + w_2_2 * states_2_2[2]
    mu = np.squeeze(states_1_2[2]-posterior_mean_2)
    posterior_cov_2 = (w_1_2 * (states_1_2[3] + mu@mu.T)
                       + w_2_2 * (states_2_2[3] + mu@mu.T))

    # get the new knowledge mean and covariance
    knowledge_mean = knowledge_probability[0] * posterior_mean_1 + knowledge_probability[1] * posterior_mean_2
    mean_diff = np.squeeze(posterior_mean_1 - posterior_mean_2)
    knowledge_cov = knowledge_probability[0] * posterior_cov_1 + knowledge_probability[1] * posterior_cov_2 + knowledge_probability[0]*knowledge_probability[1]*(mean_diff)@(mean_diff).T
    pred_mean.append(knowledge_mean)
    pred_cov.append(knowledge_cov)

# plot the porbabilities
figure3, axis = plt.subplots()
axis.plot(np.arange(1, len(t)), M1, label='Regime 1', color='red', marker='o', markersize=2)
axis.plot(np.arange(1, len(t)), M2, label='Regime 2', color='blue', marker='o', markersize=2)
axis.legend()

# plot first hidden state and observations
figure1, axis = plt.subplots()
axis.plot(t, [i[0][0] for i in pred_mean], label='Predicted Mean', color='red', marker='o', markersize=2)
axis.scatter(t[1:], observations, label='Observations', color='green', marker='s', s=10)
axis.legend()
plt.show()

# plot second hidden state
# figure2, axis = plt.subplots()
# axis.plot(np.arange(0, 501), [i[1][0] for i in pred_mean], label='Predicted Mean', color='red', marker='o', markersize=2)
# axis.legend()
# plt.show()