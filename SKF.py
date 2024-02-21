# import libraries
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm

# import needed functions
from filter import kalman_filter
from helper_functions import gaussian_mixture

# np.random.seed(235)

# create dataset from given
knowledge_mean = np.array([[10], [0]])
knowledge_cov = np.array([[4**2, 0],
                          [0, 0.001**2]])
knowledge_probability = np.array([[0.95], [0.05]])

transition_matrix_1 = np.array([[1, 0],
                                [0, 0]])
transition_matrix_2 = np.array([[1, 1],
                                [0, 1]])

model_error_cov_1_1, model_error_cov_2_2, model_error_cov_2_1 = 0, 0, 0

model_error_cov_1_2 = np.array([[0, 0],
                                [0, (2.3*10**-4)**2]])

observation_matrix = np.array([[1, 0]])
observation_error_cov = 1**2

# create time steps with 5 seconds interval
t = np.arange(0, 200)
t_switch = int(len(t)/2)

# Create probability transition matrix
Z = np.array([[0.997, 0.003],
              [0.003, 0.997]])

# Create observations from hidden states
# hidden_states = [knowledge_mean]
# for i in range(1, len(t)):
#     if i < t_switch:
#         hidden_states.append(transition_matrix_1 @ hidden_states[-1] + np.random.normal(0, 1))
#     else:
#         hidden_states.append(transition_matrix_2 @ hidden_states[i-1] + np.random.normal(0, 1))
# observations = []
# for i in range(1,len(hidden_states)):
#     y = observation_matrix @ hidden_states[i] + np.random.normal(0, observation_error_cov)
#     observations.append(y[0][0])

observations = []
obs_crt = []
s = 0.25
for i in range(1, len(t)):
    if i < t_switch:
        obs_crt.append(10)
        observations.append(obs_crt[-1] + np.random.normal(0, observation_error_cov))
    else:
        obs_crt.append(10 + s)
        observations.append(obs_crt[-1] + np.random.normal(0, observation_error_cov))
        s += 0.25
print(observations)

# Apply switching kalman filter
M1 = []
M2 = []
probability = [knowledge_probability]
pred_mean_s1 = [knowledge_mean]
pred_cov_s1 = [knowledge_cov]
pred_mean_s2 = [knowledge_mean]
pred_cov_s2 = [knowledge_cov]
pred_mean = [knowledge_mean]
pred_cov = [knowledge_cov]
# Loop through the observations
for i in range(len(observations)):
    # apply kalman filter for regime 1 to regime 1
    states_1_1 = kalman_filter(pred_mean_s1[-1], pred_cov_s1[-1], transition_matrix_1, observation_matrix,
                                        observation_error_cov, model_error_cov_1_1, observations[i])
    likelihood_1_1 = norm.pdf(observations[i], (observation_matrix @ states_1_1[0]),
                                np.sqrt((observation_matrix @ states_1_1[1] @ observation_matrix.T + observation_error_cov)))

    # apply kalman filter for regime 1 to regime 2
    states_1_2 = kalman_filter(pred_mean_s1[-1], pred_cov_s1[-1], transition_matrix_1, observation_matrix,
                                        observation_error_cov, model_error_cov_1_2, observations[i])
    likelihood_1_2 = norm.pdf(observations[i], (observation_matrix @ states_1_2[0]),
                                np.sqrt((observation_matrix @ states_1_2[1] @ observation_matrix.T + observation_error_cov)))

    # apply kalman filter for regime 2 to regime 2
    states_2_2 = kalman_filter(pred_mean_s2[-1], pred_cov_s2[-1], transition_matrix_2, observation_matrix,
                                        observation_error_cov, model_error_cov_2_2, observations[i])
    likelihood_2_2 = norm.pdf(observations[i], (observation_matrix @ states_2_2[0]),
                                np.sqrt((observation_matrix @ states_2_2[1] @ observation_matrix.T + observation_error_cov)))

    # apply kalman filter for regime 2 to regime 1
    states_2_1 = kalman_filter(pred_mean_s2[-1], pred_cov_s2[-1], transition_matrix_2, observation_matrix,
                                        observation_error_cov, model_error_cov_2_1, observations[i])
    likelihood_2_1 = norm.pdf(observations[i], (observation_matrix @ states_2_1[0]),
                                np.sqrt((observation_matrix @ states_2_1[1] @ observation_matrix.T + observation_error_cov)))

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

    # print(sum([M_1_1, M_1_2, M_2_1, M_2_2])) # check if the sum of the probabilities is 1

    # get the new knowledge probability vector
    M1.append(np.squeeze(M_1_1 + M_2_1))
    M2.append(np.squeeze(M_1_2 + M_2_2))
    knowledge_probability = np.array([[M1[-1]], [M2[-1]]])

    # knowledge_probability = knowledge_probability.reshape(2, 1)
    probability.append(knowledge_probability)

    # get the posterior mean and covariance for regime 1
    w_1_1 = M_1_1 / (M_1_1 + M_2_1)
    w_2_1 = M_2_1 / (M_1_1 + M_2_1)
    posterior_mean_1, posterior_cov_1 = gaussian_mixture(states_1_1[2], states_2_1[2], states_1_1[3], states_2_1[3], w_1_1, w_2_1)
    pred_mean_s1.append(posterior_mean_1)
    pred_cov_s1.append(posterior_cov_1)

    # get the posterior mean and covariance for regime 2
    w_1_2 = M_1_2 / (M_1_2 + M_2_2)
    w_2_2 = M_2_2 / (M_1_2 + M_2_2)
    posterior_mean_2, posterior_cov_2 = gaussian_mixture(states_1_2[2], states_2_2[2], states_1_2[3], states_2_2[3], w_1_2, w_2_2)
    pred_mean_s2.append(posterior_mean_2)
    pred_cov_s2.append(posterior_cov_2)
    # print(sum([w_1_1, w_2_2, w_1_2, w_2_1])) # check if the probabilities are correct sum to 2

    # get the new knowledge mean and covariance
    # print(knowledge_probability[0]+knowledge_probability[1]) # check if the sum of the probabilities is 1
    knowledge_mean, knowledge_cov = gaussian_mixture(posterior_mean_1, posterior_mean_2, posterior_cov_1, posterior_cov_2, knowledge_probability[0], knowledge_probability[1])
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


