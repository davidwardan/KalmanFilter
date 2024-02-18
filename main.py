# import libraries
import numpy as np
import matplotlib.pyplot as plt
import math

# import needed functions
from filter import apply_filter

################# Test on example form book ################
knowledge_mean = np.array([[10], [-1]])
knowledge_cov = np.array([[2**2, 0], [0, 1.5**2]])

transition_matrix = np.array([[1, 1], [0, 1]])
model_error_cov = np.array([[0.25, 0.5], [0.5, 1]]) * 0.1

observation_matrix = np.array([[1, 0]])
observation_error_cov = 4**2

# create time steps with 5 seconds interval
t = np.arange(0, 6)

# create observations from hidden states
observations = np.array([5, 11, 8, 6, 5])
###########################################################

# Apply kalman filter
pred_mean, pred_cov = apply_filter(t, knowledge_mean, knowledge_cov, transition_matrix, observation_matrix,
                                    observation_error_cov, model_error_cov, observations, smoothing = False)


# Apply kalman filter with smoothing
pred_mean_smoothed, pred_cov_smoothed = apply_filter(t, knowledge_mean, knowledge_cov, transition_matrix, observation_matrix,
                                    observation_error_cov, model_error_cov, observations, smoothing = True)

# plot first hidden state and observations
figure1, axis = plt.subplots()
axis.plot(t, [i[0][0] for i in pred_mean], label='Predicted Mean', color='red', marker='o')
axis.plot(t, [i[0][0] for i in pred_mean_smoothed], label='Smoothed Mean', color='blue', marker='x')
axis.scatter(t[1:], observations, label='Observations', color='green', marker='s')
axis.fill_between(t, [i[0][0] - math.sqrt(j[0][0]) for i, j in zip(pred_mean, pred_cov)], [i[0][0] + math.sqrt(j[0][0]) for i, j in zip(pred_mean, pred_cov)], color='red', alpha=0.2)
axis.fill_between(t, [i[0][0] - math.sqrt(j[0][0]) for i, j in zip(pred_mean_smoothed, pred_cov_smoothed)], [i[0][0] + math.sqrt(j[0][0]) for i, j in zip(pred_mean_smoothed, pred_cov_smoothed)], color='blue', alpha=0.2)
axis.set_ylim([0, 20])
axis.legend()
plt.show()

# plot second hidden state
figure2, axis = plt.subplots()
axis.plot(t, [i[1][0] for i in pred_mean], label='Predicted Mean', color='red', marker='o')
axis.plot(t, [i[1][0] for i in pred_mean_smoothed], label='Smoothed Mean', color='blue', marker='x')
axis.fill_between(t, [i[1][0] - math.sqrt(j[1][1]) for i, j in zip(pred_mean, pred_cov)], [i[1][0] + math.sqrt(j[1][1]) for i, j in zip(pred_mean, pred_cov)], color='red', alpha=0.2)
axis.fill_between(t, [i[1][0] - math.sqrt(j[1][1]) for i, j in zip(pred_mean_smoothed, pred_cov_smoothed)], [i[1][0] + math.sqrt(j[1][1]) for i, j in zip(pred_mean_smoothed, pred_cov_smoothed)], color='blue', alpha=0.2)
axis.set_ylim([-5, 5])
axis.legend()
plt.show()