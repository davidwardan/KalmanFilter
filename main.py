# import libraries
import numpy as np
import matplotlib.pyplot as plt
import math

# import needed functions
from filter import apply_filter

################# Test on example form book ################
knowledge_mean = np.array([[10],
                           [-1]])
knowledge_cov = np.array([[2**2, 0],
                          [0, 1.5**2]])
transition_matrix = np.array([[1, 1],
                              [0, 1]])
model_error_cov = np.array([[0.25, 0.5],
                             [0.5, 1]]) * 0.1**2

observation_matrix = np.array([[1, 0]])
observation_error_cov = 1**2

# create time steps with 5 seconds interval
t = np.arange(0, 3)

# create observations from hidden states
# np.random.seed(235)

hidden_states = [knowledge_mean]
for i in range(1, len(t)):
    hidden_states.append(transition_matrix @ hidden_states[i-1] + np.random.normal(0, 0.001))
observations = []
for i in range(1,len(hidden_states)):
    y = observation_matrix @ hidden_states[i] + np.random.normal(0, observation_error_cov)
    observations.append(y[0][0])

###########################################################

# Apply kalman filter
pred_mean, pred_cov = apply_filter(t, knowledge_mean, knowledge_cov, transition_matrix, observation_matrix,
                                    observation_error_cov, model_error_cov, observations, smoothing = False)

# Apply kalman filter with smoothing
pred_mean_smoothed, pred_cov_smoothed = apply_filter(t, knowledge_mean, knowledge_cov, transition_matrix, observation_matrix,
                                    observation_error_cov, model_error_cov, observations, smoothing = True)

# plot first hidden state and observations
figure1, axis = plt.subplots()
axis.plot(t, [i[0][0] for i in pred_mean], label='Predicted Mean', color='red')
axis.plot(t, [i[0][0] for i in pred_mean_smoothed], label='Smoothed Mean', color='blue')
axis.errorbar(t[1:], observations,yerr = [observation_error_cov] * len(observations), fmt='o', color='green', capsize=3, markersize=3, label='Observation')
axis.fill_between(t, [i[0][0] - math.sqrt(j[0][0]) for i, j in zip(pred_mean, pred_cov)], [i[0][0] + math.sqrt(j[0][0]) for i, j in zip(pred_mean, pred_cov)], color='red', alpha=0.2)
axis.fill_between(t, [i[0][0] - math.sqrt(j[0][0]) for i, j in zip(pred_mean_smoothed, pred_cov_smoothed)], [i[0][0] + math.sqrt(j[0][0]) for i, j in zip(pred_mean_smoothed, pred_cov_smoothed)], color='blue', alpha=0.2)
axis.title.set_text('First Hidden State')
axis.legend()

# plot second hidden state
figure2, axis = plt.subplots()
axis.plot(t, [i[1][0] for i in pred_mean], label='Predicted Mean', color='red')
axis.plot(t, [i[1][0] for i in pred_mean_smoothed], label='Smoothed Mean', color='blue')
axis.fill_between(t, [i[1][0] - math.sqrt(j[1][1]) for i, j in zip(pred_mean, pred_cov)], [i[1][0] + math.sqrt(j[1][1]) for i, j in zip(pred_mean, pred_cov)], color='red', alpha=0.2)
axis.fill_between(t, [i[1][0] - math.sqrt(j[1][1]) for i, j in zip(pred_mean_smoothed, pred_cov_smoothed)], [i[1][0] + math.sqrt(j[1][1]) for i, j in zip(pred_mean_smoothed, pred_cov_smoothed)], color='blue', alpha=0.2)
axis.title.set_text('Second Hidden State')
axis.legend()
plt.show()