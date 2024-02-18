# import libraries
import numpy as np
import matplotlib.pyplot as plt
import math

# import needed functions
from filter import kalman_filter

np.random.seed(235)

# create dataset from given
knowledge_mean = np.array([[10], [0]])
knowledge_cov = np.array([[0.01**2, 0], [0, 0.001**2]])
knowledge_probability = np.array([[0.5], [0.5]])

transition_matrix_1 = np.array([[1, 0], [0, 0]])
transition_matrix_2 = np.array([[1, 1], [0, 1]])

model_error_cov_1_1, model_error_cov_2_2, model_error_cov_2_1 = 0, 0, 0
model_error_cov_1_2 = np.array([[0, 0], [0, (2.3*10**-4)**2]])

observation_matrix = np.array([[1, 0]])
observation_error_cov = 0.5**2

# create time steps with 5 seconds interval
t = np.arange(0, 1000)

# Calculate hidden states using the transition matrix
hidden_states = [knowledge_mean]
for i in range(1, len(t)):
    if i < 500:
        hidden_states.append(transition_matrix_1 @ hidden_states[i-1] + np.random.normal(0, 3))
    else:
        hidden_states.append(transition_matrix_2 @ hidden_states[i-1] + np.random.normal(0, 3))


# create observations from hidden states
observations = []
for i in range(len(hidden_states)):
    observations.append(observation_matrix @ hidden_states[i] + np.random.normal(0, observation_error_cov))

# plot hidden states and observations
figure, axis = plt.subplots()
# axis.plot(t, [i[0][0] for i in hidden_states], label='Hidden State 1', color='red', marker='o')
# axis.plot(t, [i[1][0] for i in hidden_states], label='Hidden State 2', color='blue', marker='x')
axis.scatter(t, observations, label='Observations', color='green', marker='s')
axis.legend()
plt.show()

# Apply switching kalman filter

# Apply kalman filter regime 1
pred_mean_1, pred_cov_1 = kalman_filter(t, knowledge_mean, knowledge_cov, transition_matrix_1, observation_matrix,
                                    observation_error_cov, model_error_cov_1_1, observations, smoothing = False)

# Apply kalman filter regime 2
pred_mean_2, pred_cov_2 = kalman_filter(t, knowledge_mean, knowledge_cov, transition_matrix_2, observation_matrix,
                                    observation_error_cov, model_error_cov_2_2, observations, smoothing = False)








