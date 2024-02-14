# import libraries
import numpy as np
import matplotlib.pyplot as plt
import math

# import needed functions
from filter import kalman_filter

################# Test on example form book ################
knowledge_mean = np.array([[10], [-1]])
knowledge_cov = np.array([[2**2, 0], [0, 1.5**2]])

transition_matrix = np.array([[1, 1], [0, 1]])
model_error_cov = np.array([[0.25, 0.5], [0.5, 1]]) * 0.1

observation_matrix = np.array([[1, 0]])
observation_error_cov = 4**2

t = [0, 1, 2, 3, 4, 5]
observations = [5, 11, 8, 6, 5]
###########################################################

# Apply kalman filter
pred_mean, pred_cov = kalman_filter(t, knowledge_mean, knowledge_cov, transition_matrix, observation_matrix,
                                    observation_error_cov, model_error_cov, observations, smoothing = False)

# With smoothing
pred_mean_smoothed, pred_cov_smoothed = kalman_filter(t, knowledge_mean, knowledge_cov, transition_matrix, observation_matrix,
                                       observation_error_cov, model_error_cov, observations, smoothing = True)

# Plot results
figure, axis = plt.subplots()

def get_data(mean,cov):
    # prepare data for plotting
    state1_mean = []
    state2_mean = []
    state1_cov = []
    state2_cov = []
    for i in mean:
        state1_mean.append(i[0][0])
        state2_mean.append(i[1][0])
    for i in cov:
        state1_cov.append(math.sqrt(i[0][0]))
        state2_cov.append(math.sqrt(i[1][1]))
    return state1_mean, state2_mean, state1_cov, state2_cov

state1_mean, state2_mean, state1_cov, state2_cov = get_data(pred_mean, pred_cov)
state1_mean_s, state2_mean_s, state1_cov_s, state2_cov_s = get_data(pred_mean_smoothed, pred_cov_smoothed)

# Plotting the first state
plt.figure(1)
plt.plot(t, state1_mean, label='Kalman Filter', color='red', marker='o')
plt.plot(t, state1_mean_s, label='Smoothed', color='blue', marker='x')
# Filling the area between the two lines
plt.fill_between(t, np.array(state1_mean)+np.array(state1_cov), np.array(state1_mean)-np.array(state1_cov), color='red', alpha=0.2)
plt.fill_between(t, np.array(state1_mean_s)+np.array(state1_cov_s), np.array(state1_mean_s)-np.array(state1_cov_s), color='blue', alpha=0.2)
# Plotting the second line
plt.scatter(range(t[1], t[-1]+1), observations, label='Observations', color='green', marker='s')
# Adding labels and title
plt.xlabel('Time - t')
plt.ylabel('State - xt')
# Adding a legend
plt.legend()
# Setting y-axis limits
plt.ylim(-10, 15)
# Display the plot
plt.show()

# Plotting the second state
plt.figure(2)
plt.plot(t, state2_mean, label='Kalman Filter', color='red', marker='o')
plt.plot(t, state2_mean_s, label='Smoothed', color='blue', marker='o')

# Filling the area between the two lines
plt.fill_between(t, np.array(state2_mean)+np.array(state2_cov), np.array(state2_mean)-np.array(state2_cov), color='red', alpha=0.2)
plt.fill_between(t, np.array(state2_mean_s)+np.array(state2_cov_s), np.array(state2_mean_s)-np.array(state2_cov_s), color='blue', alpha=0.2)

# Adding labels and title
plt.xlabel('Time - t')
plt.ylabel('Rate of change - ẋt')
# Adding a legend
plt.legend()
# Setting y-axis limits
plt.ylim(-4, 4)
# Display the plot
plt.show()