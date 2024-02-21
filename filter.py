# import libraries
import numpy as np

# Define prediction step
def predict_step(knowledge_mean, knowledge_cov, transition_matrix, model_error_cov):
    prior_mean = transition_matrix @ knowledge_mean
    prior_cov = transition_matrix @ knowledge_cov @ transition_matrix.T + model_error_cov
    return prior_mean, prior_cov

# Define update step
def update_step(prior_mean, prior_cov, observation, observation_matrix, observation_error_cov):
    innovation_cov = observation_matrix @ prior_cov @ observation_matrix.T + observation_error_cov
    kalman_gain = prior_cov @ observation_matrix.T @  np.linalg.inv(innovation_cov)
    pred_obs = observation_matrix @ prior_mean
    innovation = observation - pred_obs

    posterior_mean = prior_mean + kalman_gain @ innovation
    posterior_cov = (np.identity(np.size(observation_matrix)) - kalman_gain @ observation_matrix) @ prior_cov

    return posterior_mean, posterior_cov

# Define smoother function
def smooth(mean, cov, mean_T, cov_T,pred_mean, pred_cov, transition_matrix):
    J = cov @ transition_matrix.T @ np.linalg.pinv(pred_cov)
    updated_mean = mean + J @ (mean_T - pred_mean)
    updated_cov = cov + J @ (cov_T - pred_cov) @ J.T

    return updated_mean, updated_cov

# Define kalman filter
def kalman_filter(knowledge_mean, knowledge_cov, transition_matrix, observation_matrix, observation_error_cov, model_error_cov, observations):
    prior_mean, prior_cov = predict_step(knowledge_mean, knowledge_cov, transition_matrix, model_error_cov)
    posterior_mean, posterior_cov = update_step(prior_mean, prior_cov, observations, observation_matrix, observation_error_cov)

    return [prior_mean, prior_cov, posterior_mean, posterior_cov]

# Define function to apply filter and smoother
def apply_filter(t, knowledge_mean, knowledge_cov, transition_matrix, observation_matrix, observation_error_cov, model_error_cov, observations, smoothing = False):
    # Apply kalman filter
    prior_mean = []
    prior_cov = []
    pred_mean = [knowledge_mean]
    pred_cov = [knowledge_cov]
    # Loop through the observations
    for i in range(len(observations)):
        # store the states into the lists by calling the kalman filter function
        states = kalman_filter(pred_mean[-1], pred_cov[-1], transition_matrix, observation_matrix,
                                            observation_error_cov, model_error_cov, observations[i])

        # set the new knowledge mean and covariance
        pred_mean.append(states[2])
        pred_cov.append(states[3])

        prior_mean.append(states[0])
        prior_cov.append(states[1])

    # Define switching kalman filter
    if smoothing == True:
        # Apply smoothing
        for i in range(len(observations)-1, -1, -1):
            mean_smooth, cov_smooth = smooth(pred_mean[i], pred_cov[i], pred_mean[i+1], pred_cov[i+1], prior_mean[i], prior_cov[i], transition_matrix)
            pred_mean[i] = mean_smooth
            pred_cov[i] = cov_smooth
    return pred_mean, pred_cov