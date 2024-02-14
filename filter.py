# import libraries
import numpy as np

# Define prediction step
def predict_step(knowledge_mean, knowledge_cov, transition_matrix, model_error_cov):
    prior_mean = transition_matrix @ knowledge_mean
    prior_cov = np.matmul(np.matmul(transition_matrix, knowledge_cov), np.transpose(transition_matrix)) + model_error_cov

    return prior_mean, prior_cov

# Define update step
def update_step(prior_mean, prior_cov, observation, observation_matrix, observation_error_cov):
    innovation_cov = np.matmul(observation_matrix @ prior_cov, np.transpose(observation_matrix)) + observation_error_cov
    kalman_gain = np.matmul(prior_cov @ np.transpose(observation_matrix), np.linalg.inv(innovation_cov))
    pred_obs = observation_matrix @ prior_mean
    innovation = observation - pred_obs

    posterior_mean = prior_mean + np.matmul(kalman_gain, innovation)
    posterior_cov = np.matmul((np.identity(np.size(observation_matrix)) - kalman_gain @ observation_matrix), prior_cov)

    return posterior_mean, posterior_cov

# Define smoother function
def smooth(mean, cov, mean_T, cov_T,pred_mean, pred_cov, transition_matrix):

    J = cov @ transition_matrix.T @ np.linalg.pinv(pred_cov)
    updated_mean = mean + J @ (mean_T - pred_mean)
    updated_cov = cov + J @ (cov_T - pred_cov) @ J.T

    return updated_mean, updated_cov

# Define kalman filter
def kalman_filter(t, knowledge_mean, knowledge_cov, transition_matrix, observation_matrix, observation_error_cov, model_error_cov, observations, smoothing = False):
    pred_mean = []
    pred_cov = []

    mean = [knowledge_mean]
    cov = [knowledge_cov]
    # iterate over each time step
    for i in range(0, len(observations)):
        prior_mean, prior_cov = predict_step(mean[-1], cov[-1], transition_matrix, model_error_cov)
        pred_mean.append(prior_mean)
        pred_cov.append(prior_cov)

        updated_mean, updated_cov = update_step(pred_mean[-1], pred_cov[-1], np.array([observations[i]]), observation_matrix, observation_error_cov)

        mean.append(updated_mean)
        cov.append(updated_cov)

    if smoothing == True:
        # Apply smoothing
        for i in range(len(observations)-1, -1, -1):
            mean_smooth, cov_smooth = smooth(mean[i], cov[i], mean[i+1], cov[i+1], pred_mean[i], pred_cov[i], transition_matrix)
            mean[i] = mean_smooth
            cov[i] = cov_smooth
    return mean, cov