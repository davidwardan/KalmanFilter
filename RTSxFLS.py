# import libraries
import matplotlib.pyplot as plt
import numpy as np

# set plot parameters
plt.rcParams.update({'font.size': 16})
plt.rcParams['text.usetex'] = True

# define a function to apply the kalman filter
def filter(A, Q, C, R, x, P, z):
    # predict step of KF (a priori estimate)
    x_pred = A @ x # predicted state
    P_pred = A @ P @ A.T + Q # predicted state covariance matrix
    
    # update step of KF (a posteriori estimate)
    K = P_pred @ C.T @ np.linalg.pinv(C @ P_pred @ C.T + R) # Standard Kalman gain
    x_filt = x_pred + K @ (z - C @ x_pred) # filtered state
    P_filt = P_pred - K @ C @ P_pred # filtered state covariance matrix
    
    return x_filt, P_filt, x_pred, P_pred

def smooth(A, x_filt, P_filt, x_pred, P_pred):
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()
    for i in range(len(x_filt)-2, -1, -1):
        J = P_filt[i] @ A.T @ np.linalg.pinv(P_pred[i+1])
        x_smooth[i] = x_filt[i] + J @ (x_smooth[i+1] - x_pred[i+1])
        P_smooth[i] = P_filt[i] + J @ (P_smooth[i+1] - P_pred[i+1]) @ J.T

    return x_smooth, P_smooth

# define parameters
N = 10 # lag window size
A = np.array([[1, 1], [0, 1]]) # train matrix
Q = np.array([[0.25, 0.5], [0.5, 1]]) * 0.1**2 # transition covariance matrix
C = np.array([[1, 0]]) # observation matrix
R = np.array([[4]]) # observation covariance matrix

x = np.array([1, -1]) # initial state
P = np.array([[2**2, 0],
                          [0, 1.5**2]]) # initial state covariance matrix
# generate data
np.random.seed(0)
n = 50
t = np.linspace(0, n, n)
z = t + np.random.normal(0, R[0], n)

x_final = [] # used to store filtered and smoothed states
P_final = [] # used to store filtered and smoothed state covariance matrices

# initialize buffers
buffer_x = [x]
buffer_P = [P]

buffer_x_pred = [x]
buffer_P_pred = [P]

j = 0 # counter for states in buffer
i = 0 # counter for observations
while i <= len(z)-1:
    x, P, x_pred, P_pred = filter(A, Q, C, R, buffer_x[j], buffer_P[j], z[i])
    i += 1
    j += 1
    buffer_x.append(x)
    buffer_P.append(P)
    buffer_x_pred.append(x_pred)
    buffer_P_pred.append(P_pred)

    if len(buffer_x) == N:
        x, P = smooth(A, buffer_x, buffer_P, buffer_x_pred, buffer_P_pred)
        x_final.append(x[0])
        P_final.append(P[0])

        buffer_x = [x[1]]
        buffer_P = [P[1]]
        buffer_x_pred = [x[1]]
        buffer_P_pred = [P[1]]

        # update i
        i = i - N + 2
        j = 0

    if i == len(z) and len(buffer_x) < N:
        # buffer_x.pop(0)
        # buffer_P.pop(0)

        # append to the final results
        x_final.extend(buffer_x)
        P_final.extend(buffer_P)

x = np.array(x_final)[:, 0]
P = np.array(P_final)[:, 0, 0]

t = range(len(z)+1)
# plot results
# plt.figure(figsize=(10, 6))
plt.scatter(t[1:], z, label='Observations', color='steelblue')
plt.plot(t, x, label='RTSxFLS states $N$='+str(N), color='green')
plt.fill_between(t, x - 2*np.sqrt(P), x + 2*np.sqrt(P), color='g', alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Value')
# plt.title('First State')
plt.legend(loc='upper left', fontsize=12)
plt.savefig('FLSxRTS.png', dpi=300, bbox_inches='tight')

x = np.array(x_final)[:, 1]
P = np.array(P_final)[:, 1, 1]

# plot results
plt.figure(figsize=(10, 6))
plt.plot(t, x, label='FLS states', color='red')
plt.fill_between(t, x - 2*np.sqrt(P), x + 2*np.sqrt(P), color='r', alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Second State')
plt.legend()
plt.show()