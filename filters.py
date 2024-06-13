import numpy as np


class KalmanFilter:
    """Kalman filter class with rts smoother"""

    def __init__(self, A, Q, C, R, x, P):
        self.z = None
        self.x = x  # initial state
        self.P = P  # initial state covariance matrix
        self.A = A  # train matrix
        self.Q = Q  # transition covariance matrix
        self.C = C  # observation matrix
        self.R = R  # observation covariance matrix
        self.x_filt = [x]  # filtered state
        self.P_filt = [P]  # filtered state covariance matrix
        self.x_pred = [x]  # predicted state
        self.P_pred = [P]  # predicted state covariance matrix
        self.x_smooth = []  # smoothed state
        self.P_smooth = []  # smoothed state covariance matrix

    def filter(self, z):
        self.z = z  # observation data

        # predict a step of KF
        x_ = self.A @ self.x  # predicted state
        P_ = self.A @ self.P @ self.A.T + self.Q  # predicted state covariance matrix

        self.x_pred.append(x_.copy())
        self.P_pred.append(P_.copy())

        # update a step of KF
        K = P_ @ self.C.T @ np.linalg.pinv(self.C @ P_ @ self.C.T + self.R)  # Kalman gain
        x_filt = x_ + K @ (self.z - self.C @ x_)  # filtered state
        P_filt = (np.identity(np.size(self.C)) - K @ self.C) @ P_  # filtered state covariance matrix

        self.x_filt.append(x_filt.copy())
        self.P_filt.append(P_filt.copy())

        self.x = x_filt.copy()
        self.P = P_filt.copy()

    def smooth(self):
        # apply RTS smoother
        self.x_smooth = self.x_filt.copy()
        self.P_smooth = self.P_filt.copy()
        for i in range(len(self.x_filt) - 2, -1, -1):
            J = self.P_filt[i] @ self.A.T @ np.linalg.pinv(self.P_pred[i + 1])
            self.x_smooth[i] = self.x_filt[i] + J @ (self.x_smooth[i + 1] - self.x_pred[i + 1])
            self.P_smooth[i] = self.P_filt[i] + J @ (self.P_smooth[i + 1] - self.P_pred[i + 1]) @ J.T


class FixLagSmoother:
    """Fixed-lag smoother class"""

    def __init__(self, N, A, Q, C, R, x, P):
        self.z = None
        self.N = N  # lag window size
        self.x = x  # initial state
        self.P = P  # initial state covariance matrix
        self.A = A  # train matrix
        self.Q = Q  # transition covariance matrix
        self.C = C  # observation matrix
        self.R = R  # observation covariance matrix
        # self.x_pred = [x] # predicted state
        # self.P_pred = [P] # predicted state covariance matrix
        self.x_smooth = []  # smoothed state
        self.Sigma_smooth = []  # smoothed state covariance matrix

        self.count = 0  # counter

    def smooth(self, z):
        self.z = z  # observation data

        k = self.count

        # predict step of KF (a priori estimate)
        x_pred = self.A @ self.x  # predicted state
        P_pred = self.A @ self.P @ self.A.T + self.Q  # predicted state covariance matrix

        # self.x_pred.append(x_pred.copy())
        # self.P_pred.append(P_pred.copy())

        # update step of KF (a posteriori estimate)
        K = P_pred @ self.C.T @ np.linalg.pinv(self.C @ P_pred @ self.C.T + self.R)  # Standard Kalman gain
        x_filt = x_pred + K @ (self.z - self.C @ x_pred)  # filtered state
        P_filt = P_pred - K @ self.C @ P_pred  # filtered state covariance matrix

        self.x_smooth.append(x_pred.copy())
        self.Sigma_smooth.append(P_pred.copy())

        # apply smoothing
        if k >= self.N:
            # P_smooth = P_pred.copy()
            # for i in range(self.N):
            #     K_i = P_smooth @ self.C.T @ np.linalg.pinv(self.C @ P_pred @ self.C.T + self.R) # smooth Kalman gain
            #     P_smooth = P_smooth @ (self.A - K @ self.C).T
            #
            #     si = k-i # index of smoothed state
            #     self.x_smooth[si] = self.x_smooth[si] + K_i @ (self.z - self.C @ x_pred) # smoothed state
            #     self.Sigma_smooth[si] = self.Sigma_smooth[si] - P_smooth @ C.T @ K_i.T # smooth state covariance matrix
            #
            for i in range(self.N):
                si = k - i  # index of smoothed state

                K_i = self.Sigma_smooth[si] @ self.C.T @ np.linalg.pinv(
                    self.C @ P_pred @ self.C.T + self.R)  # smooth Kalman gain
                P_smooth = self.Sigma_smooth[si] @ (self.A - K @ self.C).T

                self.x_smooth[si] = self.x_smooth[si] + K_i @ (self.z - self.C @ x_pred)  # smoothed state
                self.Sigma_smooth[si] = self.Sigma_smooth[si] - P_smooth @ self.C.T @ K_i.T  # smooth state covariance matrix

        else:
            self.x_smooth[k] = x_filt.copy()
            self.Sigma_smooth[k] = P_filt.copy()

        self.count += 1
        self.x = x_filt.copy()
        self.P = P_filt.copy()


#TODO: Prepare the class for the online smoother
class OnlineSmoother:
    """Online smoother class based on implementing RTS in fixed-lag smoother"""
    """The online smoother will be implemented in the next version of the code. Now you can use RTSxFLS.py"""

    def __init__(self, A, Q, C, R, x, P, N):
        self.z = None
        self.x = x  # initial state
        self.P = P  # initial state covariance matrix
        self.A = A  # train matrix
        self.Q = Q  # transition covariance matrix
        self.C = C  # observation matrix
        self.R = R  # observation covariance matrix
        self.x_final = []  # used to store filtered and smoothed states
        self.P_final = []  # used to store filtered and smoothed state covariance matrices
        self.N = N  # lag window size

        self.buffer_x = [x]
        self.buffer_P = [P]

        self.buffer_x_pred = [x]
        self.buffer_P_pred = [P]

        self.j = 0  # counter for states in buffer
        self.i = 0  # counter for observations

    def filter(self, x_, P_, z):
        # predict step of KF (a priori estimate)
        x_pred = self.A @ x_  # predicted state
        P_pred = self.A @ P_ @ self.A.T + self.Q  # predicted state covariance matrix

        # update step of KF (a posteriori estimate)
        K = P_pred @ self.C.T @ np.linalg.pinv(self.C @ P_pred @ self.C.T + self.R)  # Standard Kalman gain
        x_filter = x_pred + K @ (z - self.C @ x_pred)  # filtered state
        P_filter = P_pred - K @ self.C @ P_pred  # filtered state covariance matrix

        return x_filter, P_filter, x_pred, P_pred

    def smooth(self, x_, P_, xpred, Ppred):
        x_smooth = x_.copy()
        P_smooth = P_.copy()
        for i in range(len(x_) - 2, -1, -1):
            J = P_[i] @ self.A.T @ np.linalg.pinv(Ppred[i + 1])
            x_smooth[i] = x_[i] + J @ (x_smooth[i + 1] - xpred[i + 1])
            P_smooth[i] = P_[i] + J @ (P_smooth[i + 1] - Ppred[i + 1]) @ J.T

        return x_smooth, P_smooth
    def lag_filter(self, z):
        self.z = z

        while self.i <= len(z) - 1:
            x, P, x_pred, P_pred = self.filter(self.buffer_x[self.j], self.buffer_P[self.j], z[self.i])

            self.i += 1
            self.j += 1
            self.buffer_x.append(x)
            self.buffer_P.append(P)
            self.buffer_x_pred.append(x_pred)
            self.buffer_P_pred.append(P_pred)

            if len(self.buffer_x) == self.N:
                x, P = self.smooth(self.buffer_x, self.buffer_P, self.buffer_x_pred, self.buffer_P_pred)
                self.x_final.append(x[0])
                self.P_final.append(P[0])

                self.buffer_x = [x[1]]
                self.buffer_P = [P[1]]
                self.buffer_x_pred = [x[1]]
                self.buffer_P_pred = [P[1]]

                # update i
                self.i = self.i - self.N + 2
                self.j = 0

            if self.i == len(z) and len(self.buffer_x) < self.N:

                # append to the final results
                self.x_final.extend(self.buffer_x)
                self.P_final.extend(self.buffer_P)