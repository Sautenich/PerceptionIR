"""
This file implements the Extended Kalman Filter.
"""

import numpy as np

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle
from field_map import FieldMap


class EKF(LocalizationFilter):
    def predict(self, u):
        # TODO Implement here the EKF, perdiction part. HINT: use the auxiliary functions imported above from tools.task

        self._state_bar.mu = get_prediction(self.mu, u)[np.newaxis].T
        alphas = self._alphas
        G = np.array([[1, 0, -u[1]*np.sin(wrap_angle(self.mu[2]+u[0]))], [0, 1, u[1]*np.cos(wrap_angle(self.mu[2]+u[0]))], [0, 0, 1]])
        V = np.array([[-u[1]*np.sin(self.mu[2]+u[0]), np.cos(self.mu[2]+u[0]), 0],
                      [u[1]*np.cos(self.mu[2]+u[0]), np.sin(self.mu[2]+u[0]), 0],
                      [1, 0, 1]])
        # self._state_bar.Sigma = G.dot(self.Sigma).dot(G.T) + get_motion_noise_covariance(u, alphas)
        self._state_bar.Sigma = G.dot(self.Sigma).dot(G.T) + V.dot(get_motion_noise_covariance(u, alphas)).dot(V.T)

    def update(self, z):
        # TODO implement correction step
        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma

        field_map = FieldMap()
        lm_id = int(z[1])

        dx = field_map.landmarks_poses_x[lm_id] - self.mu_bar[0]
        dy = field_map.landmarks_poses_y[lm_id] - self.mu_bar[1]
        dn = dx**2+dy**2

        z_bar = get_expected_observation(np.squeeze(self.mu_bar), z[1])[0]

        H = np.array([[dy/dn, -dx/dn, -1]])
        S = H.dot(self.Sigma_bar).dot(H.T) + self._Q
        K = (self.Sigma_bar.dot(H.T) * S**(-1)).reshape(3, 1)

        self._state.mu = self.mu_bar.reshape(3, 1) + K * (z[0] - z_bar)
        self._state.Sigma = (np.eye(3, 3) - K.dot(H)).dot(self.Sigma_bar)