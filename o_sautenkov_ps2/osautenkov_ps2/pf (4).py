"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform
from scipy.stats import norm as gaussian

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle

class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, beta, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, beta)

        # TODO add here specific class variables for the PF

        self.num_p = num_particles
        self.global_loc = global_localization

        self.X = np.array([self.mu] * self.num_p)
        self.weight = np.ones(self.num_p)

    def predict(self, u):
        # TODO Implement here the PF, perdiction part

        for i in range(self.num_p):
            self.X[i] = sample_from_odometry(self.X[i], u, self._alphas)

        self._state_bar = get_gaussian_statistics(self.X)

    def update(self, z):
        # TODO implement correction step

        lm_id = int(z[1])

        dx = self._field_map.landmarks_poses_x[lm_id] - self.mu_bar[0]
        dy = self._field_map.landmarks_poses_y[lm_id] - self.mu_bar[1]
        dn = dx**2 + dy**2

        H = np.array([[dy/dn, -dx/dn, -1]])

        for i in range(self.num_p):
            z_bar = get_observation(np.squeeze(self.X[i]), z[1])[0]
            self.weight[i] = gaussian.pdf((z[0]-z_bar), 0, np.sqrt(H.dot(self.Sigma_bar).dot(H.T) + self._Q))

        self.weight = self.weight / np.sum(self.weight)

        R = uniform(0, 1 / self.num_p)
        c = self.weight[0]
        X = np.zeros_like(self.X)
        i = 0

        for m in range(self.num_p):
            U = R + m / self.num_p
            while U > c:
                i += 1
                c += self.weight[i]
            X[m] = self.X[i]

        self.X = X
        self._state = get_gaussian_statistics(self.X)