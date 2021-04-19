"""
Gonzalo Ferrer
g.ferrer@skoltech.ru
28-Feb-2021
"""

import numpy as np
import mrob
from scipy.linalg import inv
from slam.slamBase import SlamBase
from tools.task import get_motion_noise_covariance
from tools.jacobian import state_jacobian, augmented_jacobian, inverse_observation_jacobian, observation_jacobian, observation_augmented_jacobian



class Sam(SlamBase):
    def __init__(self, initial_state, alphas, state_dim=3, obs_dim=2, landmark_dim=2, action_dim=3, *args, **kwargs):
        super(Sam, self).__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.landmark_dim = landmark_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.alphas = alphas
        self.graph = mrob.FGraph()
        self.cur_state = initial_state.mu
        self.Sigma = initial_state.Sigma
        self.N_zero = self.graph.add_node_pose_2d(self.cur_state)
        W = self.Sigma
        self.graph.add_factor_1pose_2d(self.cur_state, self.N_zero, inv(W))
        self.ci2 = []
        self.LEHRBUCH = {}
        # self.graph.print(True)
        # self.graph.solve()


    def predict(self, u):
    	print('No_added odo:')
    	print(self.graph.get_estimated_state())
    	print('\n \n')
    	self.states_oper()
    	N_cur = self.graph.add_node_pose_2d(np.zeros(3))
    	J_u = state_jacobian(self.cur_state.T[0], u)[1]
    	cov = get_motion_noise_covariance(u, self.alphas)
    	inv_cov = J_u.dot(cov).dot(J_u.T)
    	# self.graph.print(True)
    	self.graph.add_factor_2poses_2d_odom(u, self.N_zero, N_cur, inv(inv_cov))
    	self.N_zero = N_cur
    	print('Added odo:')
    	print(self.graph.get_estimated_state())


    def update(self, z):
    	for i in range(z.shape[0]):
    		key = False
    		if int(z[i, 2]) not in self.LEHRBUCH:
    			NodeLM_ID = self.graph.add_node_landmark_2d(np.zeros(2))
    			key = True
    			self.LEHRBUCH[int(z[i, 2])] = NodeLM_ID
    		W_z = inv(self.Q)
    		self.graph.add_factor_1pose_1landmark_2d(np.array([[z[i, 0]], [z[i, 1]]]), self.N_zero, self.LEHRBUCH[int(z[i,2])], W_z, initializeLandmark=key)
    	# for i in self.graph.get_estimated_state():
    	# 	print(i)
    	# 	print('\n \n')

    def states_oper(self):
    	all_states = self.graph.get_estimated_state()
    	all_states.reverse()
    	for i in all_states:
    		if i.shape[0] == 3:
    			self.cur_state = i
    			return


    def solve(self):
    	# print('Before optimization:')
    	# for i in self.graph.get_estimated_state():
    	# 	print(i)
    	   # print('\n')
    	# print('\n After optimization:')
    	self.graph.solve(mrob.GN)
    	self.ci2.append(self.graph.chi2())
    	# for i in self.graph.get_estimated_state():
    	# 	print(i)
    		# print('\n')

        	