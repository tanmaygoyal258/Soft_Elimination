import numpy as np
from LinearOracle import TrueLinearOracle , ApproximateLinearOracle
from tqdm import tqdm
from LinearWeightedSpanner import LWS

class LinearBanditEnv():

    def __init__(self , params):
        self.params = params
        self.arm_set = params["arm_set"]
        self.dimension = params["dimension"]
        self.failure_level = params["failure_level"]
        self.horizon = params["horizon"]

        self.true_oracle = TrueLinearOracle(self.arm_set , params["theta_star"])
        
        self.estimate_theta = np.zeros(self.dimension)
        self.estimate_best_arm , self.estimate_best_val = ApproximateLinearOracle(self.estimate_theta , self.arm_set , 1/self.horizon)
        
        self.ctr = 1
        self.batch_num = 1
        self.batch_ctr = 1
        self.total_batches = np.ceil(np.log(np.log(self.horizon))) + 1
        self.prev_batch_length = 1
        self.current_batch_length = self.get_batch_lengths()
        

        self.barycentric_constant = np.exp(8) * self.dimension
        self.gamma = 8 * self.dimension * np.sqrt(self.barycentric_constant * (np.log(1/self.failure_level) + np.log(self.horizon)))

        self.expected_reward = []
        self.expected_regret = []

    def get_batch_lengths(self):
        if self.batch_num <= self.total_batches - 1:
            return int(max(2 * self.dimension , np.floor(self.horizon ** (1 - 2**(-self.batch_num)))))
        else:
            return self.horizon
        
    
    def play(self):

        while self.batch_num <= self.total_batches:
            print("Playing Batch {} with length {}".format(self.batch_num , self.current_batch_length))
            
            regrets , rewards , v_matrix , s_matrix = self.play_batch()
            self.expected_regret.append(regrets)
            self.expected_reward.append(rewards)
            
            if v_matrix is None and s_matrix is None:
                return
            
            self.theta_estimate = np.linalg.inv(v_matrix) @ s_matrix
            self.estimate_best_arm , self.estimate_best_val = ApproximateLinearOracle(self.estimate_theta , self.arm_set , 1/self.horizon)
            
            self.batch_num += 1
            self.prev_batch_length = self.current_batch_length
            self.current_batch_length = self.get_batch_lengths()
            self.batch_ctr = 1

        return
    
    def play_batch(self):

        eta = np.sqrt(self.prev_batch_length) / (8 * self.gamma)
        
        barycentric_spanner = LWS(self.params , self.arm_set , eta , self.estimate_best_arm , self.estimate_theta)

        v_matrix = np.identity(self.dimension)
        s_matrix = np.zeros(self.dimension)

        regrets = []
        rewards = []

        for a in barycentric_spanner:
            # TODO: check if A belongs to ball of radius 1/T

            gap = self.estimate_best_val - np.dot(a , self.estimate_theta)
            pi = 1/self.dimension
            denominator = (1 + eta * gap)**2
            number_pulls = int(np.ceil(pi * self.current_batch_length / (8 * denominator)))

            for _ in range(number_pulls):
                reward , expected_reg , expected_rew = self.true_oracle.pull(a)
                v_matrix += np.outer(a , a)
                s_matrix += reward * np.array(a)
                self.ctr += 1
                self.batch_ctr += 1
                regrets.append(expected_reg)
                rewards.append(expected_rew)

                if self.batch_ctr > self.current_batch_length:
                    return regrets , rewards , v_matrix , s_matrix

                if self.ctr > self.horizon:
                    return regrets , rewards , None , None

        if self.batch_ctr <= self.current_batch_length:
            while True:
                a = self.estimate_best_arm
                reward , expected_reg , expected_rew = self.true_oracle.pull(a)
                v_matrix += np.outer(a , a)
                s_matrix += reward * np.array(a)
                self.ctr += 1
                self.batch_ctr += 1
                regrets.append(expected_reg)
                rewards.append(expected_rew)

                if self.batch_ctr > self.current_batch_length:
                    return regrets , rewards , v_matrix , s_matrix

                if self.ctr > self.horizon:
                    return regrets , rewards , None , None

    def get_arrays(self):
        return self.expected_regret , self.expected_reward