import numpy as np
from LinearOracle import AdditiveLinearOracle
from LogisticOracle import (
    TrueLogisticOracle,
    sigmoid,
    dsigmoid
)
from tqdm import tqdm
from LinearWeightedSpanner import LWS

class LogisticBanditEnv():

    def __init__(self , params):
        self.params = params
        self.arm_set = params["arm_set"]
        self.dimension = params["dimension"]
        self.failure_level = params["failure_level"]
        self.horizon = params["horizon"]

        self.true_oracle = TrueLogisticOracle(self.arm_set , params["theta_star"])
        print("The best arm is {}".format(self.true_oracle.best_arm))
        self.estimate_theta = np.zeros(self.dimension)
        self.estimate_best_arm = self.arm_set[np.random.choice(len(self.arm_set))]
        self.estimate_best_val = np.dot(self.estimate_best_arm , self.estimate_theta)
        
        self.ctr = 0
        self.batch_num = 1
        self.batch_ctr = 0
        self.total_batches = np.ceil(np.log(np.log(self.horizon))) + 1
        self.prev_batch_length = 1
        self.current_batch_length = self.get_batch_lengths()

        self.kappa = self.true_oracle.get_kappa()

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
            
            self.estimate_theta = np.linalg.inv(v_matrix) @ s_matrix
            self.estimate_best_arm , self.estimate_best_val = AdditiveLinearOracle(self.estimate_theta , self.arm_set , 1/self.horizon)
            
            self.batch_num += 1
            self.prev_batch_length = self.current_batch_length
            self.current_batch_length = self.get_batch_lengths()
            self.batch_ctr = 0
            print("The new estimated best arm is {}".format(self.estimate_best_arm))
        return
    
    def play_batch(self):

        eta = np.sqrt(self.prev_batch_length) / (8 * self.gamma)
        
        print("Finding Barycentric Spanner")
        barycentric_spanner = LWS(self.params , self.arm_set , eta , self.estimate_best_arm , self.estimate_theta)
        print("Barycentric spanner found")

        v_matrix = np.identity(self.dimension)
        s_matrix = np.zeros(self.dimension)

        regrets = []
        rewards = []

        for a in barycentric_spanner:
            # TODO: check if a belongs to ball of radius 1/T

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

                if self.ctr >= self.horizon:
                    return regrets , rewards , None , None

                if self.batch_ctr >= self.current_batch_length:
                    return regrets , rewards , v_matrix , s_matrix


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

                if self.ctr >= self.horizon:
                    return regrets , rewards , None , None

                if self.batch_ctr >= self.current_batch_length:
                    return regrets , rewards , v_matrix , s_matrix


    def get_arrays(self):
        return self.expected_regret , self.expected_reward

    def get_kappa(self):
        return self.kappa