import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

class TrueLogisticOracle():

    def __init__(self , arm_set , theta_star):
        self.theta_star = theta_star
        self.best_arm = arm_set[0]
        self.best_val = -np.inf
        self.kappa = self.calculate_kappa(arm_set)
        # self.arm_resulting_kappa(arm_set)
        for arm in arm_set:
            if sigmoid(np.dot(arm , self.theta_star)) > self.best_val:
                self.best_val = sigmoid(np.dot(arm , self.theta_star))
                self.best_arm = arm
        
    def expected_reward(self , arm):
        return sigmoid(np.dot(arm , self.theta_star))
    
    def pull(self , arm):
        reward = int(np.random.uniform(0, 1) < self.expected_reward(arm))
        expected_regret = self.best_val - self.expected_reward(arm)
        return reward , expected_regret , self.expected_reward(arm)
    
    def calculate_kappa(self , arm_set):
        min_mu_dot = np.inf
        for arm in arm_set:
            min_mu_dot = min(min_mu_dot , dsigmoid(np.dot(arm , self.theta_star)))
        return 1.0 / min_mu_dot

    def arm_resulting_kappa(self , arm_set):
        for arm in arm_set:
            mu_dot = dsigmoid(np.dot(arm , self.theta_star))
            if 1.0/mu_dot == self.kappa:
                print(arm)
                print(self.theta_star)
                break
        
    def get_kappa(self):
        return self.kappa