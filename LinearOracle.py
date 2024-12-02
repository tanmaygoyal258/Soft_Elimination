import numpy as np

class TrueLinearOracle():

    def __init__(self , arm_set , theta_star):
        self.theta_star = theta_star
        self.best_arm = arm_set[0]
        self.best_val = -np.inf
        
        for arm in arm_set:
            if np.dot(arm , self.theta_star) > self.best_val:
                self.best_val = np.dot(arm , self.theta_star)
                self.best_arm = arm
        
    def expected_reward(self , arm):
        return np.dot(arm , self.theta_star)
    
    def pull(self , arm):
        reward = np.random.normal(self.expected_reward(arm) , 0.01)
        expected_regret = self.best_val - self.expected_reward(arm)
        return reward , expected_regret , self.expected_reward(arm)
    
def ApproximateLinearOracle(theta , arm_set , epsilon):
        
        if theta.sum() == 0:
            return arm_set[np.random.choice(len(arm_set))] , 0
        
        best_val = -np.inf
        for arm in arm_set:
            if np.dot(arm , theta) > best_val:
                best_val = max(np.dot(arm , theta) , best_val)

        potential_candidates = []
        for arm in arm_set:
            if np.dot(arm , theta) >= best_val - epsilon:
                potential_candidates.append(arm)
        approx_best_arm = potential_candidates[np.random.choice(len(potential_candidates))]
        approx_best_val = np.dot(approx_best_arm , theta)
        
        return approx_best_arm , approx_best_val