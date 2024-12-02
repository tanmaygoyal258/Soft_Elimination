import numpy as np
from LinearOracle import TrueLinearOracle , ApproximateLinearOracle
from tqdm import tqdm

class LinearBanditEnv():

    def __init__(self , params):
        self.arm_set = params["arm_set"]
        self.dimension = params["dimension"]
        self.failure_level = params["failure_level"]
        self.horizon = params["horizon"]

        self.true_oracle = TrueLinearOracle(self.arm_set , params["theta_star"])
        
        self.estimate_theta = np.zeros(self.dimension)
        self.estimate_best_arm , self.estimate_best_val = ApproximateLinearOracle(self.estimate_theta , self.arm_set , 1/self.horizon)
        
        self.ctr = 1
        self.batch_ctr = 1
        self.total_batches = np.ceil(np.log(np.log(self.horizon))) + 1
        self.prev_batch_length = 1
        self.current_batch_length = self.get_batch_lengths()

        self.barycentric_constant = np.exp(8) * self.dimension
        self.gamma = 8 * self.dimension * np.sqrt(self.barycentric_constant * (np.log(1/self.failure_level) + np.log(self.horizon)))

    def get_batch_lengths(self):
        if self.batch_ctr <= self.total_batches - 1:
            return np.max(2 * self.dimension , np.floor(self.horizon ** (1 - 2**(-self.batch_ctr))))
        else:
            return self.horizon
        
    
    def play(self):

        while self.batch_ctr <= self.total_batches:
            print("Playing Batch {}".format(self.batch_ctr))
            v_matrix , s_matrix = self.play_batch()
            self.theta_estimate = np.linalg.inv(v_matrix) @ s_matrix
            self.estimate_best_arm , self.estimate_best_val = ApproximateLinearOracle(self.estimate_theta , self.arm_set , 1/self.horizon)