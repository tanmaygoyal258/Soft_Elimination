import numpy as np
from LinearOracle import AdditiveLinearOracle , MultiplicativeLinearOracle
from tqdm import tqdm

def LW_ArgMax(params , arm_set , estimate_theta , eta , best_arm , theta):
    '''
    returns the action which has value within alpha-multiplicative error
    of the best action
    '''

    def weigh_the_arm(arm):
        gap = np.dot(best_arm , theta) - np.dot(arm , theta)
        return np.array(arm) / (1 + eta * gap)

    T = params["horizon"]
    W = 3 * np.log(T)
    N = 36 * W * np.log(T)**2
    s = 1 - 1/(6*np.log(T))
    eps_den = (1/T) ** (7 + 12*np.log(T))
    
    if eps_den < 1e-12:
        eps_den = 1e-12
    
    eps = (1 - np.exp(-1)) / (12 * eps_den)

    z = 2**W

    arms = []
    print("Running LW-ArgMax")
    for _ in tqdm(range(int(np.ceil(N))+1)):
        new_theta_estimate = (1 + 1/W) * z * estimate_theta + z ** (1 + 1/W) * eta * theta
        a_i , _ = AdditiveLinearOracle(new_theta_estimate / np.linalg.norm(new_theta_estimate) , arm_set , eps)
        arms.append(a_i)
        
    phi_arms = [weigh_the_arm(a) for a in arms]
    values = [np.dot(a , estimate_theta) for a in phi_arms]
    max_idx = np.argsort(values)[-1]
    return arms[max_idx]
    # phi_arms = [weigh_the_arm(a) for a in arm_set]
    # best_phi_arm , _ = MultiplicativeLinearOracle(estimate_theta , phi_arms , alpha = np.exp(-3))
    # for i in range(len(phi_arms)):
    #     if phi_arms[i].all() == best_phi_arm.all():
    #         return arm_set[i]