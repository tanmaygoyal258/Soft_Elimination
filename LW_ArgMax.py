import numpy as np

def LW_ArgMax(arm_set , estimate_theta , eta , best_arm , theta):
    return arm_set[np.random.choice(len(arm_set))]