import numpy as np

def LWS(params , arm_set , eta , best_arm , theta):
    indices = np.random.choice([i for i in range(len(arm_set))] , params["dimension"] , replace = False)
    return [arm_set[i] for i in indices]