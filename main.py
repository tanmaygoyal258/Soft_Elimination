import numpy as np
import argparse
from datetime import datetime
import json
import os
from LinearEnv import LinearBanditEnv
import matplotlib.pyplot as plt
from LogisticEnv import LogisticBanditEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type = int, default = '10000', help = 'time horizon')
    parser.add_argument('--failure_level', type = float, default = 0.05, help = 'delta')
    parser.add_argument('--dimension' , type = int , default = 5 , help = "dimension ")
    parser.add_argument('--number_arms' , type = int , default = 100 , help = 'number of arms')
    parser.add_argument('--seed', type = int, default = 123, help = 'random seed')
    parser.add_argument('--env', type = str , default = "Linear")
    parser.add_argument('--desired_norm', type = int, default = 1, help = 'desired norm for theta and arm')
    return parser.parse_args()

def create_arm_set(params):
    '''
    TODO: assumes arm_set spans the space
    '''
    arms = []
    for _ in range(params["number_arms"] - params["dimension"]):
        arm = np.array([np.random.random()*2-1 for _ in range(params["dimension"])])
        arm = arm / (np.linalg.norm(arm) / params["desired_norm"])
        arms.append(arm)

    spanning_set = np.identity(params["dimension"])
    for arm in spanning_set:
        arms.append(arm)
    # spanning_set = [[2,1,3] , [4,2,-1] , [1,1,1]]
    # for arm in spanning_set:
    #     arms.append(arm / np.linalg.norm(arm))

    

    return arms

def main():
    # read the arguments
    args = parse_args()
    params = {}
    params['horizon'] = args.horizon
    params['failure_level'] = args.failure_level
    params['dimension'] = args.dimension
    params['number_arms'] = args.number_arms
    params['seed'] = args.seed
    params["env"] = args.env
    params["desired_norm"] = args.desired_norm

    # setting the seed
    np.random.seed(params['seed'])

    # creating the arm_set and the optimal parameter
    arm_set = create_arm_set(params)
    params["arm_set"] = [arm.tolist() for arm in arm_set]
    params["theta_star"] = np.array([np.random.random()*2 - 1 for i in range(params['dimension'])])
    params["theta_star"] = params["theta_star"] / (np.linalg.norm(params["theta_star"]) / params["desired_norm"])
    params["theta_star"] = params["theta_star"].tolist()
    
    # create the instance
    if "linear" in params["env"].lower():
        env = LinearBanditEnv(params)
    else:
        env = LogisticBanditEnv(params)
        print("The Kappa for this instance is {}".format(env.get_kappa()))
        params["kappa"] = env.get_kappa()

    # Store config as a JSON file
    now = datetime.now()
    timestamp = now.strftime("%d-%m_%H-%M")
    print(timestamp)
    path = "Data_Files_{}/".format(params["env"]) + timestamp
    if not os.path.exists(path):
        os.makedirs(path)
    with open("Data_Files_{}/".format(params["env"]) + timestamp+"/config.json", "w") as outfile:
        json.dump(params, outfile) 

    env.play()
    regret , reward = env.get_arrays()
    flattened_regret = []
    flattened_reward = []
    for batch in regret:
        for r in batch:
            flattened_regret.append(r)
    for batch in reward:
        for r in batch:
            flattened_reward.append(r)
    
    np.save("Data_Files_{}/".format(params["env"]) + timestamp + "regret_array" , flattened_regret)

    cumulative_regret = np.cumsum(flattened_regret)
    cumulative_reward = np.cumsum(flattened_reward)
    
    plt.figure(figsize = (20,10))
    
    plt.subplot(1 , 2 , 1)
    plt.plot([i for i in range(params["horizon"])] , cumulative_regret)
    plt.title("Expected Regret v/s Time")
    plt.xlabel("Time")
    plt.ylabel("Expected Regret")
    plt.grid(True)

    plt.subplot(1 , 2 , 2)
    plt.plot([i for i in range(params["horizon"])] , cumulative_reward)
    plt.title("Expected Reward v/s Time")
    plt.xlabel("Time")
    plt.ylabel("Expected Reward")
    plt.grid(True)
    
    plt.savefig("Data_Files_{}/".format(params["env"]) + timestamp + "graph.png")
    

if __name__ == "__main__":
    main()
