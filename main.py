import numpy as np
import argparse
from datetime import datetime
import json
import os
from LinearEnv import LinearBanditEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type = int, default = '10000', help = 'time horizon')
    parser.add_argument('--failure_level', type = float, default = 0.05, help = 'delta')
    parser.add_argument('--dimension' , type = int , default = 5 , help = "dimension ")
    parser.add_argument('--number_arms' , type = int , default = 100 , help = 'number of arms')
    parser.add_argument('--seed', type = int, default = 123, help = 'random seed')
    return parser.parse_args()

def create_arm_set(params):
    '''
    TODO: assumes arm_set spans the space
    '''
    arms = []
    for _ in range(params["number_arms"] - params["dimension"]):
        arm = np.array([np.random.random() for _ in range(params["dimension"])])
        arm = arm / np.linalg.norm(arm)
        arms.append(arm)

    spanning_set = np.identity(params["dimension"])
    for arm in spanning_set:
        arms.append(arm)

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

    # setting the seed
    np.random.seed(params['seed'])

    # creating the arm_set and the optimal parameter
    params["arm_set"] = create_arm_set(params)
    params["theta_star"] = np.array([np.random.random() for i in range(params['dimension'])])
    params["theta_star"] = params["theta_star"] / np.linalg.norm(params["theta_star"])

    #Store config as a JSON file
    now = datetime.now()
    timestamp = now.strftime("%d-%m_%H-%M")
    path = "Data_Files_Linear/"+timestamp
    if not os.path.exists(path):
        os.makedirs(path)
    with open("Data_Files_Linear/"+timestamp+"/config.json", "w") as outfile:
        json.dump(params, outfile) 

    # create the instance
    env = LinearBanditEnv(params)
