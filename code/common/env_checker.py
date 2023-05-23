import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'
curr_path = os.path.dirname(os.path.abspath(__file__))  # absolute path of current file
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add current path to sys path


# ======================================= SB3 ===========================================

# from stable_baselines3.common.env_checker import check_env
# from envs.JSP_v1 import env1
# from envs.JSP_v2 import env2
# from envs.JSP_v3 import env3

# env = env2('ta01')
# check_env(env)


# ===================================== Elegant RL ========================================

import gym
import argparse
from datetime import datetime
from stable_baselines3.common.env_checker import check_env
from envs.JSP_v5 import env

env = env('ta21')
env.reset()
def get_args(env):
    """ Hyperparameters
    """
    curr_time = datetime.now().strftime("%Y%m%d-%H%M")  # Obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='PPO',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='JSP-v3',type=str,help="name of environment")
    parser.add_argument('--n_envs',default=4,type=int,help="numbers of environments")
    parser.add_argument('--n_sample', default=100,type=int,help="num of sample points")
    parser.add_argument('--n_epochs',default=100,type=int,help="max steps of training")
    parser.add_argument('--n_steps',default=env.states.num_jobs * env.states.num_machines,type=int,help="num of steps to complete a env")
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
    parser.add_argument('--lr',default=1e-3,type=float,help="learning rate")
    parser.add_argument('--hidden_dim',default=64,type=int)
    parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda") 
    parser.add_argument('--model_name', default=parser.parse_args().algo_name + '-' + parser.parse_args().env_name)
    parser.add_argument('--data_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/training_data/')
    parser.add_argument('--result_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + \
            '/' + curr_time + '/results/')
    parser.add_argument('--model_path',default="/Users/sijinghua/WorkSpace/code/python/MyRL/model/", type=str) # path to save models
    parser.add_argument('--seed', default=7543, type=int, help="random seed")
    parser.add_argument('--target_kl', default=0.5, type=float, help="random seed")
    parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")
    parser.add_argument('--tensorboard_log', default="./code/test/outputs/tensorboard/" + parser.parse_args().model_name + "/")     
    args = parser.parse_args()
    return args

cfg = get_args(env)

check_env(env)