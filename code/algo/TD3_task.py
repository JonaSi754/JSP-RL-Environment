import argparse
from copy import deepcopy
from datetime import datetime
from json import load
import os
import sys
from loguru import logger

os.environ['KMP_DUPLICATE_LIB_OK']='True'
curr_path = os.path.dirname(os.path.abspath(__file__))  # absolute path of current file
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add parent path to sys path

import numpy as np
import gym
from envs.JSP_v5 import env, make_env
from stable_baselines3.td3.td3 import TD3
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from common.utils import plot_rewards, plot_makespans, make_dir
import matplotlib.pyplot as plt




#---------------------------establish env and config paramaters---------------------------#
instance = 'ta21'
env = env(instance)
obs = env.reset()

def get_args(env):
    """ Hyperparameters
    """
    curr_time = datetime.now().strftime("%Y%m%d-%H%M")  # Obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='TD3',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='JSP-v6',type=str,help="name of environment")
    parser.add_argument('--n_envs',default=10,type=int,help="numbers of environments")
    parser.add_argument('--n_sample', default=200,type=int,help="num of sample points")
    parser.add_argument('--n_epochs',default=5,type=int,help="max steps of training")
    parser.add_argument('--n_steps',default=env.states.num_jobs * env.states.num_machines,type=int,help="num of steps to complete a env")
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
    parser.add_argument('--lr',default=1e-3,type=float,help="learning rate")
    parser.add_argument('--hidden_dim',default=64,type=int)
    parser.add_argument('--device',default='cuda:0',type=str,help="cpu or cuda") 
    parser.add_argument('--model_name', default=parser.parse_args().algo_name + '-' + parser.parse_args().env_name)
    parser.add_argument('--data_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + '/' + parser.parse_args().algo_name + \
            '/' + curr_time + '/training_data/')
    parser.add_argument('--result_path',default=curr_path + "/outputs/" + parser.parse_args().env_name + '/' + parser.parse_args().algo_name + \
            '/' + curr_time + '/results/')
    parser.add_argument('--model_path',default="/home/sssjh/Workspace/MyRL/model/" + parser.parse_args().algo_name + '/', type=str) # path to save models
    parser.add_argument('--seed', default=7543, type=int, help="random seed")
    parser.add_argument('--target_kl', default=0.5, type=float, help="random seed")
    parser.add_argument('--save_fig',default=True,type=bool,help="if save figure or not")
    parser.add_argument('--tensorboard_log', default="./code/test/outputs/t ensorboard/" + parser.parse_args().model_name + "/")     
    args = parser.parse_args()
    return args


cfg = get_args(env)
envs = make_env(instance, cfg.n_envs)
makespans = [[] for _ in range(cfg.n_envs)]
rewards = [[] for _ in range(cfg.n_envs)]
total_reward = 0
min_makespan = 100000
min_total_makespan = 100000

#----------------------------train and get learning curve-------------------------------#
model = TD3('MlpPolicy', env=env, learning_rate=cfg.lr, batch_size=25, device=cfg.device, buffer_size=100000, learning_starts=cfg.n_steps)
# model = PPO.load("model/PPO-JSP-v3_20_20", env=env)
for ep in range(cfg.n_sample):
    model.learn(total_timesteps=cfg.n_epochs * cfg.n_steps)
    total_makespan = 0
    for i in range(cfg.n_envs):
        obs = envs[i].reset()
        for j in range(cfg.n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = envs[i].step(action)
            total_reward += reward
        # when an epoch of some env has finished
        makespan = envs[i].states.makespan
        total_makespan += makespan
        min_makespan = min(min_makespan, makespan)
        makespans[i].append(makespan)
        rewards[i].append(total_reward)
        total_reward = 0
    # when all envs has finished
    if total_makespan < min_total_makespan:
        bestModel = deepcopy(model)
        min_total_makespan = total_makespan
    if ep % 10 == 0:
        ep_reward = np.array(rewards, dtype=np.float16).mean(axis=0)[-1]
        ep_makespan = np.array(makespans, dtype=np.float16).mean(axis=0)[-1]
        logger.info(f"current reward: {round(ep_reward)}, current makespan: {round(ep_makespan)}")
        # print(f"current reward is {ep_reward}")

make_dir(cfg.result_path, cfg.data_path)
plot_makespans(makespans, cfg, tag='train', instance=env.instance)
plot_rewards(rewards, cfg, tag='train', instance=env.instance)
print(f"Min makespan got from test is: {min_makespan}")
plt.show()

bestModel.save(cfg.model_path + cfg.model_name)
del(model)
del(bestModel)



#-----------------------------use rule and plot the gantt chart-----------------------------#
# model = DQN('MultiInputPolicy', env).learn(0)

# for i in range(cfg.n_steps):
#     action, _state = model.predict(obs, deterministic=True)
#     # actions.append(action)
#     action = 0
#     obs, reward, done, info = env.step(action)
    
# [u1, u2, cro, crj] = env.states.get_states()
# print(env.states.makespan)

# # get gantt chart plot
# for job in env.Jobs:
#     machine_arrange = [job.machine_arrange[i] for i in range(env.states.num_machines)]
#     width = [job.endTime[i] - job.startTime[i] for i in range(env.states.num_machines)]
#     left = [job.startTime[i] for i in range(env.states.num_machines)]
#     plt.barh(machine_arrange, width, left=left, height=0.8)
#     for i in range(env.states.num_machines):
#         plt.text(left[i] + width[i]/3, machine_arrange[i]-0.2, 'J%s\nT%s'%(job.No+1, i+1), color='white', size=8)
# plt.show()


# # get states plot
# plt.show()
# plt.plot(u1, label='u_avg')
# plt.plot(u2, label='u_std')
# plt.plot(cro, label='cro')
# plt.plot(crj, label = 'crj')
# plt.legend()
# plt.show()


