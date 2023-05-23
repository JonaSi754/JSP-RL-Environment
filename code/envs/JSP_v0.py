import os
from platform import machine
from re import A
import sys
from turtle import pos

os.environ['KMP_DUPLICATE_LIB_OK']='True'
curr_path = os.path.dirname(os.path.abspath(__file__))  # absolute path of current file
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add path to sys path

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from common.read import read_JSP
from common.utils import plot_rewards
from envs import actions_v0
from envs.observations_v0 import states


def make_env(instance, env_nums):
    envs = []
    for _ in range(env_nums):
        envs.append(env(instance))
    return envs


class env(gym.Env):
    """env: JSP-v0

    This env takes global vision as observation
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, instance) -> str:
        self.action_space = spaces.Discrete(8,)
        self.observation_shape = (4,)
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape, dtype=np.float32),
                                            high=np.ones(self.observation_shape, dtype=np.float32))
        self.instance = instance
        self.Jobs = read_JSP(self.instance)
        self.done = False
        self.states = states(self.Jobs)
        self.state = np.zeros(self.observation_shape, dtype=np.float32)


    def reset(self):
        for job in self.Jobs:
            job.reset()
        self.states = states(self.Jobs)
        self.done = False
        self.state = np.zeros(self.observation_shape, dtype=np.float32)
        return self.state

    
    def render(self, mode='human'):
        return self.states.makespan

    
    def close(self):
        pass


    def get_action_meaning(self, action):
        action_name = ['FIFO', 'FILO', 'SPT', 'LPT', 'LOPNR', "MOPNR", 'MWKR', "LWKR", 'Random']
        return action_name[action]


    def denseReward(self, job):
        start = job.startTime[-1]
        end = job.endTime[-1]
        reward = 0
        for possessedTime in self.states.machine_possesssed:
            reward += max(0, min(end - start, end - possessedTime))
        return job.process_time[job.cur - 1] - reward

    
    def step(self, action):
        
        # Deal with action
        # Get the job and machine that we choose
        acts = actions_v0.getReg()
        act = self.get_action_meaning(action)
        machine_buffer, this_machine = self.states.select_machine_buffer(self.Jobs)


        # log: 2022.09.21
        # buffer = [job for job in machine_buffer if not job.endTime or job.endTime[-1] <= possessed]
        # if buffer:
        #     this_job = acts.get(act)(buffer)
        # else:
        this_job = acts.get(act)(machine_buffer)
        
        
        # Calculate the start time and end time of this operation
        # This step should consider both the states of machine and job
        if len(this_job.startTime) == 0 and self.states.machine_possesssed[this_machine] == 0:
            this_job.startTime.append(0)
        elif len(this_job.startTime) == 0:
            this_job.startTime.append(self.states.machine_possesssed[this_machine])
        elif self.states.machine_possesssed[this_machine] == 0:
            this_job.startTime.append(this_job.endTime[-1])
        else:
            this_job.startTime.append(max(this_job.endTime[-1], self.states.machine_possesssed[this_machine]))
        this_job.endTime.append(this_job.startTime[-1] + this_job.process_time[this_job.cur])
        this_job.remain_time -= this_job.process_time[this_job.cur]
        
        
        # Judge if this batch of task has been done
        this_job.cur += 1
        if(this_job.cur == self.states.num_machines):
            this_job.done = True
        if sum([job.done for job in self.Jobs]) == len(self.Jobs):
            self.done = True
        
        # record last makespan
        last_makespan = self.states.makespan
        
        # Update observation space
        self.state = self.states.update_state(this_job, this_machine)
            
        # Give a dense reward
        # if self.done:
        #     reward = self.states.makespan * -1.0
        # else:
        #     reward = self.denseReward(this_job) * 1.0
        # Give a reward
        reward = last_makespan - self.states.makespan


        # Record extra information of this step
        info = {}
        
        return self.state, reward, self.done, info
