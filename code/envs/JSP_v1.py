import os
from platform import machine
from re import A
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'
curr_path = os.path.dirname(os.path.abspath(__file__))  # absolute path of current file
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add path to sys path

import numpy as np
import gym
from gym import spaces
from common.read import read_JSP
from envs import actions_v0
from envs.observations_v1 import states


def make_env(instance, env_nums):
    envs = []
    for _ in range(env_nums):
        envs.append(env(instance))
    return envs


class env(gym.Env):
    """env: JSP-v1

    This env uses CTDE framework to focus more on the deciding machine,
    obs-v1 provide both global and local state to the agent,
    action-v0 provide a discrete rule selection.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, instance) -> str:
        self.instance = instance
        self.Jobs = read_JSP(self.instance)
        self.action_space = spaces.Discrete(5,)
        self.observation_shape = (4,)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]), dtype=np.float32, 
                                            high=np.array([len(self.Jobs), 1, len(self.Jobs), sum([job.total_process_time for job in self.Jobs])], dtype=np.float32),
                                            shape=self.observation_shape)
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
        action_name = ['FIFO', 'SPT', 'LPT', 'MWKR', 'Random']
        return action_name[action]

    
    def step(self, action):
        
        # Deal with action
        # Get the job and machine that we choose
        acts = actions_v0.getReg()
        act = self.get_action_meaning(action)
        this_machine = self.states.select_machine()
        this_job = acts.get(act)(this_machine.buffer)
        
        
        # Calculate the start time and end time of this operation
        # This step should consider both the states of machine and job
        if len(this_job.startTime) == 0 and this_machine.possessed_time == 0:
            this_job.startTime.append(0)
        elif len(this_job.startTime) == 0:
            this_job.startTime.append(this_machine.possessed_time)
        elif this_machine.possessed_time == 0:
            this_job.startTime.append(this_job.endTime[-1])
        else:
            this_job.startTime.append(max(this_job.endTime[-1], this_machine.possessed_time))
        this_job.endTime.append(this_job.startTime[-1] + this_job.process_time[this_job.cur])
        
        
        # Judge if this batch of task has been done
        this_job.cur += 1
        if(this_job.cur == self.states.num_machines):
            this_job.done = True
        if sum([job.done for job in self.Jobs]) == len(self.Jobs):
            self.done = True
            
        
        # Record last makespan
        last_makespan = self.states.makespan
        
        
        # Update observation space
        self.state = self.states.update_state(this_job, this_machine.No)
            
        # Give a reward
        reward = last_makespan - self.states.makespan


        # Record extra information of this step
        info = {}
        
        return self.state, reward, self.done, info
