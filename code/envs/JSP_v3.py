import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'
curr_path = os.path.dirname(os.path.abspath(__file__))  # absolute path of current file
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add path to sys path

import numpy as np
import gym
from gym import spaces
from common.read import read_JSP
from envs.actions_v3 import action_v3
from envs.observations_v2 import states


def make_env(instance, num_envs):
    envs = []
    for _ in range(num_envs):
        envs.append(env(instance))
    return envs


class env(gym.Env):
    """JSP v3

    JSP v3 use observation v2 and action v3,
    observation space -> Discrete(13,)
    action space -> Box(5,)

    Here we get symmetric action space, double the choice we can make.
    With a bigger observation space, contains global infos, machine 
    infos and job infos.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, instance) -> str:
        super().__init__()
        self.instance = instance
        self.Jobs = read_JSP(self.instance)
        self.action_shape = (5,)
        self.action_space = spaces.Box(low=np.ones(self.action_shape, dtype=np.float32) * -1,
                                        high=np.ones(self.action_shape, dtype=np.float32))
        self.observation_shape = (9,)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape)


    def reset(self):
        for job in self.Jobs:
            job.reset()
        self.states = states(self.Jobs)
        self.done = False
        self.state = np.zeros(self.observation_shape, dtype=np.float32)
        self.next_machine = self.states.select_machine()
        return self.state

    
    def render(self, mode='human'):
        return self.states.makespan

    
    def close(self):
        pass


    def denseReward(self, job):
        start = job.startTime[-1]
        end = job.endTime[-1]
        empty = 0
        for machine in self.states.machines:
            if machine.endTime:
                et = machine.endTime[-1]
                empty += max(0, end - et)
            else:
                empty += end - start
        return (job.process_time[job.cur - 1] - empty) / self.states.makespan


    def forward_insert(self, job):
        # if there is a proper blank, insert
        if job.endTime:
            job_et = job.endTime[-1]
        else:
            job_et = 0
        for i in range(len(self.next_machine.startTime) - 1):
            if self.next_machine.endTime[i] >= job_et\
                and self.next_machine.startTime[i + 1] - self.next_machine.endTime[i] >= job.process_time[job.cur]:
                # arrange job
                job.startTime.append(self.next_machine.endTime[i])
                job.endTime.append(self.next_machine.endTime[i] + job.process_time[job.cur])
                job.remain_time -= job.process_time[job.cur]
                # update machine
                self.next_machine.startTime.insert(i, job.startTime[-1])
                self.next_machine.endTime.insert(i, job_et)
                return
        
        # else append on the tail
        # Calculate the start time and end time of this operation
        # This step should consider both the states of machine and job
        if not job.startTime and not self.next_machine.startTime:
            job.startTime.append(0)
        elif not job.startTime:
            job.startTime.append(self.next_machine.endTime[-1])
        elif not self.next_machine.endTime:
            job.startTime.append(job.endTime[-1])
        else:
            job.startTime.append(max(job.endTime[-1], self.next_machine.endTime[-1]))
        job.endTime.append(job.startTime[-1] + job.process_time[job.cur])
        job.remain_time -= job.process_time[job.cur]
        # update machine
        self.next_machine.startTime.append(job.startTime[-1])
        self.next_machine.endTime.append(job.endTime[-1])


    
    def step(self, action):
        
        # Deal with action
        # Get the job and machine that we choose
        selector = action_v3(self.action_shape)
        buffer = [job for job in self.next_machine.buffer if not job.endTime or (self.next_machine.endTime and job.endTime[-1] <= self.next_machine.endTime[-1])]
        if buffer:
            this_job = selector.select_job(buffer, action)
        else:
            this_job = selector.select_job(self.next_machine.buffer, action)
        
        
        # search forward for a position to insert
        # Or append on the tail
        self.forward_insert(this_job)
        
        
        # Judge if this batch of task has been done
        this_job.cur += 1
        if(this_job.cur == self.states.num_machines):
            this_job.done = True
        if sum([job.done for job in self.Jobs]) == len(self.Jobs):
            self.done = True
            
        
        # Record last makespan
        last_makespan = self.states.makespan
        
        # Update state space
        self.states.update_state(this_job, self.next_machine.No)
            
        # Give a reward
        reward = last_makespan - self.states.makespan
        # reward = self.denseReward(this_job)

        if not self.done:
            # Select next machine
            self.next_machine = self.states.select_machine()

            # Calculate observation for next machine
            self.state = self.states.observe(self.next_machine.No)

        # Record extra information of this step
        info = {}
        
        return self.state, reward, self.done, info
