import os
from platform import machine
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
from envs.observations_v4 import states


def make_env(instance, env_nums):
    envs = []
    for _ in range(env_nums):
        envs.append(env(instance))
    return envs


class env(gym.Env):
    """JSP v4

    In previous versions, schedule points are set at when machines release.
    However, jobs are added to buffer before their current operation has
    finished. So, it seems that schedule at the points when an operation 
    finish is a better choice.

    In v4, discrete actions are chosen to compare with v1 in order to see
    whether there is a difference.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, instance) -> None:
        super().__init__()
        self.instance = instance
        self.Jobs = read_JSP(self.instance)
        self.action_space = spaces.Discrete(8,)
        self.observation_shape = (9,)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape)


    def render(self):
        pass


    def close(self):
        pass


    def reset(self):
        for job in self.Jobs:
            job.reset()
        self.states = states(self.Jobs)
        self.done = False
        self.state = np.zeros(self.observation_shape, dtype=np.float32)
        self.timer = 0
        self.next_machines = []
        self.offload = [False for _ in range(len(self.Jobs))]
        self.findMachine()
        return self.state


    def get_action_meaning(self, action):
        action_name = ['FIFO', 'FILO', 'SPT', 'LPT', 'LOPNR', 'MOPNR', 'MWKR', 'LWKR', 'Random']
        return action_name[action]


    def check_machine(self):
        for m in self.states.machines:
            if (not m.endTime or m.endTime[-1] <= self.timer) and m.buffer:
                    self.next_machines.append(m)


    def update_jobs(self):
        for job in self.Jobs:
            if not job.endTime or self.offload[job.No]:
                continue
            elif job.done and not self.offload[job.No]:
                last_machine = self.states.machines[job.machine_arrange[job.cur - 1]]
                last_machine.remove(job)
                self.offload[job.No] = True
            elif job.endTime[-1] == self.timer and job not in self.states.machines[job.machine_arrange[job.cur]].buffer:
                last_machine = self.states.machines[job.machine_arrange[job.cur - 1]]
                this_machine = self.states.machines[job.machine_arrange[job.cur]]
                last_machine.remove(job)
                this_machine.append(job)
                
    
    def forward_insert(self, job):
        # if there is a proper blank, insert
        if job.endTime:
            job_et = job.endTime[-1]
        else:
            job_et = 0
        for i in range(len(self.next_machines[0].startTime) - 1):
            if self.next_machines[0].endTime[i] >= job_et\
                and self.next_machines[0].startTime[i + 1] - self.next_machines[0].endTime[i] >= job.process_time[job.cur]:
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
        if not job.startTime and not self.next_machines[0].startTime:
            job.startTime.append(0)
        elif not job.startTime:
            job.startTime.append(self.next_machines[0].endTime[-1])
        elif not self.next_machines[0].endTime:
            job.startTime.append(job.endTime[-1])
        else:
            job.startTime.append(max(job.endTime[-1], self.next_machines[0].endTime[-1]))
        job.endTime.append(job.startTime[-1] + job.process_time[job.cur])
        job.remain_time -= job.process_time[job.cur]
        # update machine
        self.next_machines[0].startTime.append(job.startTime[-1])
        self.next_machines[0].endTime.append(job.endTime[-1])


    def findMachine(self):
        # if there is a machine can be processed, choose this machine
        # else add the timer util there is a machine released
        while not self.next_machines and not self.done:
            self.update_jobs()
            self.check_machine()
            if not self.next_machines:
                self.timer += 1

    
    def step(self, action):
        
        # Deal with action
        # Get the job and machine that we choose
        acts = actions_v0.getReg()
        act = self.get_action_meaning(action)
        this_machine = self.next_machines[0]
        this_job = acts.get(act)(this_machine.buffer)


        # search forward for a position to insert
        # Or append on the tail
        self.forward_insert(this_job)
            
        
        # Record last makespan
        last_makespan = self.states.makespan
        
        
        # Update observation space
        self.states.update_state(this_job, this_machine.No)
        

        # Judge if this batch of task has been done
        if sum([job.done for job in self.Jobs]) == len(self.Jobs):
            self.done = True
        

        # Give a reward
        reward = last_makespan - self.states.makespan


        # get state for next machine
        # for m in self.states.machines:
        self.next_machines.remove(this_machine)
        self.findMachine()
        if not self.done:
            self.state = self.states.observe(self.next_machines[0].No)


        # Record extra information of this step
        info = {}
        
        return self.state, reward, self.done, info
