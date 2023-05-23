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
from envs.observations_v5 import states


def make_env(instance, env_num):
    envs = []
    for _ in range(env_num):
        envs.append(env(instance))
    return envs


class env(gym.Env):
    """JSP v5

    JSP v5 use observation v5 and action v3.
    As extension of v3 and v4, v5 holds a continous action space and make decisions with less jobs
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, instance) -> str:
        super().__init__()
        self.instance = instance
        self.Jobs = read_JSP(self.instance)
        self.action_shape = (5,)
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, 0]), high=1, shape=self.action_shape)
        self.observation_shape = (9,)
        self.observation_space = spaces.Box(low=0, high=1, shape=self.observation_shape)


    def reset(self):
        for job in self.Jobs:
            job.reset()
        self.states = states(self.Jobs)
        self.done = False
        self.state = np.zeros(self.observation_shape, dtype=np.float32)
        self.next_machines = []
        self.timer = 0
        self.offload = [False for _ in range(len(self.Jobs))]
        self.fillMachines()
        return self.state

    
    def render(self, mode='human'):
        pass

    
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
        for i in range(len(self.next_machines[0].startTime) - 1):
            if self.next_machines[0].endTime[i] >= job_et\
                and self.next_machines[0].startTime[i + 1] - self.next_machines[0].endTime[i] >= job.process_time[job.cur]:
                # arrange job
                job.startTime.append(self.next_machines[0].endTime[i])
                job.endTime.append(self.next_machines[0].endTime[i] + job.process_time[job.cur])
                job.remain_time -= job.process_time[job.cur]
                # update machine
                self.next_machines[0].startTime.insert(i, job.startTime[-1])
                self.next_machines[0].endTime.insert(i, job_et)
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


    def fillMachines(self):
        # fill machines with new released machines
        # update jobs at the same time
        while not self.next_machines and not self.done:
            # print([machine.No for machine in self.next_machines])
            # print(self.timer)
            self.update_jobs()
            for m in self.states.machines:
                if (not m.endTime or m.endTime[-1] <= self.timer) and m.buffer:
                    self.next_machines.append(m)
            if not self.next_machines:
                self.timer += 1
                

    def update_jobs(self):
        for job in self.Jobs:
            if not job.endTime or self.offload[job.No]:
                continue
            elif job.done and not self.offload[job.No]:
                last_machine = self.states.machines[job.machine_arrange[job.cur - 1]]
                last_machine.remove(job)
                self.offload[job.No] = True
            elif job.endTime[-1] <= self.timer and job not in self.states.machines[job.machine_arrange[job.cur]].buffer:
                last_machine = self.states.machines[job.machine_arrange[job.cur - 1]]
                this_machine = self.states.machines[job.machine_arrange[job.cur]]
                last_machine.remove(job)
                this_machine.append(job)


    
    def step(self, action):
        
        # Deal with action
        # Get the job and machine that we choose
        selector = action_v3(self.action_shape)
        buffer = [job for job in self.next_machines[0].buffer if not job.endTime or (self.next_machines[0].endTime and job.endTime[-1] <= self.next_machines[0].endTime[-1])]
        if buffer:
            this_job = selector.select_job(buffer, action)
        else:
            this_job = selector.select_job(self.next_machines[0].buffer, action)
        

        # search forward for a position to insert
        # Or append on the tail
        self.forward_insert(this_job)
            
        
        # Record last makespan
        last_makespan = self.states.makespan
        
        
        # Update state space
        self.states.update_state(this_job, self.next_machines[0].No)
        # print([job.done for job in self.Jobs])
        # print([job.cur for job in self.Jobs])
        # if not this_job.done:
        #     print(f"this job is going to machine {this_job.machine_arrange[this_job.cur]}")
        # else:
        #     print("this job has been done")
        # print()

            
        # Give a reward
        reward = last_makespan - self.states.makespan
        # reward = -self.states.u_avg * self.states.num_machines
        
        
        # Judge if this batch of task has been done
        if sum([job.done for job in self.Jobs]) == len(self.Jobs):
            self.done = True

        # skip to the next schedule point
        self.next_machines.remove(self.next_machines[0])
        if not self.done:
            # Calculate observation for next machine
            self.fillMachines()
            self.state = self.states.observe(self.next_machines[0].No)

        # Record extra information of this step
        info = {}
        
        return self.state, reward, self.done, info
