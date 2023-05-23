import os
from re import A
import sys

from common.read import Job

os.environ['KMP_DUPLICATE_LIB_OK']='True'
curr_path = os.path.dirname(os.path.abspath(__file__))  # absolute path of current file
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add path to sys path

import numpy as np

class states:
    """obs v0

    obs v0 is the first version of obs. In this version of env, obs is designed to be
    a Discrete(4,). 
    
    Focused on global states, it contains:
    - average utilization,
    - standard deviation of machine utilization, 
    - completion of operations,
    - completion of jobs
    """
    def __init__(self, Jobs) -> None:
        self.makespan = 0  # max completion time of jobs
        self.u_avg = 0  # average utilization rate
        self.u_avgs = [0]
        self.u_std = 0  # standard deviation of machine utilization
        self.u_stds = [0]
        self.cro_avg = 0  # completion rate of OPERATIONs
        self.cro_avgs = [0]
        self.crj_avg = 0  # completion rate of JOBs
        self.crj_avgs = [0]
        self.idx = 0
        
        self.num_jobs = len(Jobs)  # num of jobs
        self.num_machines = len(Jobs[0].process_time)  # num of jobs
        self.num_ops = self.num_jobs * self.num_machines  # num of machines
        self.utilizations = [0 for i in range(len(Jobs))]  # utilized time of each job

        self.machine_buffers = []  # wait list of each machine
        for _ in range(len(Jobs[0].process_time)):
            self.machine_buffers.append([])
        for job in Jobs:
            self.machine_buffers[job.machine_arrange[job.cur]].append(job)
        
        self.machine_possesssed = [0 for _ in range(self.num_machines)]  # each machine's next available time

    
    def update_state(self, job, machine):
        # calculate new states
        self.cal_u_avg(job)
        self.cal_u_std()
        self.cal_cro_avg(job)
        self.cal_crj_avg(job)

        # record states
        self.u_avgs.append(self.u_avg)
        self.u_stds.append(self.u_std)
        self.cro_avgs.append(self.cro_avg)
        self.crj_avgs.append(self.crj_avg)

        # update machine possessed
        self.machine_possesssed[machine] = job.endTime[-1]

        # update machine buffers
        self.machine_buffers[machine].remove(job)
        if not job.done:
            self.machine_buffers[job.machine_arrange[job.cur]].append(job)

        # return new states
        return np.asarray([
            self.u_avg, 
            self.u_std, 
            self.cro_avg, 
            self.crj_avg
        ], dtype=np.float32)
    

    def select_machine_buffer(self, Jobs):
        id = min([i for i in range(self.num_machines) if self.machine_buffers[i]], key=lambda x: self.machine_possesssed[x])
        return self.machine_buffers[id], id


    def cal_u_avg(self, job):
        self.makespan = max(self.makespan, job.endTime[-1])
        self.utilizations[job.No] = (self.utilizations[job.No] + job.endTime[-1] - job.startTime[-1])
        self.u_avg = sum(self.utilizations) / (self.num_jobs * self.makespan)
    

    def cal_u_std(self):
        square_deviation = sum([(i / self.makespan - self.u_avg) ** 2 for i in self.utilizations]) / self.num_jobs
        self.u_std = square_deviation ** 0.5
    

    def cal_cro_avg(self, job):
        self.cro_avg = self.cro_avg + 1 / self.num_ops
    

    def cal_crj_avg(self, job):
        if job.done:
            self.crj_avg = self.crj_avg + 1.0 / self.num_jobs
            
    
    def get_states(self):
        return [self.u_avgs, self.u_stds, self.cro_avgs, self.crj_avgs]
    