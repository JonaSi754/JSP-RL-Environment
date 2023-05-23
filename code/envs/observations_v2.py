import os
from platform import machine
from re import A
import re
from statistics import mean
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'
curr_path = os.path.dirname(os.path.abspath(__file__))  # absolute path of current file
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add path to sys path

import numpy as np

class states:
    """obsv2

    v2 is based on v1, add some specific infos of jobs in the wait list:
    - least next op cost
    - most next op cost
    - mean next op cost
    - least remain process time
    - most remain process time
    """
    def __init__(self, Jobs) -> None:
        self.makespan = 0.0  # max completion time of jobs
        self.u_avg = 0.  # average utilization rate
        self.u_std = 0.  # standard deviation of machine utilization
        self.cro_avg = 0.  # completion rate of OPERATIONs
        self.crj_avg = 0.  # completion rate of JOBs
        self.idx = 0
        
        self.num_jobs = len(Jobs)  # num of jobs
        self.num_machines = len(Jobs[0].process_time)  # num of jobs
        self.num_ops = self.num_jobs * self.num_machines  # num of machines
        self.longest_op = max([max(j.process_time) for j in Jobs])  # operation cost most time
        self.longest_job = max([j.total_process_time for j in Jobs])  # job cost most time

        self.machines = []
        for i in range(len(Jobs[0].process_time)):
            self.machines.append(Machine(i))
        for job in Jobs:
            self.machines[job.machine_arrange[0]].append(job)

    
    def update_state(self, job, machine_No):
        # calculate new states
        self.cal_u_avg(job)
        self.cal_u_std()
        self.cal_cro_avg(job)
        self.cal_crj_avg(job)

        # update machine state (part of that has been processed in forward_insert())
        selected_machine = self.machines[machine_No]
        selected_machine.total_possessed += job.endTime[-1] - job.startTime[-1]
        selected_machine.num_completion += 1
        selected_machine.utilization_rate = selected_machine.total_possessed / self.makespan
        selected_machine.buffer.remove(job)
        if not job.done:
            self.machines[job.machine_arrange[job.cur]].append(job)


    def observe(self, machine_No):
        selected_machine = self.machines[machine_No]
        job_list = selected_machine.buffer
        next_op_list = [job.process_time[job.cur] for job in job_list]
        remain_time_list = [job.remain_time for job in job_list]
        if selected_machine.endTime:
            et = selected_machine.endTime[-1]
        else:
            et = 0

        # return an observation
        return np.array([
            # self.u_avg, self.u_std, self.cro_avg, self.crj_avg, 
            len(selected_machine.buffer) / self.num_jobs, 
            selected_machine.utilization_rate, 
            selected_machine.num_completion / self.num_jobs, 
            et, # selected_machine.endTime[-1]
            min(next_op_list) / self.longest_op,
            max(next_op_list) / self.longest_op,
            mean(next_op_list) / self.longest_op,
            min(remain_time_list) / self.longest_job,
            max(remain_time_list) / self.longest_job], 
            dtype=np.float32
        )
    

    def select_machine(self):
        # return machines never started first
        machines = [m for m in self.machines if not m.endTime and m.buffer]
        if machines:
            return machines[np.random.randint(len(machines))]

        # return ONE of earliest released machine
        min_possessed = 100000
        for m in self.machines:
            if not m.buffer:
                continue
            elif m.endTime[-1] < min_possessed:
                min_possessed = m.endTime[-1]
                machines.clear()
                machines.append(m)
            elif m.endTime[-1] == min_possessed:
                machines.append(m)
        idx = np.random.randint(len(machines))
        return machines[idx]


    def cal_u_avg(self, job):
        self.makespan = max(self.makespan, job.endTime[-1])
        self.u_avg = sum([m.total_possessed for m in self.machines]) / (self.num_jobs * self.makespan)
    

    def cal_u_std(self):
        self.u_std = sum([(m.utilization_rate - self.u_avg) ** 2 for m in self.machines]) ** 0.5
    

    def cal_cro_avg(self, job):
        self.cro_avg = self.cro_avg + 1.0 / self.num_ops
    

    def cal_crj_avg(self, job):
        if job.done:
            self.crj_avg = self.crj_avg + 1.0 / self.num_jobs


class Machine:
    def __init__(self, No):
        self.No = No
        self.buffer = []
        self.num_completion = 0
        self.utilization_rate = 0.
        self.startTime = []
        self.endTime = []
        self.total_possessed = 0.
    
    def append(self, job):
        self.buffer.append(job)
        
    def remove(self, job):
        self.buffer.remove(job)