import os
from platform import machine
from re import A
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'
curr_path = os.path.dirname(os.path.abspath(__file__))  # absolute path of current file
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add path to sys path

import numpy as np

class states:
    """obs v1

    global states make it hard for agent to converge. So maybe use local states of
    machines will help agent know how to make decisions.

    Focused on local states, it contains:
    - length of wait list,
    - this machine's utilization, 
    - num of jobs have been processed on this machine,
    - next release time.
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

        self.machines = []
        for i in range(len(Jobs[0].process_time)):
            self.machines.append(Machine(i))
        for job in Jobs:
            self.machines[job.machine_arrange[0]].append(job)

    
    def update_state(self, job, machine_No):
        # calculate new states
        self.cal_u_avg(job)

        # update machine
        selected_machine = self.machines[machine_No]
        selected_machine.possessed_time = job.endTime[-1]
        selected_machine.total_possessed += job.endTime[-1] - job.startTime[-1]
        selected_machine.num_completion += 1
        selected_machine.utilization_rate = selected_machine.total_possessed / self.makespan
        selected_machine.buffer.remove(job)
        if not job.done:
            self.machines[job.machine_arrange[job.cur]].append(job)

        # return new states
        return np.array(
            [
                # self.u_avg, self.u_std, self.cro_avg, self.crj_avg, 
                len(selected_machine.buffer) / self.num_jobs, 
                selected_machine.utilization_rate, 
                selected_machine.num_completion / self.num_jobs, 
                selected_machine.possessed_time / self.makespan
            ], 
            dtype=np.float32
        )
    

    def select_machine(self):
        machines = []
        min_possessed = 100000
        for m in self.machines:
            if not m.buffer:
                continue
            elif m.possessed_time == min_possessed:
                machines.append(m)
            elif m.possessed_time < min_possessed:
                min_possessed = m.possessed_time
                machines.clear()
                machines.append(m)
        idx = np.random.randint(len(machines))
        return machines[idx]


    def cal_u_avg(self, job):
        self.makespan = max(self.makespan, job.endTime[-1])
        self.u_avg = sum([m.total_possessed for m in self.machines]) / (self.num_jobs * self.makespan)
    
    @DeprecationWarning
    def cal_u_std(self):
        self.u_std = sum([(m.utilization_rate - self.u_avg) ** 2 for m in self.machines]) ** 0.5
    
    @DeprecationWarning
    def cal_cro_avg(self, job):
        self.cro_avg = self.cro_avg + 1 / self.num_ops
    
    @DeprecationWarning
    def cal_crj_avg(self, job):
        if job.done:
            self.crj_avg = self.crj_avg + 1.0 / self.num_jobs
            
    @DeprecationWarning
    def get_states(self):
        return [self.u_avgs, self.u_stds, self.cro_avgs, self.crj_avgs]


class Machine:
    def __init__(self, No):
        self.No = No
        self.buffer = []
        self.num_completion = 0
        self.utilization_rate = 0.
        self.possessed_time = 0
        self.total_possessed = 0.
    
    def append(self, job):
        self.buffer.append(job)
        
    def remove(self, job):
        self.buffer.remove(job)