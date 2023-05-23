import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'
curr_path = os.path.dirname(os.path.abspath(__file__))  # absolute path of current file
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add current path to sys path

import random
import numpy as np

class action_v2:
    """action space v2 mapping

    action sapce v2 is designed to satisfy env v2, which change action space
    from Discrete to Box. That means agent choose action by the weighted combination
    of action, rather than a single rule.
    """
    def __init__(self, action_shape) -> None:
        self.score = [[] for _ in range(action_shape[0])]

    def select_job(self, Jobs, action):
        self.FIFO(Jobs, 0)
        self.SPT(Jobs, 1)
        self.LPT(Jobs, 2)
        self.MWKR(Jobs, 3)
        self.Random(Jobs, 4)
        weighted_jobs = np.dot(action, self.score)
        return Jobs[list(weighted_jobs).index(max(weighted_jobs))]

    # Fisrt In First Out
    def FIFO(self, Jobs, i):
        self.score[i] = np.linspace(len(Jobs), 1, len(Jobs))

    # Shortest Operation Process time
    def SPT(self, Jobs, i):
        importance = sorted(Jobs, key=lambda x: x.process_time[x.cur])
        for job in Jobs:
            self.score[i].append(len(Jobs) - importance.index(job))

    # Longest Operation Process time
    def LPT(self, Jobs, i):
        importance = sorted(Jobs, key=lambda x: x.process_time[x.cur], reverse=True)
        for job in Jobs:
            self.score[i].append(len(Jobs) - importance.index(job))

    # Most number of Operations Remaining
    def MWKR(self, Jobs, i):
        importance = sorted(Jobs, key=lambda x: x.cur)
        for job in Jobs:
            self.score[i].append(len(Jobs) - importance.index(job))

    # Select a random job
    def Random(self, Jobs, i):
        importance = [i for i in range(len(Jobs))]
        random.shuffle(importance)
        for j in range(len(Jobs)):
            self.score[i].append(len(Jobs) - importance[j])

