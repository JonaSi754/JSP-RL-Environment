import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'
curr_path = os.path.dirname(os.path.abspath(__file__))  # absolute path of current file
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add current path to sys path

import random
import numpy as np

class action_v6:
    """action space v3 mapping

    action space v3 first time take symmetric actions into account,
    so that LPT can be expressed by SPT, so do LWKR and MWKR.
    Beside of this, change the meaning of -MWKR- to "most remain processing time",
    add the rule -MOPNR- which meaning "most operations remains".
    """
    def __init__(self, action_shape) -> None:
        self.score = [[] for _ in range(action_shape[0])]

    def select_job(self, Jobs, action, mask):
        self.FIFO(Jobs, 0)
        self.SPT(Jobs, 1)
        self.MOPNR(Jobs, 2)
        self.MWKR(Jobs, 3)
        self.Random(Jobs, 4)
        weighted_jobs = np.dot(action, self.score) * mask
        return Jobs[list(weighted_jobs).index(max(weighted_jobs))]

    # Fisrt In First Out
    def FIFO(self, Jobs, i):
        self.score[i] = np.linspace(len(Jobs), 1, len(Jobs))

    # Shortest Operation Process time
    def SPT(self, Jobs, i):
        importance = sorted(Jobs, key=lambda x: x.process_time[x.cur], reverse=True)
        for job in Jobs:
            self.score[i].append(importance.index(job))

    # Most Operations remaining
    def MOPNR(self, Jobs, i):
        importance = sorted(Jobs, key=lambda x: x.cur, reverse=True)
        for job in Jobs:
            self.score[i].append(importance.index(job))

    # Most processing time remaining
    def MWKR(self, Jobs, i):
        importance = sorted(Jobs, key=lambda x: x.remain_time)
        for job in Jobs:
            self.score[i].append(importance.index(job))

    # Select a random job
    def Random(self, Jobs, i):
        importance = [i for i in range(len(Jobs))]
        random.shuffle(importance)
        for j in range(len(Jobs)):
            self.score[i].append(importance[j])

