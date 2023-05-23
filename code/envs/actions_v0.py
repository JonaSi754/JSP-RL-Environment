from fvcore.common.registry import Registry
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'
curr_path = os.path.dirname(os.path.abspath(__file__))  # absolute path of current file
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add current path to sys path

import random

"""action v0

action space v0 is the first vision of action space,
In this part, Register is used to pass function to env, so that step() can make the
choice of jobs directly
"""

registry_machine = Registry('registry_machine')

# Fisrt In First Out
@registry_machine.register()
def FIFO(Jobs):
    return Jobs[0]

# Fisrt In Last Out
@registry_machine.register()
def FILO(Jobs):
    return Jobs[-1]

# Shortest Operation Process time
@registry_machine.register()
def SPT(Jobs):
    return min(Jobs, key=lambda x: x.process_time[x.cur])

# Longest Operation Process time
@registry_machine.register()
def LPT(Jobs):
    return max(Jobs, key=lambda x: x.process_time[x.cur])

# Least number of Operations Remaining
@registry_machine.register()
def LOPNR(Jobs):
    return max(Jobs, key=lambda x: x.cur)

# Most number of Operations Remaining
@registry_machine.register()
def MOPNR(Jobs):
    return min(Jobs, key=lambda x: x.cur)

# Most processing time Remaining
@registry_machine.register()
def MWKR(Jobs):
    return max(Jobs, key=lambda x: x.remain_time)

# Most processing time Remaining
@registry_machine.register()
def LWKR(Jobs):
    return min(Jobs, key=lambda x: x.remain_time)

# Select a random job
@registry_machine.register()
def Random(Jobs):
    return random.choice(Jobs)

# get Registry of four function
def getReg():
    return registry_machine
