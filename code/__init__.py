from importlib.metadata import entry_points
from gym.envs.registration import register

register(
    id = 'JSP-v0',
    entry_points = 'code.envs:JSP_v0',
)