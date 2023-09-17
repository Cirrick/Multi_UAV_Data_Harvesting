from abc import ABC, abstractmethod

import numpy as np
from ..ChannelModel.ChannelDefines import *
from ..CitySimulator.CityConfigurator import *


class EnvironmentBase(ABC):
    def __init__(self):
        self.env_param = None
        self.agent_pose = np.zeros(shape=[1, 3])
        self.agent_state = None
        self.agent_info = None

    @abstractmethod
    def comm_step(self, collected_meas, d_id, model):
        """ Returns the collected data """
        pass

    def step(self, actions):
        """ Returns reward, terminated, info """
        raise NotImplementedError

    def get_obs(self):
        """ Returns all agent observations in a list """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    def get_state(self):
        """ Return the global state """
        raise NotImplementedError

    def get_state_size(self):
        """ Return the shape of the state"""
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def reset(self):
        """ Returns initial observations and states"""
        raise NotImplementedError

    def get_env_info(self):
        """ Returns the env information"""
        raise NotImplementedError

