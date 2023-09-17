import os
from .EnvironmentBase import *
from .IoTDevice import DeviceList


class AgentModel:
    def __init__(self,
                 start_pose=np.array([[0, 0, 0]]),
                 end_pose=np.array([[0, 0, 0]]),
                 battery_budget=50.0,
                 height=60.0
                 ):
        # hover, north, west, south, east, no-op when running out of battery
        self.action_space = np.array([[0, 0, 0],
                                      [0, 1, 0],
                                      [-1, 0, 0],
                                      [0, -1, 0],
                                      [1, 0, 0],
                                      [0, 0, 0],
                                      ])
        self.power_consumption_table = np.ones(shape=[len(self.action_space)])
        self.reward_table = np.ones(shape=[len(self.action_space)])
        self.power_consumption_table[0] *= 0.5
        self.power_consumption_table[len(self.action_space) - 1] = 0.0
        self.reward_table[len(self.action_space) - 1] = 0.0
        self.battery_budget = battery_budget
        self.battery = 0
        self.step_size = 20

        self.start_pose = start_pose
        self.end_pose = end_pose
        self.height = height
        self.current_pose = None
        self.current_pose_index = None
        self.done = False
        self.collected_device = np.array([-1])

    def agent_move(self, action):
        self.current_pose_index += self.action_space[int(action)]
        self.current_pose = self.index_to_pose(self.current_pose_index)
        self.battery -= self.power_consumption_table[int(action)]

    def pose_to_index(self, pos):
        index = pos / self.step_size
        index[:, -1] = pos[:, -1]
        return index.astype(int)

    def index_to_pose(self, index):
        pose = (index * self.step_size)
        pose[:, -1] = index[:, -1]
        return pose


class EnvInfoStr:
    def __init__(self):
        self.agent_pose = 0
        self.collected_data = 0
        self.battery = 0


class DataCollection(EnvironmentBase):
    def __init__(self,
                 args,
                 params=None,
                 learning_channel_model=None,
                 ):
        super().__init__()

        self.args = args
        self.params = params
        # if the model is not None, then the radio channel is learned
        self.learning_channel_model = learning_channel_model
        # environment information
        self.city = params['city']
        self.step_size = 20
        # the second dimension is x and the first dimension is y
        self.max_index_x = int(self.city.urban_config.map_x_len / self.step_size)
        self.max_index_y = int(self.city.urban_config.map_y_len / self.step_size)
        self.radio_ch_model = params['radio_ch_model']
        self.link_status = self.init_link_status()

        self.safety_controller_flag = True

        # Device information
        self.device_position = params['device_position'].copy()
        self.est_device_pos = self.device_position.copy()
        self.n_devices = self.device_position.shape[0]
        self.device_list = DeviceList(position=params['device_position'].copy(), color=params['color'].copy(),
                                      data=params['data'].copy(), num_device=params['num_device'])
        self.device_data = np.array([device.remaining_data for device in self.device_list.devices]).reshape(
            (len(self.device_position), 1)).copy()

        # agent information
        self.start_pose = params['start_pose'][0:args.n_agents].copy()
        self.end_pose = params['end_pose'][0:args.n_agents].copy()
        self.battery_budget = params['battery_budget'].copy()
        self.n_agents = args.n_agents
        # initialize agents
        self.agents = [AgentModel(params['start_pose'][i].copy(), params['end_pose'][i].copy(),
                                  params['battery_budget'][i]) for i in range(self.n_agents)]
        # action
        self.n_actions = 6
        self.last_action = np.zeros((self.n_agents, self.n_actions))

        # global state names of attributes
        self.ally_state_sttr_names = [
            "battery",
            "extra_battery_to_end",
            "pos_index_x",
            "pos_index_y",
            "done",
        ]
        self.device_state_attr_name = [
            "remaining_data",
            "pos_index_x",
            "pos_index_y",
        ]

        # observation dimension
        self.obs_move_feats_size = 6
        self.obs_device_feats_size = (self.n_devices, 7)
        self.obs_ally_feats_size = (self.n_agents - 1, 6)
        self.obs_own_feats_size = 2

        # sight range between UAV and device
        self.device_sight_range = 10
        # sight range between UAVs
        self.uav_sight_range = 10

        # reward setting
        self.reward_scale = 1000
        self.movement_penalty = 0
        self.reward = 0
        self.total_collected_data = 0
        self.reward_penalty = 0

        # episode setting
        self._episode_steps = 0
        self.episode_limit = int(max(params['battery_budget']) * 2)

        # whether add last_action or extra_battery_to_end to global state
        self.state_last_action = True
        self.state_extra_battery_to_end = False

        # scale of the state
        self.data_scale = max(self.device_data)
        self.dis_scale = max(self.max_index_x, self.max_index_y) * self.step_size  # distance scale
        self.battery_scale = max(params['battery_budget'])

        # the snr between UAVs and devices
        self.device_snr = np.zeros((self.n_agents, self.n_devices))
        # indicate which device is connected by which UAV
        self.device_access = np.zeros(self.n_agents)

        self.reset()

    def reset(self, model=False):
        if model:
            self.set_est_device_pos()  # reset the device positions to the estimated positions
        else:
            self.set_default_position()
        self._episode_steps = 0

        # reset agents
        for i, agent in enumerate(self.agents):
            agent.current_pose = agent.start_pose.copy()
            agent.current_pose_index = agent.pose_to_index(agent.current_pose)
            agent.collected_data = 0
            agent.battery = agent.battery_budget
            agent.done = False

        # reset devices
        for device in self.device_list.devices:
            device.remaining_data = device.data.copy()

        self.agent_info = EnvInfoStr()
        self.total_collected_data = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.device_snr = self.get_agents_device_snr(model=model)
        self.device_access = self.get_avail_devices()

        return

    def comm_step(self, collected_meas, d_id, model):
        if model:
            throughput = collected_meas
        else:
            throughput = collected_meas.ch_capacity
        collect_data = self.device_list.collect_data(throughput, d_id)
        return collect_data

    def pose_to_index(self, pos):
        index = pos / self.step_size
        index[:, -1] = pos[:, -1]
        return index.astype(int)

    def index_to_pose(self, index):
        pose = (index * self.step_size)
        pose[:, -1] = index[:, -1]
        return pose

    def check_uav_pos(self, pos_idx):
        """Check whether the position of the agent is inside the map after taking an action"""
        x, y = pos_idx[0, 0], pos_idx[0, 1]
        if 0 <= x <= self.max_index_x and 0 <= y <= self.max_index_y:
            return 1
        else:
            return 0

    def safety_controller(self, agent):
        """Make sure the agent can reach the terminal position with enough battery"""

        avail_movement_actions = [0] * self.n_actions

        end_pose_index = self.pose_to_index(agent.end_pose)
        dist_to_end = end_pose_index - agent.current_pose_index
        horizontal_steps = dist_to_end[0, 0]
        vertical_steps = dist_to_end[0, 1]
        horizontal_action_type = (horizontal_steps > 0) * 4 + (horizontal_steps < 0) * 2
        vertical_action_type = (vertical_steps > 0) * 1 + (vertical_steps < 0) * 3

        horizontal_actions = np.ones(shape=[int(abs(horizontal_steps))]) * horizontal_action_type
        vertical_actions = np.ones(shape=[int(abs(vertical_steps))]) * vertical_action_type

        action_type = [horizontal_action_type, vertical_action_type]

        required_actions_to_end = np.concatenate([horizontal_actions, vertical_actions])
        required_battery_to_end = 0
        for ac in required_actions_to_end:
            required_battery_to_end += agent.power_consumption_table[int(ac)]

        extra_battery_to_end = agent.battery - required_battery_to_end

        if extra_battery_to_end == 0:
            for i, action_index in enumerate(action_type):
                if action_index != 0:
                    avail_movement_actions[action_index] = 1
        elif extra_battery_to_end < 2:
            avail_movement_actions[0] = 1
        else:
            avail_movement_actions = [1] * (self.n_actions - 1) + [0]

        return avail_movement_actions, extra_battery_to_end

    def step(self, actions, model=False):
        """ Returns reward, terminated, info """
        self.last_action = np.eye(self.n_actions)[np.array(actions)]
        agent_collected_data = []
        movement_reward = []
        terminated = False
        info = []

        for a_id, action in enumerate(actions):
            agent = self.get_agent_by_id(a_id)
            if agent.battery > 0 and agent.done is False:
                # get the connected device's index
                device_id = self.device_access[a_id]
                # if there is a device connecting to this agent, then collect data from this device
                if device_id != -1:
                    device = self.device_list.get_device(int(device_id))
                    if model:
                        collected_meas = self.learning_channel_model.get_user_capacity(agent.current_pose,
                                                                                       device.position)
                        collected_data = self.comm_step(collected_meas, int(device_id), model)
                    else:
                        collected_meas = self.radio_ch_model.get_measurement_from_device(self.city,
                                                                                         agent.current_pose,
                                                                                         device.position)
                        collected_data = self.comm_step(collected_meas, int(device_id), model)
                    agent.collected_device = np.array([device_id])
                    agent_collected_data.append(collected_data)
                else:
                    agent.collected_device = np.array([-1])
                agent.agent_move(action)
                # if the agent runs out of the battery, then this agent is done
                if agent.battery <= 0:
                    agent.done = True
                movement_reward.append(self.movement_penalty * agent.reward_table[int(action)])
            else:
                agent.collected_device = np.array([-1])

        # calculate the reward
        total_data = np.sum(agent_collected_data)
        self.total_collected_data += total_data
        data_reward = total_data / self.reward_scale
        reward = data_reward - np.sum(movement_reward)

        # update the snr of the device and the access status of the device
        self.device_snr = self.get_agents_device_snr(model=model)
        self.device_access = self.get_avail_devices()

        # determine whether all the agents arrive the destination and are done
        n_done = 1
        for idx, agent in enumerate(self.agents):
            n_done *= agent.done
        if n_done == 1 or self._episode_steps >= self.episode_limit:
            terminated = True
        self._episode_steps += 1
        return reward, terminated, info

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for each agent:
        - agent movement features (where it can move to)
        - device features (available_to_access, remaining data, SNR_db , distance, relative_x, relative_y, whether connected)
        _ ally features (communicate, SNR_db, remaining battery, distance, relative_x, relative_y)
        - agent feature (remaining battery, extra battery to destination, relative_x to destination, relative_y to destination)
        """
        agent = self.agents[agent_id]

        move_feats_dim = self.obs_move_feats_size
        device_feats_dim = self.obs_device_feats_size
        ally_feats_dim = self.obs_ally_feats_size
        own_feats_dim = self.obs_own_feats_size

        move_feats = np.zeros(move_feats_dim, dtype=np.float32)
        device_feats = np.zeros(device_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        # if the agents still has enough battery and doesn't finish the task
        if agent.battery > 0 and agent.done is False:

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)
            for m in range(self.n_actions):
                move_feats[m] = avail_actions[m]

            # Device features
            for d_id, device in enumerate(self.device_list.devices):
                pos = self.est_device_pos[d_id].copy()
                snr_db = self.device_snr[agent_id, d_id]
                device_feats[d_id, 2] = snr_db / self.device_sight_range  # SNR_db
                device_feats[d_id, 3] = np.linalg.norm(pos - agent.current_pose) / self.dis_scale  # distance
                device_feats[d_id, 4] = (self.pose_to_index(pos)[0, 0] - agent.current_pose_index[
                    0, 0]) / self.max_index_x  # relative x
                device_feats[d_id, 5] = (self.pose_to_index(pos)[0, 1] - agent.current_pose_index[
                    0, 1]) / self.max_index_y  # relative y
                device_feats[d_id, 6] = self.device_access[agent_id] == d_id  # whether collecting data from this device
                # the signal strength is enough to let the agent get access to the device
                if snr_db >= self.device_sight_range:
                    device_feats[d_id, 0] = 1  # whether the agent can access this device
                    device_feats[d_id, 1] = device.remaining_data / self.data_scale  # remaining data

            # Ally features
            a_ids = [a_id for a_id in range(self.n_agents) if a_id != agent_id]
            for i, a_id in enumerate(a_ids):
                ally = self.get_agent_by_id(a_id)
                snr_db = self.radio_ch_model.snr_measurement(self.city, ally.current_pose, agent.current_pose, 'uav')
                ally_feats[i, 1] = snr_db / self.device_sight_range  # SNR_db
                if snr_db >= self.uav_sight_range:
                    ally_feats[i, 0] = 1  # whether communicate with this agent
                    ally_feats[i, 2] = ally.battery / self.battery_scale  # remaining battery
                    ally_feats[i, 3] = np.linalg.norm(
                        ally.current_pose - agent.current_pose) / self.dis_scale  # distance
                    ally_feats[i, 4] = (ally.current_pose_index[0, 0] - agent.current_pose_index[
                        0, 0]) / self.max_index_x  # relative x
                    ally_feats[i, 5] = (ally.current_pose_index[0, 1] - agent.current_pose_index[
                        0, 1]) / self.max_index_y  # relative y

            # Own features
            own_feats[0] = agent.battery / self.battery_scale  # remaining battery
            _, extra_battery_to_end = self.safety_controller(agent)
            own_feats[1] = extra_battery_to_end / self.battery_scale  # extra battery to destination
            # own_feats[2] = (agent.current_pose_index[0, 0] - self.pose_to_index(agent.end_pose)[0, 0]) / self.max_index_x  # relative x to destination
            # own_feats[3] = (agent.current_pose_index[0, 1] - self.pose_to_index(agent.end_pose)[0, 1]) / self.max_index_y  # relative y to destination

        agent_obs = np.concatenate(
            (
                move_feats.flatten(),
                device_feats.flatten(),
                ally_feats.flatten(),
                own_feats.flatten(),
            )
        )

        return agent_obs

    def get_state(self):
        """ Return the global state.
        Note: Can not be used when execution
        """
        state_dict = self.get_state_dict()

        state = np.append(
            state_dict["allies"].flatten(), state_dict["devices"].flatten()
        )
        if "last_action" in state_dict:
            state = np.append(state, state_dict["last_action"].flatten())

        return state

    def get_state_dict(self):
        """Returns the global state as a dictionary.

        - allies: numpy array containing agents and their attributes
        - devices: numpy array containing devices and their attributes
        - last_action: numpy array of previous actions for each agent

        NOTE: This function should not be used during decentralised execution.
        """

        # number of features in global state of each UAV and device
        nf_al = len(self.ally_state_sttr_names)
        nf_de = len(self.device_state_attr_name)

        ally_state = np.zeros((self.n_agents, nf_al))
        device_state = np.zeros((self.n_devices, nf_de))

        for al_id, agent in enumerate(self.agents):
            _, extra_battery_to_end = self.safety_controller(agent)
            ally_state[al_id, 0] = agent.battery / self.battery_scale  # remaining battery
            ally_state[al_id, 1] = extra_battery_to_end / self.battery_scale
            ally_state[al_id, 2] = agent.current_pose_index[0, 0] / self.max_index_x  # x
            ally_state[al_id, 3] = agent.current_pose_index[0, 1] / self.max_index_y  # y
            ally_state[al_id, 4] = agent.done  # whether the agent finishes the task

        for de_id, device in enumerate(self.device_list.devices):
            est_ue_pos = self.est_device_pos[de_id].copy()
            device_state[de_id, 0] = device.remaining_data / self.data_scale  # remaining data
            device_state[de_id, 1] = self.pose_to_index(est_ue_pos)[0, 0] / self.max_index_x  # x
            device_state[de_id, 2] = self.pose_to_index(est_ue_pos)[0, 1] / self.max_index_y  # y

        state = {"allies": ally_state, "devices": device_state}

        if self.state_last_action:
            state["last_action"] = self.last_action

        return state

    def get_obs_size(self):
        """ Returns the size of the observation """
        move_feats_dim = self.obs_move_feats_size
        n_devices, device_feats_dim = self.obs_device_feats_size
        n_allies, ally_feats_dim = self.obs_ally_feats_size
        own_feats_dim = self.obs_own_feats_size

        size = move_feats_dim + n_devices * device_feats_dim \
               + n_allies * ally_feats_dim + own_feats_dim

        return size

    def get_state_size(self):
        """ Return the size of the global state"""
        nf_al = len(self.ally_state_sttr_names)  # number of agents
        nf_de = len(self.device_state_attr_name)  # number of devices

        size = self.n_agents * nf_al + self.n_devices * nf_de

        if self.state_last_action:
            size += self.n_agents * self.n_actions
        if self.state_extra_battery_to_end:
            size += self.n_agents

        return size

    def get_avail_actions(self):
        """ Returns the available actions of all agents in a list """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        agent = self.get_agent_by_id(agent_id)
        avail_actions = [0] * self.n_actions

        # if the agent runs out of battery, it can only take no-op action
        if agent.done is True or agent.battery <= 0:
            avail_actions = [0] * (self.n_actions - 1) + [1]
            return avail_actions

        avail_movement_actions, _ = self.safety_controller(agent)
        for i, action in enumerate(avail_movement_actions):
            if action != 0:
                pos_index = agent.current_pose_index + agent.action_space[i]
                avail_actions[i] = self.check_uav_pos(pos_index)

        assert (sum(avail_actions) > 0), "Agent {} cannot preform action".format(agent_id)

        return avail_actions

    def get_agent_device_snr(self, agent_id, model=False):
        """ Return a numpy array of the SNR of each device for agent_id"""

        agent = self.get_agent_by_id(agent_id)
        agent_device_snr = np.zeros(self.n_devices)  # the SNR_db of each device
        if agent.battery > 0 and agent.done is False:
            for d_id, device in enumerate(self.device_list.devices):
                if not device.depleted:
                    pos = device.position
                    if model:
                        snr_db = self.learning_channel_model.get_user_snr_db(q_pt=agent.current_pose, user_pos=pos)
                    else:
                        snr_db = self.radio_ch_model.snr_measurement(self.city, pos, agent.current_pose, 'device')
                    # the signal strength is enough to let the UAV get access to the info of the device
                    agent_device_snr[d_id] = snr_db
        return agent_device_snr

    def get_agents_device_snr(self, model=False):
        """ Return a numpy array of the SNR of each device for each agent"""

        agents_device_snr = np.zeros((self.n_agents, self.n_devices))
        for agent_id in range(self.n_agents):
            agents_device_snr[agent_id, :] = self.get_agent_device_snr(agent_id, model=model)
        return agents_device_snr

    def get_avail_devices(self):
        """ Returns the index of connected devices of all agents, -1 if no device is connected,
        each agent can only connect to one device and the device can only be connected to one agent"""

        snr_array = self.device_snr.copy()
        snr_array = np.where(snr_array < self.device_sight_range, -np.inf, snr_array)
        device_access = np.zeros(self.n_agents) - 1
        for i in range(len(snr_array)):
            max_snr = np.max(snr_array)
            if max_snr < self.device_sight_range:
                break
            max_index = np.unravel_index(snr_array.argmax(), snr_array.shape)
            agent_index = max_index[0]
            device_index = max_index[1]
            device_access[agent_index] = device_index
            snr_array[agent_index, :] = -np.inf
            snr_array[:, device_index] = -np.inf
        return device_access

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_agent_by_id(self, a_id):
        """Get ally agent by ID."""
        return self.agents[a_id]

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.n_actions,
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def init_link_status(self):
        ''' Initialize the link status of the city '''
        if self.args.map == 'RBM':
            save_path = './config/RBM_link_status.npy'
        if self.args.map == 'RDM':
            save_path = './config/RDM_link_status.npy'
        if not os.path.exists(save_path):
            # generate the link status of the city to save computational time for calculating LoS or NLoS condition
            status = self.radio_ch_model.init_city_link_status(self.city, self.step_size)
            np.save(save_path, status)
        status = np.load(save_path)
        self.radio_ch_model.link_status = status

        return status

    def set_est_device_pos(self):
        ''' Update the position of the devices for model-aided learning '''
        self.device_position = self.est_device_pos.copy()
        self.device_list.update_position(self.device_position)

    def set_default_position(self):
        self.device_position = self.params['device_position'].copy()
        self.device_list.update_position(self.device_position)

    def set_device_position(self, est_ue_pos):
        self.est_device_pos = est_ue_pos.copy()
        self.set_est_device_pos()
