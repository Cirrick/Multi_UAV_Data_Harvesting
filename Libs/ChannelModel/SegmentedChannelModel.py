from .RadioChannelBase import *


class SegmentedChannelModel(RadioChannelBase):
    def __init__(self, ch_param=ChannelParamStr()):
        super().__init__()
        self.ch_param = ch_param
        self.link_status = None

    def collect_measurements(self, city, uav_poses, device_poses=None):
        """collect measurements from all the devices to all the UAVs"""
        num_device = device_poses.shape[0]
        num_uav = uav_poses.shape[0]

        # check the link status between a UAV and all the devices
        device_status = city.link_status_to_user(uav_poses, device_poses)
        measurements = []

        for i_u in range(num_uav):
            uav_pose = uav_poses[i_u]
            device_meas = []
            for i_d in range(num_device):
                meas = RadioMeasurementStr()
                meas.dist = np.linalg.norm((device_poses[i_d] - uav_pose))
                meas.ch_gain_db, meas.rssi_db, meas.snr_db, meas.ch_capacity = self.channel_response(meas.dist,
                                                                                                     device_status[
                                                                                                         i_u, i_d])
                meas.los_status = device_status[i_u, i_d]
                meas.uav_pose = uav_pose
                meas.device_pose = device_poses[i_d]

                device_meas.append(meas)
            measurements.append(device_meas)

        return measurements

    def collect_measurement(self, city, uav_pose, device_pose=None):
        """collect measurements from a device to a UAV"""

        device_status = self.check_link_status(device_pose, uav_pose)

        meas = RadioMeasurementStr()
        meas.dist = np.linalg.norm((device_pose - uav_pose))
        meas.ch_gain_db, meas.rssi_db, meas.snr_db, meas.ch_capacity = self.channel_response(meas.dist,
                                                                                             device_status)
        meas.los_status = device_status
        meas.uav_pose = uav_pose
        meas.device_pose = device_pose

        return meas

    def snr_measurement(self, city, pt1, pt2, link_stype='device'):
        # check the link status between UAV and device
        if link_stype == 'device':
            link_status = self.check_link_status(pt1, pt2)

        # the link status between UAV and UAV is always 1
        elif link_stype == 'uav':
            link_status = 1

        distance = np.linalg.norm((pt1 - pt2))
        ch_gain_db, rssi_db, snr_db, ch_capacity = self.channel_response(distance, link_status)

        return snr_db

    def channel_response(self, dist, link_status):
        log_dist = 10 * np.log10(dist)
        ch_gain_los = self.ch_param.los_bias_db + self.ch_param.los_exp * log_dist + np.random.normal(
            scale=self.ch_param.los_var_db)
        ch_gain_nlos = self.ch_param.nlos_bias_db + self.ch_param.nlos_exp * log_dist + np.random.normal(
            scale=self.ch_param.nlos_var_db)

        ch_gain_db = link_status * ch_gain_los + (1 - link_status) * ch_gain_nlos

        return self.get_different_ch_output(self.ch_param, ch_gain_db)

    def get_different_ch_output(self, ch_param=ChannelParamStr(), ch_gain_db=None):
        rssi_db = ch_param.p_tx_db + ch_gain_db
        snr_db = rssi_db - ch_param.noise_level_db
        snr = np.power(10, snr_db / 10)
        capacity = ch_param.band_width * np.log2(1 + snr)
        return ch_gain_db, rssi_db, snr_db, capacity

    def init_city_link_status(self, city, step_size, uav_height=60):
        """initialize the link status between all device nodes and all possible UAV nodes"""
        d_size_x = int(city.urban_config.map_x_len) + 1
        d_size_y = int(city.urban_config.map_y_len) + 1

        # discretize the map into nodes in including the edge nodes
        size_x = int(city.urban_config.map_x_len / step_size) + 1
        size_y = int(city.urban_config.map_y_len / step_size) + 1

        status = np.zeros((d_size_x, d_size_y, size_x, size_y))

        for d_i in range(d_size_x):
            print(d_i)
            for d_j in range(d_size_y):
                for u_i in range(size_x):
                    for u_j in range(size_y):
                        pt1 = np.array([d_i, d_j, 0])
                        pt2 = np.array([u_i * step_size, u_j * step_size, uav_height])
                        status[d_i, d_j, u_i, u_j] = city.check_link_status(pt1, pt2)
        return status

    def save_city_link_status(self, status):
        pass

    def check_link_status(self, d_pos, u_pos):
        """check the link status between a device and a UAV in the link status matrix"""
        step_size = 20
        d_index = d_pos // step_size
        u_index = u_pos // step_size
        index = np.hstack((d_index.flatten()[:2], u_index.flatten()[:2]))
        index = tuple(index)
        link_status = self.link_status[index]

        return link_status



