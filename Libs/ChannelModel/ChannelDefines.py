import numpy as np


class ChannelParamStr:
    def __init__(self):
        self.los_exp = 0
        self.los_bias_db = 0
        self.los_var_db = 0
        self.nlos_exp = 0
        self.nlos_bias_db = 0
        self.nlos_var_db = 0
        self.p_tx_db = 0
        self.noise_level_db = 0
        self.band_width = 0


class RadioMeasurementStr:
    def __init__(self):
        self.ch_gain_db = 0
        self.rssi_db = 0
        self.snr_db = 0
        self.ch_capacity = 0
        self.dist = 0
        self.uav_pose = np.zeros(shape=[1, 3])
        self.device_pose = np.zeros(shape=[1, 3])
        self.los_status = 0


class RadioMapStr:
    def __init__(self, shape=None):
        if shape is None:
            shape = [1, 1, 1]
        self.los = np.zeros(shape=shape)
        self.rssi_db = self.los.copy()
        self.ch_gain_db = self.los.copy()
        self.capacity = self.los.copy()




