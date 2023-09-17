import numpy as np
from Libs.ChannelModel.SegmentedChannelModel import SegmentedChannelModel


class IoTDevice:
    data: float
    collected_data: float

    def __init__(self, position=np.array([0, 0, 0]), color='blue', data=10000.0):
        self.position = position  # device's position
        self.color = color  # given a certain color in order to easier plot
        self.data = data  # device's initial data
        self.remaining_data = data  # device's remaining data

    def collect_data(self, collect):
        if collect == 0:
            return 0
        c = min(collect, self.remaining_data)
        self.remaining_data -= c

        # return collection ratio
        return c

    @property
    def depleted(self):
        return self.remaining_data <= 0


class DeviceList:

    def __init__(self, position, color, data, num_device):
        self.devices = [IoTDevice(position[k], color[k], data[k])
                        for k in range(num_device)]

    def get_best_data_rate(self, collected_meas):
        throughput = np.array([meas.ch_capacity for meas in collected_meas[0]])
        for i, device in enumerate(self.devices):
            if device.depleted:
                throughput[i] = 0
        idx = np.argmax(throughput) if throughput.any() else -1
        return throughput[idx], idx

    def collect_data(self, collect, idx):
        ratio = 0
        if idx != -1:
            ratio = self.devices[idx].collect_data(collect)
        return ratio

    def update_position(self, positions):
        for i, device in enumerate(self.devices):
            device.position = positions[i]

    def get_devices(self):
        return self.devices

    def get_device(self, idx):
        return self.devices[idx]

    def get_total_data(self):
        return sum(list([device.data for device in self.devices]))

    def get_collected_data(self):
        return sum(list([device.collected_data for device in self.devices]))

    @property
    def num_devices(self):
        return len(self.devices)

