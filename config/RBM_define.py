import matplotlib.pyplot as plt
import numpy as np
from numpy import save
from Libs.CitySimulator.CityConfigurator import *
from Libs.ChannelModel.SegmentedChannelModel import *
# from Libs.Environments.DataCollection import *
from Libs.Environments.IoTDevice import IoTDevice

# City topology
urban_config = CityConfigStr()

urban_config.map_grid_size = 5  # 3D map discretization settings
urban_config.map_x_len = 600  # Map length along the x-axis [m]
urban_config.map_y_len = 800  # Map length along the y-axis [m]
urban_config.ave_width = 60  # The width of the avenues (main streets) [m]
urban_config.st_width = 20  # The width of the streets (lanes) [m]
urban_config.blk_size_x = 200  # The width of each block in the city (between avenues)
urban_config.blk_size_y = 200  # The length of each block in the city (between avenues)
urban_config.blk_size_small = 100  # Average small block size (between streets)
urban_config.blk_size_min = 80  # Average min block size (between streets)
urban_config.bld_height_avg = 15
urban_config.bld_height_max = 50
urban_config.bld_height_min = 5
urban_config.bld_size_avg = 80
urban_config.bld_size_min = 50
urban_config.bld_dense = 0.001  # The density of the building in each block

city = CityConfigurator(gen_new_map=False, save_map=False, urban_config=urban_config,
                        city_file_name='config/RBM_map.npy')

# Radio Channel parameters
ch_param = ChannelParamStr()
ch_param.los_exp = -2.5
ch_param.los_bias_db = -30
ch_param.los_var_db = np.sqrt(2)
ch_param.nlos_exp = -3.04
ch_param.nlos_bias_db = -35
ch_param.nlos_var_db = np.sqrt(5)
ch_param.p_tx_db = 43
ch_param.noise_level_db = -60
ch_param.band_width = 100

radio_ch_model = SegmentedChannelModel(ch_param)

uav_height = 60
bs_height = 30

ColorMap = ["brown", "orange", "green", "olive", "purple", "blue", "pink", "gray","red" , "cyan", "black"]

device_position = np.array([
    [[20, 200, 0]],
    [[60, 320, 0]],
    [[60, 760, 0]],
    [[120, 600, 0]],
    [[200, 80, 0]],
    [[320, 120, 0]],
    [[320, 760, 0]],
    [[460, 720, 0]],
    [[460, 200, 0]],
    [[580, 520, 0]]
])
known_device_idx = np.array([1, 5, 9])
unknown_device_idx = np.array([0, 2, 3, 4, 6, 7, 8])
data = np.array([16000, 16000, 16000, 16000, 16000, 16000, 16000, 16000, 16000, 16000])

uav_start_pose = np.array([
    [[320, 460, 55]],
    [[320, 460, 60]],
    [[320, 460, 65]]])
uav_terminal_pose = np.array([
    [[320, 460, 55]],
    [[320, 460, 60]],
    [[320, 460, 65]]])
uav_battery_budget = np.array([60.0, 60.0, 60.0])

colors = ColorMap[:len(data)]
devices_params = {'position': device_position, 'color': colors, 'data': data, 'device_num': len(device_position),
                  'known_device_idx': known_device_idx, 'unknown_device_idx': unknown_device_idx}
agent_params = {'start_pose': uav_start_pose, 'end_pose': uav_terminal_pose,
                'battery_budget': uav_battery_budget}

params = {'city': city, 'ch_param': ch_param, 'radio_ch_model': radio_ch_model,
          'device_position': device_position, 'color': colors,
          'data': data, 'num_device': len(device_position), 'known_device_idx': known_device_idx,
          'unknown_device_idx': unknown_device_idx, 'start_pose': uav_start_pose, 'end_pose': uav_terminal_pose,
          'battery_budget': uav_battery_budget
          }
