from Libs.ChannelModel.SegmentedChannelModel import *

# City topology
urban_config = CityConfigStr()

urban_config.map_grid_size = 5  # 3D map discretization settings
urban_config.map_x_len = 1000  # Map length along the x-axis [m]
urban_config.map_y_len = 1200  # Map length along the y-axis [m]
urban_config.ave_width = 90  # The width of the avenues (main streets) [m]
urban_config.st_width = 30  # The width of the streets (lanes) [m]
urban_config.blk_size_x = 300  # The width of each block in the city (between avenues)
urban_config.blk_size_y = 300  # The length of each block in the city (between avenues)
urban_config.blk_size_small = 120  # Average small block size (between streets)
urban_config.blk_size_min = 100  # Average min block size (between streets)
urban_config.bld_height_avg = 30
urban_config.bld_height_max = 50
urban_config.bld_height_min = 5
urban_config.bld_size_avg = 100
urban_config.bld_size_min = 75
urban_config.bld_dense = 0.001  # The density of the building in each block

city = CityConfigurator(gen_new_map=False, save_map=False, urban_config=urban_config,
                        city_file_name='config/RDM_map.npy')

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

# Landmark positions
device_position = np.array([
    [[200, 700, 0]],
    [[280, 1100, 0]],
    [[360, 500, 0]],
    [[460, 400, 0]],
    [[540, 700, 0]],
    [[660, 220, 0]],
    [[600, 1100, 0]],
    [[660, 800, 0]],
    [[800, 520, 0]],
    [[860, 920, 0]],
])

uav_start_pose = np.array([
    [[280, 300, 65]],
    [[280, 300, 60]],
    [[280, 300, 55]],
])
uav_terminal_pose = np.array([
    [[660, 920, 65]],
    [[660, 920, 60]],
    [[660, 920, 55]],
])

known_device_idx = np.array([1, 4, 8])
unknown_device_idx = np.array([0, 2, 3, 5, 6, 7, 9])
data = np.array([20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000])
uav_battery_budget = np.array([80.0, 80.0, 80.0])
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