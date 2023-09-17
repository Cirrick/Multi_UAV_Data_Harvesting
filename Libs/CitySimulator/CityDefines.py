import numpy as np


class CityConfigStr:
    def __init__(self):
        self.map_grid_size = 5  # 3D map discretization settings
        self.map_x_len = 600    # Map length along the x-axis [m]
        self.map_y_len = 600    # Map length along the y-axis [m]
        self.ave_width = 60     # The width of the avenues (main streets) [m]
        self.st_width = 20      # The width of the streets (lanes) [m]
        self.blk_size_x = 200   # The width of each block in the city (between avenues)
        self.blk_size_y = 200  # The length of each block in the city (between avenues)
        self.blk_size_small = 100  # Average small block size (between streets)
        self.blk_size_min = 80  # Average min block size (between streets)
        self.bld_height_avg = 15
        self.bld_height_max = 50
        self.bld_height_min = 5
        self.bld_size_avg = 80
        self.bld_size_min = 50
        self.bld_dense = 0.001  # The density of the building in each block
