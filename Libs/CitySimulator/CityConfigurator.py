from ..Utils.utils import *
from numpy import save
from numpy import load
from .CityDefines import *


class CityConfigurator:
    def __init__(self, gen_new_map=False, save_map=False, urban_config=CityConfigStr(), city_file_name=None):

        GENERATE_RANDOM_MAP = gen_new_map
        SAVE_NEW_MAP = save_map
        map_ready = 0
        self.tallest_bld_height = 0

        # City topology
        self.urban_config = urban_config

        if GENERATE_RANDOM_MAP:
            self.buildings, map_x_len, map_y_len = self.generate_city_topology()
            map_ready = 1
        elif city_file_name is not None:
            self.buildings, map_x_len, map_y_len, self.grid_x, self.grid_y, self.height_grid_map = load(
                city_file_name, allow_pickle=True)
            map_ready = 1

        if map_ready:
            self.urban_config.map_x_len = map_x_len
            self.urban_config.map_y_len = map_y_len
            self.num_blds = len(self.buildings)

            if GENERATE_RANDOM_MAP:
                self.grid_x, self.grid_y, self.height_grid_map = self.gen_height_grid_map()
                if SAVE_NEW_MAP:
                    to_be_stored = (self.buildings, map_x_len, map_y_len, self.grid_x, self.grid_y, self.height_grid_map)
                    save(city_file_name, to_be_stored)
            self.tallest_bld_height = np.max(self.height_grid_map.ravel())


    ''' Generate a topology for the city by using the set of parameters (dict)
    which was initialized in the constructor function (above) '''

    def generate_city_topology(self):
        # An array containing all the buildings (to be filled later)
        city_blds = np.zeros(shape=[0, 4, 3], dtype=float)

        # Just for ease of use
        u_c = self.urban_config
        map_x_len = u_c.map_x_len
        map_y_len = u_c.map_y_len

        num_ave_v = int(np.ceil(u_c.map_x_len / (u_c.blk_size_x + u_c.ave_width)))
        shift_x = np.random.random(1) * u_c.blk_size_x
        # X-coordinate of vertical N-S avenues (left)
        ave_v = shift_x + np.r_[0:num_ave_v] * (u_c.blk_size_x + u_c.ave_width)

        num_ave_h = int(np.ceil(u_c.map_y_len / (u_c.blk_size_y + u_c.ave_width)))
        shift_y = np.random.random(1) * u_c.blk_size_y
        # Y-coordinate of horizontal W-E avenues (bottom)
        ave_h = shift_y + np.r_[0:num_ave_h] * (u_c.blk_size_y + u_c.ave_width)

        for iv in range(num_ave_v):
            if iv == 0:
                blk_w = ave_v[iv]
                cur_blk_x = 0
            else:
                blk_w = ave_v[iv] - ave_v[iv - 1] - u_c.ave_width
                cur_blk_x = ave_v[iv - 1] + u_c.ave_width

            for ih in range(num_ave_h):
                if ih == 0:
                    blk_h = ave_h[ih]
                    cur_blk_y = 0
                else:
                    blk_h = ave_h[ih] - ave_h[ih - 1] - u_c.ave_width
                    cur_blk_y = ave_h[ih - 1] + u_c.ave_width

                # Vertical and Horizontal streets
                st_v = np.zeros(shape=[int(np.ceil(u_c.blk_size_x / u_c.blk_size_small)) * 10, 1])
                st_h = np.zeros(shape=[int(np.ceil(u_c.blk_size_y / u_c.blk_size_small)) * 10, 1])

                i_st_v = int(-1)
                blk_cover_x = 0
                while (blk_w - blk_cover_x) > u_c.blk_size_min:
                    delta = max(u_c.blk_size_min, np.random.poisson(u_c.blk_size_small))
                    if (blk_w - blk_cover_x - u_c.st_width - delta) < u_c.blk_size_min:
                        delta = blk_w - blk_cover_x - u_c.st_width - u_c.blk_size_min

                    i_st_v += 1
                    st_v[i_st_v] = blk_cover_x + delta
                    blk_cover_x += delta + u_c.st_width

                N_st_v = i_st_v + 1
                st_v[N_st_v] = blk_w
                st_v = st_v[:N_st_v + 1]

                i_st_h = int(-1)
                blk_cover_y = 0
                while (blk_h - blk_cover_y) > u_c.blk_size_min:
                    delta = max(u_c.blk_size_min, np.random.poisson(u_c.blk_size_small))
                    if blk_h - blk_cover_y - u_c.st_width - delta < u_c.blk_size_min:
                        delta = blk_h - blk_cover_y - u_c.st_width - u_c.blk_size_min

                    i_st_h += 1
                    st_h[i_st_h] = blk_cover_y + delta
                    blk_cover_y += delta + u_c.st_width

                N_st_h = i_st_h + 1
                st_h[N_st_h] = blk_h
                st_h = st_h[:N_st_h + 1]

                for i_st_v in range(N_st_v + 1):
                    if i_st_v == 0:
                        cur_grid_x = cur_blk_x
                    else:
                        cur_grid_x = cur_blk_x + st_v[i_st_v - 1] + u_c.st_width
                    cur_grid_x2 = cur_blk_x + st_v[i_st_v]

                    for i_st_h in range(N_st_h + 1):
                        if i_st_h == 0:
                            cur_grid_y = cur_blk_y
                        else:
                            cur_grid_y = cur_blk_y + st_h[i_st_h - 1] + u_c.st_width
                        cur_grid_y2 = cur_blk_y + st_h[i_st_h]

                        grid_area = (cur_grid_x2 - cur_grid_x) * (cur_grid_y2 - cur_grid_y)
                        n_bld = 0
                        if grid_area > 0:
                            n_bld = int(np.random.poisson(grid_area * u_c.bld_dense))
                        for ib in range(n_bld):
                            pos_x = cur_grid_x + np.random.random(1) * (cur_grid_x2 - cur_grid_x)
                            pos_y = cur_grid_y + np.random.random(1) * (cur_grid_y2 - cur_grid_y)
                            w = max(u_c.bld_size_min, np.random.exponential(u_c.bld_size_avg))
                            h = max(u_c.bld_size_min, np.random.exponential(u_c.bld_size_avg))

                            x1 = max(cur_grid_x, pos_x - w / 2)
                            x1 = min(x1, self.urban_config.map_x_len)
                            x2 = min(cur_grid_x2, pos_x + w / 2)
                            x2 = min(x2, self.urban_config.map_x_len)

                            y1 = max(cur_grid_y, pos_y - h / 2)
                            y1 = min(y1, self.urban_config.map_y_len)
                            y2 = min(cur_grid_y2, pos_y + h / 2)
                            y2 = min(y2, self.urban_config.map_y_len)

                            if abs(x2 - x1) > 30 and abs(y2 - y1) > 30:
                                bld_h = min(np.random.exponential(u_c.bld_height_avg),
                                            abs(np.random.normal() * 3 * np.sqrt(
                                                u_c.bld_height_max - u_c.bld_height_avg)) + u_c.bld_height_avg)
                                bld_h = max(bld_h, u_c.bld_height_min)
                                bld_h = min(bld_h, u_c.bld_height_max)
                                building = np.array([[[x1, y1, bld_h],
                                                      [x2, y1, bld_h],
                                                      [x2, y2, bld_h],
                                                      [x1, y2, bld_h]]], dtype=float)
                                city_blds = np.r_[city_blds, building]
        return city_blds, map_x_len, map_y_len

    def gen_height_grid_map(self):
        u_c = self.urban_config
        grid_x, grid_y = np.meshgrid(np.arange(0, u_c.map_x_len + 0.01, self.urban_config.map_grid_size),
                                     np.arange(0, u_c.map_y_len + 0.01, self.urban_config.map_grid_size))
        height_map = np.zeros_like(grid_x)

        xq = np.zeros(shape=[height_map.shape[0] * height_map.shape[1], 1])
        yq = np.zeros(shape=[height_map.shape[0] * height_map.shape[1], 1])
        idx_arr = np.zeros(shape=[height_map.shape[0] * height_map.shape[1], 2], dtype=int)

        idx = int(0)
        for i in range(height_map.shape[0]):
            for j in range(height_map.shape[1]):
                xq[idx] = grid_x[i, j]
                yq[idx] = grid_y[i, j]
                idx_arr[idx] = np.array([i, j], dtype=int)
                idx += 1

        for bld in self.buildings:
            xv = np.r_[bld[:, 0], bld[0, 0]]
            yv = np.r_[bld[:, 1], bld[0, 1]]
            in_flag = inpolygon(xq, yq, xv, yv)
            for j in range(len(in_flag)):
                if in_flag[j]:
                    height_map[idx_arr[j, 0], idx_arr[j, 1]] = max(height_map[idx_arr[j, 0], idx_arr[j, 1]], bld[0, -1])

        return grid_x, grid_y, height_map

    def user_scattering(self, num_user, user_altitude=0):
        city_len_x = self.urban_config.map_x_len - 100
        city_len_y = self.urban_config.map_y_len - 100

        user_arr = np.ones(shape=[num_user, 3]) * user_altitude

        ue_counter = int(0)
        while True:
            rnd_points = np.random.random(size=(4 * num_user, 2))
            rnd_points[:, 0] = rnd_points[:, 0] * city_len_x + 50
            rnd_points[:, 1] = rnd_points[:, 1] * city_len_y + 50

            rnd_pts_status = self.in_building_point(rnd_points, 2)
            for i in range(len(rnd_pts_status)):
                if not rnd_pts_status[i]:
                    user_arr[ue_counter, :2] = rnd_points[i]
                    ue_counter += 1
                    if ue_counter >= num_user:
                        return user_arr

    def user_scattering_on_grid(self, num_user, grid_size):
        min_x = 50
        max_x = self.urban_config.map_x_len + 0.01
        min_y = 50
        max_y = self.urban_config.map_y_len + 0.01
        grid_x, grid_y = np.meshgrid(np.arange(0, max_x, grid_size),
                                     np.arange(0, max_y, grid_size))

        points = np.zeros(shape=[grid_x.shape[0] * grid_x.shape[1], 2])
        counter = 0
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                if (min_x <= grid_x[i, j] <= max_x) and (min_y <= grid_y[i, j] <= max_y):
                    points[counter] = np.array([grid_x[i, j], grid_y[i, j]])
                    counter += 1
        points = points[:counter]
        np.random.shuffle(points)
        user_arr = np.ones(shape=[num_user, 3]) * self.user_height
        ue_counter = int(0)
        while True:
            rnd_pts_status = self.in_building_point(points, 2)
            for i in range(len(rnd_pts_status)):
                if not rnd_pts_status[i]:
                    user_arr[ue_counter, :2] = points[i]
                    ue_counter += 1
                    if ue_counter >= num_user:
                        return user_arr

    ''' Check whether the given point is inside the buildings or not.
    Note that the x axis is aligned with the second index of the matrix and y is the first one'''

    def in_building_point(self, q_pts, conservative):
        pts_idx = q_pts / self.urban_config.map_grid_size
        pts_idx = pts_idx.astype(int)
        h_map = self.height_grid_map
        num_row = h_map.shape[0]
        num_col = h_map.shape[1]

        in_pts_status = np.zeros(shape=[len(q_pts)])
        for i in range(len(pts_idx)):
            pt_idx = np.reshape(pts_idx[i], newshape=(1, 2))
            c_pt_idx = pt_idx
            if conservative:
                c_pt_idx = np.repeat(pt_idx, 5, axis=0)
                c_pt_idx[1, 0] = min(c_pt_idx[1, 0] + conservative, num_col - 1)
                c_pt_idx[2, 0] = max(c_pt_idx[2, 0] - conservative, 0)
                c_pt_idx[3, 1] = min(c_pt_idx[3, 1] + conservative, num_row - 1)
                c_pt_idx[4, 1] = max(c_pt_idx[4, 1] - conservative, 0)
            c_pt_idx = c_pt_idx.astype(int)
            for idx in c_pt_idx:
                if h_map[idx[1], idx[0]] > 0:  # Note: x is j and y is i in the matrix form
                    in_pts_status[i] = True
                    break
        return in_pts_status

    ''' 
    Arguments:
    q_poses: a numpy array of N 3D robot's positions (Nx3) 
    user_pos: a numpy array of K 3D users' positions (Kx3) 

    Returns
    link status for all the N robot locations and K users positions (NxK)
    '''

    def link_status_to_user(self, q_poses, user_poses=None):
        num_users = user_poses.shape[0]
        num_q = q_poses.shape[0]

        # LoS: 1,  NLoS: 0
        link_status_mat = np.ones(shape=[num_q, num_users])

        for i_q in range(num_q):
            q_pose = q_poses[i_q]
            for i_ue in range(num_users):
                link_status_mat[i_q, i_ue] = self.check_link_status(user_poses[i_ue], q_pose)
        return link_status_mat

    ''' 
    Arguments:
    q_pts: a single UAV position
    
    Returns
    link status for all the users in the city
    '''

    def link_status_to_users(self, q_pts, user_pos=None):
        if len(user_pos.shape) == 1:
            user_pos = np.array([user_pos])
        num_users = len(user_pos)

        # LoS: 1,  NLoS: 0
        ue_status = np.ones(shape=[num_users])

        for i_ue in range(num_users):
            ue = user_pos[i_ue]
            ue_status[i_ue] = self.check_link_status(ue, q_pts)
        return ue_status

    ''' 
    Arguments:
    q_pts: a single or multiple UAV position(s)
    ue_pos: the position of a single user
    
    Returns
    link status for all points on the trajectory for the given user
    '''

    def link_status_to_given_user(self, q_pts, ue_pos):

        if len(q_pts.shape) == 1:
            q_pts = np.array([q_pts])
        # LoS: 1,  NLoS: 0
        trj_len = len(q_pts)
        uav_ue_status = np.ones(shape=[trj_len])

        for i_trj in range(trj_len):
            uav_ue_status[i_trj] = self.check_link_status(ue_pos, q_pts[i_trj])

        return uav_ue_status

    def occupied_grids(self, pt1, pt2):
        line_samples = sample_from_line(pt1, pt2, self.urban_config.map_grid_size)
        occ_grids = np.zeros_like(line_samples)
        for i in range(len(line_samples)):
            # Note: j is for x-axis and i is for y-axis
            occ_grids[i] = np.array([int(line_samples[i, 1] / self.urban_config.map_grid_size), int(line_samples[i, 0] / self.urban_config.map_grid_size),
                                     line_samples[i, 2]])
        return occ_grids

    def check_link_status(self, pt1, pt2):
        # ensure the dimension of the 2 nodes
        pt1 = pt1.flatten()
        pt2 = pt2.flatten()

        occ_grids = self.occupied_grids(pt1, pt2)
        max_height = max(pt1[2], pt2[2])
        start_from_ground = 0
        for idx, grid in enumerate(occ_grids):
            if grid[2] > 0:
                if grid[2] < max_height:
                    start_from_ground = 1
                if self.height_grid_map[int(grid[0]), int(grid[1])] >= grid[2]:
                    return 0
                elif (grid[2] >= self.tallest_bld_height) and (start_from_ground == 1):
                    return 1
        return 1
