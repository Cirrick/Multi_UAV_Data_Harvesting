import numpy as np
from collections import deque
from pyswarms.single.global_best import GlobalBestPSO
from network.channel_net import ChannelNet
import torch
import torch.nn as nn

''' Simultaneous channel learning for device localization'''


class SLAL:
    def __init__(self, buffer_size, channel_param=None, hidden_layers=None, city=None, device='cpu'):
        self.device = torch.device(device)
        self.buffer = deque(maxlen=buffer_size)
        self.input_size = 4
        self.ch_param = channel_param

        self.city = city

        self.model = ChannelNet(hidden_layers, input_size=self.input_size, output_size=1).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.localization_buff = dict()
        self.estimated_user_pos = dict()

    def add_to_buffer(self, new_obvs):
        if len(new_obvs.shape) == 1:
            new_obvs = np.array([new_obvs])
        for obv in new_obvs:
            self.buffer.append(obv)

    def add_to_localization_buff(self, user_id, measurements):
        user_id = int(user_id)
        if not str(user_id) in self.localization_buff.keys():
            self.localization_buff[str(user_id)] = deque(maxlen=1000)
        for data in measurements:
            self.localization_buff[str(user_id)].append(data)

    def read_from_localization_buff(self, user_id):
        user_id = int(user_id)
        meas = np.array([data for data in self.localization_buff[str(user_id)]])
        return meas

    def clear_localization_buff(self):
        self.localization_buff = dict()

    def train_model(self, ep=5, bt_size=10, val_data=[]):
        data_train = np.array([self.buffer[i] for i in range(len(self.buffer))])
        x_train, y_train = prepare_measurements_for_training(data_train)
        loss_list = []

        for e in range(ep):
            idx = np.random.randint(0, len(x_train), bt_size)
            x_tr = x_train[idx]
            y_tr = y_train[idx]
            x_tr, y_tr = torch.tensor(x_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)
            x_tr, y_tr = x_tr.to(self.device), y_tr.to(self.device)
            y_pred = self.model.forward(x_tr)
            loss = nn.MSELoss()(y_pred, y_tr)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # convert loss from device to cpu
            loss = loss.cpu()
            loss_list.append(loss.detach().numpy())

        return loss_list

    def pso_cost_function(self, particles, city, measurements):
        uav_trj = measurements[:, 5:8]
        trj_len = len(uav_trj)
        rssi = np.reshape(measurements[:, 0], newshape=[trj_len, 1])
        cost_arr = np.zeros(shape=[len(particles)])
        for i_p in range(len(particles)):
            particle = particles[i_p]
            ue_pos = np.array([[particle[0], particle[1], 0]])
            link_status = city.link_status_to_given_user(uav_trj, ue_pos[0])
            rep_ue_pos = np.repeat(ue_pos, trj_len, axis=0)
            dist_arr = uav_trj - rep_ue_pos
            dist_arr = np.linalg.norm(dist_arr, axis=1)
            standard_meas_format = np.c_[rssi, dist_arr, rep_ue_pos, uav_trj, link_status, 1-link_status]
            x_test, _ = prepare_measurements_for_training(standard_meas_format)
            rssi_pred = self.model.forward(torch.tensor(x_test, dtype=torch.float32).to(self.device))
            rssi_pred = rssi_pred.cpu().detach().numpy()
            cost_arr[i_p] = np.mean((rssi - rssi_pred) ** 2)

        return cost_arr

    def user_localization(self, city, user_id, num_samples, num_particles=100, num_itr=10, user_initial_pos=None, sample_method='random'):
        collected_measurements = self.read_from_localization_buff(user_id)
        num_samples = min(num_samples, len(collected_measurements))
        if sample_method == 'random':
            sample_idx = np.random.randint(0, len(collected_measurements), num_samples)
        elif sample_method == 'random_wo_duplicates':
            idx = self.remove_duplicates()
            sample_idx = np.random.choice(idx, num_samples)
        elif sample_method == 'uniform':
            sample_idx = np.linspace(0, len(collected_measurements) - 1, num_samples, dtype=int)
        elif sample_method == 'kmeans':
            sample_idx = self.kmeans_sample_idx(num_samples)
        collected_measurements = collected_measurements[sample_idx]
        print('Localization buffer size:', collected_measurements.shape)
        x_max = np.array([city.urban_config.map_x_len, city.urban_config.map_y_len])
        x_min = np.zeros(shape=[2])
        bounds = (x_min, x_max)
        options = {'c1': 1.0, 'c2': 0.8, 'w': 0.85}
        random_particles = city.user_scattering(num_particles)
        if user_initial_pos is not None:
            random_particles[0] = user_initial_pos
        random_particles = random_particles[:, :2]
        optimizer = GlobalBestPSO(n_particles=num_particles, dimensions=len(x_max), options=options, bounds=bounds,
                                  init_pos=random_particles)
        cost, pos = optimizer.optimize(self.pso_cost_function, num_itr,
                                       city=city,
                                       measurements=collected_measurements)
        print('PSO:', cost, pos)
        ue_est_pos = np.ones(shape=[1, 3]) * 0
        ue_est_pos[0, :2] = pos
        self.estimated_user_pos[str(user_id)] = ue_est_pos

        return ue_est_pos, collected_measurements

    def get_user_rates(self, city, q_pts, user_pos):
        num_user = len(user_pos)
        # q_pts = np.reshape(q_pts, newshape=(1, 3))

        link_status = city.link_status_to_users(q_pts, user_pos)
        rep_q_pts = np.repeat(np.array([q_pts]), num_user, axis=0)
        dist_arr = rep_q_pts - user_pos
        dist_arr = np.linalg.norm(dist_arr, axis=1)
        standard_meas_format = np.zeros(shape=[num_user, 10])
        standard_meas_format[:,1] = dist_arr
        standard_meas_format[:, 2:5] = user_pos
        standard_meas_format[:, 5:8] = rep_q_pts
        standard_meas_format[:, 8] = link_status
        standard_meas_format[:, 9] = 1-link_status
        x_test, _ = prepare_measurements_for_training(standard_meas_format)
        rssi_db = self.model.forward(torch.tensor(x_test, dtype=torch.float32)).detach().numpy()
        rssi_db = rssi_db[:, 0]
        rssi = np.power(10, rssi_db / 10)
        SNR = rssi * (self.ch_param['P_tx'] / self.ch_param['noise_level'])
        rate = self.ch_param['B'] * np.log2(1 + SNR)
        return rate

    def get_user_rates_status(self, city, q_pts, user_pos):
        num_user = len(user_pos)
        # q_pts = np.reshape(q_pts, newshape=(1, 3))

        link_status = city.link_status_to_users(q_pts, user_pos)
        rep_q_pts = np.repeat(np.array([q_pts]), num_user, axis=0)
        dist_arr = rep_q_pts - user_pos
        dist_arr = np.linalg.norm(dist_arr, axis=1)
        standard_meas_format = np.zeros(shape=[num_user, 10])
        standard_meas_format[:,1] = dist_arr
        standard_meas_format[:, 2:5] = user_pos
        standard_meas_format[:, 5:8] = rep_q_pts
        standard_meas_format[:, 8] = link_status
        standard_meas_format[:, 9] = 1-link_status
        x_test, _ = prepare_measurements_for_training(standard_meas_format)
        rssi_db = self.model.forward(torch.tensor(x_test, dtype=torch.float32)).detach().numpy()
        rssi_db = rssi_db[:, 0]
        rssi = np.power(10, rssi_db / 10)
        SNR = rssi * (self.ch_param['P_tx'] / self.ch_param['noise_level'])
        rate = self.ch_param['B'] * np.log2(1 + SNR)
        return rate, link_status

    def get_perfect_user_rates_status(self, city, q_pts, user_pos):
        return self.get_user_rates_status(city, q_pts, user_pos)

    def get_user_avg_rates(self, city, q_pts, user_pos, num_samples=None):
        return self.get_user_rates(city, q_pts, user_pos)

    def get_user_avg_rates_status(self, city, q_pts, user_pos, num_samples=None):
        return self.get_user_rates_status(city, q_pts, user_pos)

    def get_user_snr_db(self, q_pt, user_pos):     
        rssi_pred = self.get_channel_gain(q_pt, user_pos)
        snr_db_pred = rssi_pred + self.ch_param.p_tx_db - self.ch_param.noise_level_db

        return snr_db_pred

    def get_user_capacity(self, q_pt, user_pos):

        snr_db = self.get_user_snr_db(q_pt, user_pos)
        snr = np.power(10, snr_db / 10)
        capacity = self.ch_param.band_width * np.log2(1 + snr)
        return capacity

    def get_channel_gain(self, q_pt, user_pos):
        q_pt = q_pt.flatten()
        user_pos = user_pos.flatten()
        dist = np.linalg.norm((q_pt - user_pos))
        theta = np.arcsin(q_pt[2] / dist)
        link_status = self.city.check_link_status(user_pos, q_pt)
        x_test = np.c_[dist, theta, link_status, 1 - link_status]
        # x_test = np.c_[dist, link_status]
        rssi_pred = self.model.forward(torch.tensor(x_test, dtype=torch.float32).to(self.device))
        rssi_pred = rssi_pred.cpu().detach().numpy()
        return rssi_pred

    def remove_duplicates(self):
        """Remove duplicate positions and get the unique indices"""
        positions = []
        # get the first key of the buffer
        unknown_user_idx = list(self.localization_buff.keys())
        measurements = self.read_from_localization_buff(user_id=unknown_user_idx[0])
        for meas in measurements:
            positions.append(meas[5:8])
        # change the positions to numpy array
        points = np.array(positions)
        # Remove duplicates and get the original indices
        unique_points, unique_indices = np.unique(points, axis=0, return_index=True)

        return unique_indices
    
    def kmeans_sample_idx(self, num_samples):
        positions = []
        # get the first key of the buffer
        unknown_user_idx = list(self.localization_buff.keys())
        measurements = self.read_from_localization_buff(user_id=unknown_user_idx[0])
        for meas in measurements:
            positions.append(meas[5:8])
        # change the positions to numpy array
        points = np.array(positions)
        # Remove duplicates and get the original indices
        unique_points, unique_indices = np.unique(points, axis=0, return_index=True)

        # Apply k-means clustering with k=50
        k = min(num_samples, len(unique_points))
        centroids, labels = k_means(unique_points, k)

        # Get the chosen points (centroids) and their indices in the unique_points array
        chosen_points = centroids
        unique_indices_chosen = []
        for i in range(k):
            unique_indices_chosen.append(np.argmin(euclidean_distance(unique_points, chosen_points[i])))

        # Get the original indices of the chosen points
        original_indices_chosen = unique_indices[unique_indices_chosen]

        return original_indices_chosen


def prepare_measurements_for_training(measurements):
    if len(measurements.shape) == 1:
        measurements = np.array([measurements])
    meas_len = len(measurements)
    x_train = np.c_[measurements[:, 1], np.arcsin(measurements[:, 7] / measurements[:, 1])]
    # x_train = np.c_[measurements[:, 1], measurements[:, -2]]
    x_train = np.c_[x_train, measurements[:, -2:]]
    y_train= np.reshape(measurements[:, 0], newshape=[meas_len, 1])

    return x_train, y_train


def select_data(data_in):
    add_data = 1
    last_status = 0
    selected_data = []
    selected_data_idx = []
    thr = len(data_in)/5
    counter = 0
    for i in range(len(data_in)):
        data = data_in[i]
        counter += 1
        if i == 0:
            add_data = 1
        if last_status != data[-2]:
            add_data = 1
            last_status = data[-2]
        if add_data:
            if i > 0:
                selected_data.append(data_in[i-1])
                selected_data_idx.append(i-1)
            selected_data.append(data)
            selected_data_idx.append(i)
            add_data = 0
            counter = 0

        if counter >= thr:
            selected_data.append(data)
            selected_data_idx.append(i)
            counter = 0
    selected_data.append(data_in[-1])
    selected_data_idx.append(i)
    data_out = np.array([data for data in selected_data])
    data_out_idx = np.array([data for data in selected_data_idx])
    return data_out, data_out_idx


def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)


def k_means(points, k, max_iterations=100, tol=1e-4):
    centroids = points[np.random.choice(points.shape[0], k, replace=False)]
    for _ in range(max_iterations):
        distances = np.array([euclidean_distance(centroids[i], points) for i in range(k)]).T
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([points[labels == i].mean(axis=0) for i in range(k)])

        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return centroids, labels