import matplotlib.pyplot as plt

from scipy import io
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import path


def sample_from_line(pt1, pt2, resolution):
    N = np.ceil(np.linalg.norm((pt1 - pt2)) / resolution)
    N = max(N, 1)
    samples = np.zeros(shape=[int(N) + 1, len(pt2)])
    for i in range(int(N) + 1):
        a = i / N
        pt_new = a * pt2 + (1 - a) * pt1
        samples[i] = pt_new
    return samples


def generate_uav_trajectory_from_points(q_poses=[], sampling_resolution=10):
    num_q = q_poses.shape[0]

    sampled_q_poses = None
    for i in range(num_q - 1):
        if sampled_q_poses is None:
            sampled_q_poses = sample_from_line(q_poses[i], q_poses[i + 1], sampling_resolution)
        else:
            new_samples = sample_from_line(q_poses[i], q_poses[i + 1], sampling_resolution)
            new_samples = new_samples[1:]
            sampled_q_poses = np.r_[sampled_q_poses, new_samples.copy()]
    return sampled_q_poses


def inpolygon(xq, yq, xv, yv):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)


def separate_los_nlos_measurements(total_measurements):
    los_rssi = total_measurements[total_measurements[:, -2] == 1]
    nlos_rssi = total_measurements[total_measurements[:, -2] == 0]
    return los_rssi, nlos_rssi


def load_trj_from_mat_file(file_name):
    data = io.loadmat(file_name)
    data = data['RandTrj_set']
    return data


'''load a trajectory from trajectory set, if trj_id is None, it will return a random trj from set'''


def get_trj_from_random_trj_set(trj_set, trj_id=None):
    if trj_id is None:
        trj_id = int(np.random.randint(0, len(trj_set), 1))

    trj = np.array(trj_set[trj_id][0])
    return np.transpose(trj)


'''
Compute the local los probability pr(los) = 1/(1 + a * exp( -b * atan(h/r) + c))
outputs:
    parameters [a, b, c]
    max. radius that the los prob. > 0.5
'''


def local_los_probability(city, ue_pose, uav_alt=None, cut_off=0, los_pr_thr=0.5, display=0):
    num_rnd_pos = 1000  # number of sample taken from 3D map to construct the los prob.

    ue_pose = np.reshape(ue_pose, newshape=(1, 3))
    if uav_alt is None:
        uav_alt = city.urban_config['bld_height_max'] + 10

    xy_radius = min(uav_alt / np.tan(0.18), city.urban_config['map_x_len'],
                    city.urban_config['map_y_len'])  # maximum radius for taking the samples around the user

    rnd_uav_pos_set = []
    uav_elevation_set = []
    dr = 10
    for r in np.arange(10, xy_radius, dr):
        dtheta = 20 / r
        for theta in np.arange(0, 2 * np.pi, dtheta):
            rnd_pos = np.array([[r * np.cos(theta), r * np.sin(theta), uav_alt]]) + ue_pose
            if (0 < rnd_pos[:, 0] < city.urban_config['map_x_len']) and (
                    0 < rnd_pos[:, 1] < city.urban_config['map_y_len']):
                rnd_uav_pos_set.append(rnd_pos)
                dist = np.linalg.norm((rnd_pos[0, :2] - ue_pose[0, :2]))
                elevation = np.array([np.arctan(rnd_pos[0, 2] / dist)])
                uav_elevation_set.append(elevation)

    rnd_uav_pos = np.array([rnd[0] for rnd in rnd_uav_pos_set])
    uav_elevation = np.array([elv for elv in uav_elevation_set])
    num_rnd_pos = len(rnd_uav_pos)
    pos_status = city.link_status_to_given_user(rnd_uav_pos, ue_pose)
    max_nlos_theta = 0
    min_nlos_theta = 10
    max_los_theta = 0
    for idx in range(num_rnd_pos):
        if pos_status[idx] == 0:
            max_nlos_theta = max(max_nlos_theta, uav_elevation[idx])
            min_nlos_theta = min(min_nlos_theta, uav_elevation[idx])
        if pos_status[idx] == 1:
            max_los_theta = max(max_los_theta, uav_elevation[idx])

    x_train = []
    y_train = []

    max_nlos_theta = min(max_nlos_theta, 75 * 3.14 / 180)
    for idx in range(num_rnd_pos):
        if (pos_status[idx] == 0) or (uav_elevation[idx] > max_nlos_theta * cut_off):
            x_train.append(uav_elevation[idx])
            y_train.append(pos_status[idx])

    for theta in np.linspace(0, min_nlos_theta, 150):
        x_train.append(np.array([theta]))
        y_train.append(0)

    for theta in np.linspace(max_los_theta, 89 * 3.14 / 180, 100):
        x_train.append(np.array([theta]))
        y_train.append(1)

    xdata = np.array([x[0] for x in x_train], dtype=float)
    ydata = np.array([y for y in y_train], dtype=float)

    def func(x, a, b, c):
        return 1 / (1 + a * np.exp(b * x + c))

    bound = (np.array([0, -np.inf, 0]), np.array([np.inf, 0, np.inf]))
    popt, pcov = curve_fit(func, xdata, ydata, p0=np.array([1, -20, 15]), method='trf')
    if popt is None:
        popt = np.array([1, -20, 15])
    los_dist = 50
    for theta in np.linspace(min_nlos_theta, max_nlos_theta * 1.1, 200):
        if func(theta, *popt) > los_pr_thr:
            los_dist = uav_alt / np.tan(theta)
            break

    if display:
        x = np.linspace(0, 1.5, 200)
        plt.plot(x, func(x, *popt), 'r*', )
        plt.plot(xdata, ydata, 'b*', label='data')

    return popt, los_dist

