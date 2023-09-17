import matplotlib

matplotlib.interactive(True)
import matplotlib.pyplot as plt
from matplotlib import cm
# from mayavi import mlab
import numpy as np


def plot_city_top_view(city, fig_id, figsize=(8, 6), fontsize=12):
    ub_config = city.urban_config
    st = ub_config.bld_size_min / 2
    len_x = ub_config.map_x_len
    len_y = ub_config.map_y_len

    num_blds = len(city.buildings)

    max_height = np.max(city.height_grid_map.ravel())
    min_height = np.min(city.height_grid_map.ravel())
    max_height = np.ceil(max_height / 5) * 5 + 5

    city_blds = city.buildings
    bld_height = city_blds[:, 0, -1]
    bld_h_sorted_idx = np.argsort(bld_height)

    cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in bld_height]

    fig = plt.figure(fig_id, figsize=figsize)
    ax = fig.add_subplot(111)
    for bld_idx in bld_h_sorted_idx:
        vx = city_blds[bld_idx, :, 0]
        vy = city_blds[bld_idx, :, 1]
        h = city_blds[bld_idx, :, 2]
        ax.fill(vx, vy, color=rgba[bld_idx])

    cax = cm.ScalarMappable(cmap=cmap)
    cax.set_array(bld_height)
    fcb = fig.colorbar(cax)
    fcb.set_label('Building height [m]', fontsize=fontsize)

    plt.xlabel('X-axis [m]', fontsize=fontsize)
    plt.ylabel('Y-axis [m]', fontsize=fontsize)
    plt.xlim(0, ub_config.map_x_len)
    plt.ylim(0, ub_config.map_y_len)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    return ax

def plot_radio_map(radio_map, resolution=1, cb_title=None, fig_id=None):
    if fig_id is None:
        fig = plt.figure()
    else:
        fig = plt.figure(fig_id)

    c_r_map = radio_map.T
    for x_idx, x in enumerate(radio_map):
        for y_idx, y in enumerate(x):
            c_r_map[y_idx, x_idx] = y
    c = plt.pcolor(c_r_map, cmap=cm.get_cmap('jet'))
    x_locs, x_lables = plt.xticks()
    new_x_locs = x_locs
    new_x_labels = ['{:.0f}'.format(a * resolution) for a in new_x_locs]
    plt.xticks(new_x_locs, new_x_labels)

    y_locs, y_lables = plt.yticks()
    new_y_locs = y_locs
    new_y_labels = ['{:.0f}'.format(a * resolution) for a in new_y_locs]
    plt.yticks(new_y_locs, new_y_labels)

    plt.xlabel('X-axis [m]')
    plt.ylabel('Y-axis [m]')

    cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
    cax = cm.ScalarMappable(cmap=cmap)
    cax.set_array(np.reshape(c_r_map, newshape=[-1]))
    fcb = fig.colorbar(cax)
    if cb_title is not None:
        fcb.set_label(cb_title)


def plot_user_positions(ue_pose, fig_id, color='r', marker='o', marker_size=50):
    print('Plotting UE positions')
    print(fig_id)
    plt.figure(fig_id)
    plt.scatter(ue_pose[:, 0], ue_pose[:, 1], marker=marker, c=color, s=marker_size)
    # plt.show()


def plot_rssi_measurements(rssi_meas, fig_id, color='b', marker='o'):
    plt.figure(fig_id)
    plt.scatter(rssi_meas[:, 1], rssi_meas[:, 0], marker=marker, c=color)
    plt.show()


def plot_uav_trajectory(uav_trj, fig_id, line_style='-', line_width=2, color='g', marker='', marker_size=5):
    plt.figure(fig_id)
    plt.plot(uav_trj[:, 0], uav_trj[:, 1], ls=line_style, lw=line_width, c=color, marker=marker, ms=marker_size)
    plt.show()
