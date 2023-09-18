import numpy as np

from runner import Runner
import os
import torch
from Libs.Utils.Plots import *
from common.arguments import get_common_args, get_mixer_args
from Libs.Environments.DataCollection import DataCollection


def plot_device(ue_pose, fig_id, color=None, marker='o', marker_size=50):
    plt.figure(fig_id)
    plt.scatter(ue_pose[:, 0], ue_pose[:, 1], marker=marker, c=color, s=marker_size,  label='Estimated Device Position', zorder=12)


def plot_uav(uav_trj, fig_id, color=None, marker='', marker_size=5):
    plt.figure(fig_id)
    plt.scatter(uav_trj[:, 0], uav_trj[:, 1], c=color, marker=marker, s=marker_size)


ColorMap = ["brown", "orange", "green", "olive", "purple", "blue", "pink", "gray","red" , "cyan", "black"]


if __name__ == '__main__':
    args = get_common_args()
    args = get_mixer_args(args)

    if args.map == 'RBM':
        from config.RBM_define import params
    elif args.map == 'RDM':
        from config.RDM_define import params
    else:
        raise Exception("No such map!")

    env = DataCollection(args,
                         params=params)

    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    device = torch.device(args.device)
    runner = Runner(env, args)
    result_dir = args.result_dir + '/' + args.alg + '/' + args.map + args.tag
    model_dir = args.model_dir + '/' + args.alg + '/' + args.map + args.tag

    # load the model, you can also choose the best model to evaluate by changing final to best in the file name
    if args.alg == 'qmix':
        if os.path.exists(model_dir + '/final_rnn_net_params.pkl'):
            path_rnn = model_dir + '/final_rnn_net_params.pkl'
            path_qmix = model_dir + '/final_qmix_net_params.pkl'
            map_location = device
            runner.agents.policy.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
            runner.agents.policy.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
            print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
        else:
            raise Exception("No model!")

        # set target network parameters
        runner.agents.policy.target_rnn.load_state_dict(runner.agents.policy.eval_rnn.state_dict())
        runner.agents.policy.target_qmix_net.load_state_dict(runner.agents.policy.eval_qmix_net.state_dict())

        runner.agents.policy.eval_parameters = list(runner.agents.policy.eval_qmix_net.parameters()) + list(runner.agents.policy.eval_rnn.parameters())
    
    if args.alg == 'iql':
        if os.path.exists(model_dir + '/final_rnn_net_params.pkl'):
            path_rnn = model_dir + '/final_rnn_net_params.pkl'
            map_location = device
            runner.agents.policy.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
            print('Successfully load the model: {}'.format(path_rnn))
        else:
            raise Exception("No model!")

        # set target network parameters
        runner.agents.policy.target_rnn.load_state_dict(runner.agents.policy.eval_rnn.state_dict())
        runner.agents.policy.eval_parameters = list(runner.agents.policy.eval_rnn.parameters())

    total_collected_data, episode_rewards, uav_trjs, device_idx, action_record = runner.evaluate()
    print('Total collected data: {}'.format(total_collected_data / np.sum(params['data'])))
    device_color_list = [[] for _ in range(args.n_agents)]

    for i in range(args.n_agents):
        uav_trjs[i] = np.array([pose for pose in uav_trjs[i]])
        action_record[i] = np.array([action for action in action_record[i]])
        device_idx[i] = np.array([idx for idx in device_idx[i]])
        for j in range(len(uav_trjs[i])):
            device_color_list[i].append(ColorMap[int(device_idx[i][j])])
        device_color_list[i] = np.array([color for color in device_color_list[i]])

    ax = plot_city_top_view(params['city'], 10, figsize=(8,6), fontsize=12)
    device_position = params['device_position'].copy()
    unknown_device_position = device_position[params['unknown_device_idx']]
    unknown_device_position_colors = np.array(params['color'])[params['unknown_device_idx']]
    anchor_nodes = device_position[params['known_device_idx']]
    anchor_nodes_colors = np.array(params['color'])[params['known_device_idx']]

    device_position = np.resize(device_position, (len(device_position), 3))
    unknown_device_position = np.resize(unknown_device_position, (len(unknown_device_position), 3))
    anchor_nodes = np.resize(anchor_nodes, (len(anchor_nodes), 3))

    uav_start_pose = np.resize(params['start_pose'], (len(params['start_pose']), 3))
    uav_terminal_pose = np.resize(params['end_pose'], (len(params['end_pose']), 3))
    plt.scatter(unknown_device_position[:, 0], unknown_device_position[:, 1], marker='*', s=100,
                c=unknown_device_position_colors, label='Unknown Device Position')
    plt.scatter(anchor_nodes[:, 0], anchor_nodes[:, 1], marker='^', s=100, c=anchor_nodes_colors, label='Anchor Device Position')

    if args.model:
        path_estimation = result_dir + '/est_device_pos.npy'
        est_device_pos = np.load(path_estimation)
        est_device_pos = np.array([pos.flatten() for pos in est_device_pos[-1]])
        plot_device(est_device_pos[params['unknown_device_idx']], 10, marker='+', marker_size=50, color='r')
        
    plt.scatter(uav_start_pose[:, 0], uav_start_pose[:, 1], marker='H', s=150, c='lightgray',
                label='UAV Start Zone')
    plt.scatter(uav_terminal_pose[:, 0], uav_terminal_pose[:, 1], marker='H', s=150, c='lightblue',
                label='UAV Terminal Zone')

    markers = ['o', 's', 'D']

    # plot the trajectories of the UAVs
    for i in range(args.n_agents):
        plt.scatter(uav_trjs[i][:, 0], uav_trjs[i][:, 1], marker=markers[i], s=7, c=device_color_list[i], zorder=10)

    # plot the label in the legend
    for i in range(args.n_agents):
        plt.scatter(uav_trjs[i][0, 0], uav_trjs[i][0, 1], marker=markers[i], s=20, c='b',
                    label="UAV{}'s Trajectory".format(i+1), zorder=-1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
              fancybox=True, shadow=True, ncol=3, fontsize=11, columnspacing=0.2, handlelength=1, handletextpad=0.2)

    # save the figure
    plt.savefig(result_dir + '/trj.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(result_dir + '/trj.png', format='png',bbox_inches='tight', pad_inches=0.1, dpi=1200)



