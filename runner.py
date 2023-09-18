import numpy as np
import os, time
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, env, args):
        self.env = env

        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        if not args.evaluate:
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.episode_rewards = []
        self.episode_data = []
        self.training_episode_rewards = []
        self.training_episode_data = []
        self.est_de_pos_list = []
        self.channel_loss_list = []
        self.collected_measurements = []

        # save plot files
        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map + args.tag
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, model=False):
        time_steps, train_steps, episode_steps, evaluate_steps = 0, 0, 0, -1
        # episodes_steps is the number of collected episodes
        best_episode_date = 0.0
        while episode_steps < self.args.total_episodes:
            # When training without a model, to get a smooth curve in log scale, \
            # we evaluate the model every training step before the first 1000 training steps
            if not model and episode_steps != 0 and episode_steps < 1000 and (episode_steps + 1) % self.args.evaluate_cycle != 0:
                episode_data, episode_reward, _, _, _ = self.evaluate()
                self.episode_rewards.append(episode_reward)
                self.episode_data.append(episode_data)
                self.plt(num='')
            if episode_steps == 0 or (episode_steps + 1) % self.args.evaluate_cycle == 0:
                episode_data, episode_reward, _, _, _ = self.evaluate()
                print('train_steps {}'.format(train_steps))
                print('collected data is ', episode_data)
                if episode_data > best_episode_date:
                    self.agents.policy.save_model(num='',save_type='best')
                    best_episode_date = episode_data
                self.episode_rewards.append(episode_reward)
                self.episode_data.append(episode_data)
                self.plt(num='')
                evaluate_steps += 1
                # print('time for evaluation', time.time() - start_time)
                # start_time = time.time()
            if model:
                if episode_steps % self.args.model_learning_period == 0:
                    # Set the position of the devices to the default position for evaluation
                    self.env.set_default_position()
                    # Update the channel model and estimate the positions of the devices
                    est_de_pos = self.model_learning(episode_steps // self.args.model_learning_period)
                    self.env.set_device_position(est_de_pos)
                    self.est_de_pos_list.append(est_de_pos)
                    print('Estimated device position{}: '.format(episode_steps // self.args.model_learning_period), est_de_pos)
                    np.save(self.save_path + '/est_de_pos', self.est_de_pos_list)
            episodes = []
            for episode_idx in range(self.args.n_episodes):
                episode, episode_reward, steps, total_collected_data, uav_trajs, d_idx, a_record \
                    = self.rolloutWorker.generate_episode(episode_idx, model=model)
                episodes.append(episode)
                self.training_episode_rewards.append(episode_reward)
                self.training_episode_data.append(total_collected_data)
                time_steps += steps
                episode_steps += 1
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1
        np.save(self.save_path + '/training_episode_data_{}', self.training_episode_data)
        np.save(self.save_path + '/training_episode_rewards_{}', self.training_episode_rewards)
        self.agents.policy.save_model()

    def evaluate(self, model=False):
        total_collected_data = 0
        rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, _, collected_data, trjs, device_idx, action_record = self.rolloutWorker.generate_episode(epoch, evaluate=True, model=model)
            total_collected_data += collected_data
            rewards += episode_reward
        return total_collected_data / self.args.evaluate_epoch, rewards / self.args.evaluate_epoch, trjs, device_idx, action_record

    def plt(self, num=None):
        plt.figure()

        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.episode_data)), self.episode_data)
        plt.xlabel('episodes*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_data')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('episodes*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(str(num)), format='png')
        np.save(self.save_path + '/episode_data_{}'.format(str(num)), self.episode_data)
        np.save(self.save_path + '/episode_rewards_{}'.format(str(num)), self.episode_rewards)
        plt.close()

    def federated_run(self, i, train_steps, evaluate_steps, episode_steps, data_train_list, reward_train_list,
                      data_evl_list, reward_evl_list, best_episode_data, model):
        episode_start_steps = episode_steps
        while (episode_steps - episode_start_steps) < self.args.aggregation_period:
            if (episode_steps + 1) % self.args.evaluate_cycle == 0 or episode_steps == 0:
                data_evl, reward_evl, _, _, _ = self.evaluate()
                if data_evl > best_episode_data:
                    self.agents.policy.save_model(i, save_type='best')
                    best_episode_data = data_evl
                self.episode_rewards.append(reward_evl)
                self.episode_data.append(data_evl)
                self.plt(i)
                data_evl_list.append(data_evl)
                reward_evl_list.append(reward_evl)
            episodes = []
            for episode_idx in range(self.args.n_episodes):
                episode, episode_reward, steps, total_collected_data, _, _, _ = \
                    (self.rolloutWorker.generate_episode(episode_idx, model=model))
                episodes.append(episode)
                data_train_list.append(total_collected_data)
                reward_train_list.append(episode_reward)
                episode_steps += 1
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1

        return train_steps, evaluate_steps, episode_steps, best_episode_data

    def model_learning(self, k=None):
        if k < 1:
            episode, episode_reward, step, total_collected_data, uav_trjs, d_idx, a_record = self.rolloutWorker.generate_episode(random_action=True)
        else:
            episode, episode_reward, step, total_collected_data, uav_trjs, d_idx, a_record = self.rolloutWorker.generate_episode(evaluate=True)

        # concatenate the array inside the list to a single array
        trjs = np.array([np.array(uav_trj) for uav_trj in uav_trjs])
        # concatenate the array respect to the first dimension
        trjs = np.concatenate(trjs, axis=0)

        known_device_idx = self.env.params['known_device_idx']
        unknown_device_idx = self.env.params['unknown_device_idx']
        device_pos = self.env.params['device_position'].copy()
        est_device_pos = np.zeros(shape=[len(unknown_device_idx), 3])

        collected_meas = self.env.radio_ch_model.get_measurement_from_devices(city=self.env.city, uav_poses=trjs,
                                                                device_poses=device_pos, sampling_resolution=None)

        for k_d_idx in known_device_idx:
            all_ch_gains = np.array([meas[k_d_idx].ch_gain_db for meas in collected_meas])
            all_meas_dist = np.array([meas[k_d_idx].dist for meas in collected_meas])
            sampled_device_pose = np.array([meas[k_d_idx].device_pose.flatten() for meas in collected_meas])
            sampled_uav_trj = np.array([meas[k_d_idx].uav_pose for meas in collected_meas])
            sampled_los_status = np.array([meas[k_d_idx].los_status for meas in collected_meas])

            collected_meas_for_learning = np.c_[
                all_ch_gains, all_meas_dist, sampled_device_pose, sampled_uav_trj, sampled_los_status, 1 - sampled_los_status]

            self.env.learning_channel_model.add_to_buffer(collected_meas_for_learning)

        loss_list = self.env.learning_channel_model.train_model(ep=5000, bt_size=32)
        self.channel_loss_list += loss_list

        if self.args.clear_localization_buff:
            self.env.learning_channel_model.clear_localization_buff()

        for u_d_idx in unknown_device_idx:
            all_ch_gains = np.array([meas[u_d_idx].ch_gain_db for meas in collected_meas])
            all_meas_dist = np.array([meas[u_d_idx].dist for meas in collected_meas])
            sampled_device_pose = np.array([meas[u_d_idx].device_pose.flatten() for meas in collected_meas])
            sampled_uav_trj = np.array([meas[u_d_idx].uav_pose for meas in collected_meas])
            sampled_los_status = np.array([meas[u_d_idx].los_status for meas in collected_meas])

            collected_meas_for_learning = np.c_[
                all_ch_gains, all_meas_dist, sampled_device_pose, sampled_uav_trj, sampled_los_status, 1 - sampled_los_status]

            self.env.learning_channel_model.add_to_localization_buff(u_d_idx, collected_meas_for_learning)
        collected_measurements = []
        for idx, device_idx in enumerate(unknown_device_idx):
            est_device_pos[idx], collected_meas = self.env.learning_channel_model.user_localization(self.env.city, user_id=device_idx,
                                                                                   num_samples=50, num_particles=300,
                                                                                   num_itr=5,
                                                                                   user_initial_pos=est_device_pos[idx],
                                                                                   sample_method = self.args.sample_method)
            collected_measurements.append(collected_meas)
        self.collected_measurements.append(collected_measurements)
        np.save(self.save_path + '/channel_loss', self.channel_loss_list)
        np.save(self.save_path + '/collected_measurements', np.array(self.collected_measurements))
        device_pos[unknown_device_idx] = est_device_pos[:, np.newaxis, :]

        return device_pos


