from runner import Runner
from Libs.Environments.DataCollection import DataCollection
import random, os, json
from Libs.ChannelEstimator import *


def fl_train(args, params):

    n_workers = args.workers
    envs = []

    for i in range(n_workers):
        env = DataCollection(args,
                             learning_channel_model=None,
                             params=params,)
        envs.append(env)

    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    save_path = args.result_dir + '/' + args.alg + '/' + args.map + args.tag

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log_path = save_path + '/' + 'log_params.txt'
    with open(log_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    total_episodes = args.total_episodes
    aggregation_period = args.aggregation_period
    model_learning_period = args.model_learning_period

    runners = []
    data_evl_list = [[] for i in range(n_workers)]
    reward_evl_list = [[] for i in range(n_workers)]
    data_train_list = [[] for i in range(n_workers)]
    reward_train_list = [[] for i in range(n_workers)]

    epsilon_list = [0.3, 0.1, 0.05]

    for i in range(n_workers):
        runner = Runner(envs[i], args)
        if args.epsilon == -1:
            runner.rolloutWorker.epsilon = epsilon_list[i]
            runner.rolloutWorker.min_epsilon = epsilon_list[i]
        runners.append(runner)

    global_rewards, global_data = [], [],

    # Initialize the parameters of the models, set the parameters of first agent to others
    for i in range(n_workers):
        runners[i].agents.policy.eval_qmix_net.load_state_dict(runners[0].agents.policy.eval_qmix_net.state_dict())
        runners[i].agents.policy.eval_rnn.load_state_dict(runners[0].agents.policy.eval_rnn.state_dict())
        runners[i].agents.policy.target_qmix_net.load_state_dict(runners[0].agents.policy.target_qmix_net.state_dict())
        runners[i].agents.policy.target_rnn.load_state_dict(runners[0].agents.policy.target_rnn.state_dict())

    if args.model:
        ch_hidden_layers = [(50, 'tanh'), (20, 'linear')]
        for i in range(n_workers):
            runners[i].env.learning_channel_model = \
                SLAL(buffer_size=10000, channel_param=params['ch_param'], hidden_layers=ch_hidden_layers,
                     city=params['city'], device=args.device)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    global_data_evl, global_reward_evl, _, _, _ = runners[0].evaluate()
    global_data.append(global_data_evl)
    global_rewards.append(global_reward_evl)

    rewards, data, est_device_pos_list = [], [], []
    #  episodes_steps is the number of collected episodes of each worker
    train_steps, evaluate_steps, episode_steps = [0] * n_workers, [-1] * n_workers, [0] * n_workers
    best_episode_data = [0] * n_workers

    for k in range(int(total_episodes / aggregation_period)):
        if args.model:
            if episode_steps[0] % model_learning_period == 0:
                # Set the position of the users to the default position for evaluation
                runners[0].env.set_default_position()
                # Update the channel model and estimate the positions of the users
                est_device_pos = runners[0].model_learning(k)
                est_device_pos_list.append(est_device_pos)
                print('Estimated device position{}: '.format(episode_steps[0] // model_learning_period), est_device_pos)
                model_params = runners[0].env.learning_channel_model.model.parameters()
                model_params = torch.nn.utils.parameters_to_vector(model_params)

                for i in range(n_workers):
                    runners[i].env.set_device_position(est_device_pos)
                    torch.nn.utils.vector_to_parameters(model_params,
                                                        runners[i].env.learning_channel_model.model.parameters())
        params_q = []
        params_rnn = []
        data_sum = 0

        for i, runner in enumerate(runners):
            train_steps[i], evaluate_steps[i], episode_steps[i], best_episode_data[i] = \
                runner.federated_run(i, train_steps[i], evaluate_steps[i], episode_steps[i], data_train_list[i],
                                     reward_train_list[i], data_evl_list[i], reward_evl_list[i], best_episode_data[i],
                                     model=args.model)
            # Convert the parameters of each model to a vector
            params_q.append(torch.nn.utils.parameters_to_vector(runner.agents.policy.eval_qmix_net.parameters()))
            params_rnn.append(torch.nn.utils.parameters_to_vector(runner.agents.policy.eval_rnn.parameters()))
            data_sum += data_evl_list[i][-1]

        # Average the vectors according to the performance of each model
        if args.aggregation_method == 'weighted_average':
            weighted_params_q = []
            weighted_params_rnn = []
            for i in range(n_workers):
                weighted_params_q.append((data_evl_list[i][-1] / data_sum) * params_q[i])
                weighted_params_rnn.append((data_evl_list[i][-1] / data_sum) * params_rnn[i])
            average_params_q = sum(weighted_params_q)
            average_params_rnn = sum(weighted_params_rnn)
            
        # Average the vectors
        elif args.aggregation_method == 'average':
            average_params_q = sum(params_q) / n_workers
            average_params_rnn = sum(params_rnn) / n_workers

        # Set the average value as the parameters of the models
        for j in range(n_workers):
            torch.nn.utils.vector_to_parameters(average_params_q, runners[j].agents.policy.eval_qmix_net.parameters())
            torch.nn.utils.vector_to_parameters(average_params_rnn, runners[j].agents.policy.eval_rnn.parameters())

        # Evaluate the performance of the aggregated model
        global_data_evl, global_reward_evl, _, _, _ = runners[0].evaluate()
        global_data.append(global_data_evl)
        global_rewards.append(global_reward_evl)
        print('Global collected data : ', global_data_evl)
        runners[0].agents.policy.save_model()

        np.save(save_path + '/episode_data_{}'.format(i), data_evl_list[i])
        np.save(save_path + '/episode_rewards_{}'.format(i), reward_evl_list[i])
        np.save(save_path + '/episode_data_train_{}'.format(i), data_train_list[i])
        np.save(save_path + '/episode_rewards_train_{}'.format(i), reward_train_list[i])
        np.save(save_path + '/global_data', global_data)
        np.save(save_path + '/global_rewards', global_rewards)
        np.save(save_path + '/est_device_pos', est_device_pos_list)
        np.save(save_path + '/channel_loss', runners[0].channel_loss_list)


def train(args, params):
    learning_channel_model = None
    if args.model:
        ch_hidden_layers = [(50, 'tanh'), (20, 'linear')]
        learning_channel_model = SLAL(buffer_size=10000, channel_param=params['ch_param'], hidden_layers=ch_hidden_layers,
                                      city=params['city'], device=args.device)
    env = DataCollection(args,
                         learning_channel_model=learning_channel_model,
                         params=params,
                         )

    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    runner = Runner(env, args)

    runner.run(model=args.model)
    log_path = args.result_dir + '/' + args.alg + '/' + args.map + args.tag + '/' + 'log_params.txt'
    with open(log_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)




