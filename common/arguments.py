import argparse


def get_common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='RDM', help='the map of the environment')
    parser.add_argument('--tag', type=str, default='', help='specific tag for saving the model')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
    parser.add_argument('--n_steps', type=int, default=1000000, help='total time steps')
    parser.add_argument('--total_episodes', type=int, default=30000, help='total training episodes')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--federated', type=bool, default=False, help='whether trian the algorithm with federated learning')
    parser.add_argument('--aggregation_period', type=int, default=50, help='the frequency to aggregate the model')
    parser.add_argument('--aggregation_method', type=str, default='average', help='how to aggregate the model')
    parser.add_argument('--model_learning_period', type=int, default=1000, help='the frequency to learn the model')
    parser.add_argument('--last_action', type=bool, default=True,
                        help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="Adam", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=50, help='how often to evaluate the model')
    parser.add_argument('--evaluate_epoch', type=int, default=5, help='number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--device', type=str, default='cpu', help='The device to run the program')
    parser.add_argument('--info', type=str, default='', help='additional information of the model')
    parser.add_argument('--workers', type=int, default=3, help='the number of the workers, which is equal to the number of the UAVs')
    parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon value of epsilon-greedy')
    parser.add_argument('--min_epsilon', type=float, default=0.05, help='min epsilon value of epsilon-greedy')
    parser.add_argument('--anneal_steps', type=int, default=50000, help='the number of steps to anneal epsilon')
    parser.add_argument('--model', type=bool, default=False, help='whether use the model')
    parser.add_argument('--clear_localization_buff', type=bool, default=False,
                        help='whether clear the localization buffer')
    parser.add_argument('--buffer_size', type=int, default=5000, help='the size of the replay buffer')
    parser.add_argument('--n_agents', type=int, default=3, help='the number of the agents')
    parser.add_argument('--user_location_info', default=True, action='store_false', help='whether to use the user location in observation')
    parser.add_argument('--sample_method', type=str, default='random', help='how to choose the samples from the localization buffer') 

    args = parser.parse_args()
    return args


# arguments of qmix
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / args.anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 1

    # experience replay
    args.batch_size = 32

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args



