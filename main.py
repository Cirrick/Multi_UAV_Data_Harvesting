import random, numpy as np, torch
from common.arguments import get_common_args, get_mixer_args
from training_procedures import fl_train, train


def main():

    args = get_common_args()
    if args.alg == 'qmix':
        args = get_mixer_args(args)

    if args.map == 'RBM':
        from config.RBM_define import params
    elif args.map == 'RDM':
        from config.RDM_define import params
    else:
        raise Exception("No such map!")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.federated:
        fl_train(args, params)
    else:
        train(args, params)


if __name__ == '__main__':
    main()




