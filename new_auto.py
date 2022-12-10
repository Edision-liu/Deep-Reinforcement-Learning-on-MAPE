# -- coding:utf-8 --
# From: 刘志浩
# Date: 2021/3/11  10:23

import numpy as np
from main import train
import random
from arguments import get_args
import torch
from argparse import Namespace

np.set_printoptions(suppress=True, precision=4)

if __name__ == '__main__':
    args = get_args()
    if args.seed is None:
        args.seed = random.randint(0, 10000)
    args.num_updates = args.num_frames // args.num_steps // args.num_processes
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # NA = [3, 5, 7, 10]
    # last_savedir = './ep12200.pt'
    args_copy = Namespace(**vars(args))
    # if args.identity_size > 0:
    #     assert args.identity_size >= max(NA), 'identity size should either be 0 or >= number of agents!'

    # for i in range(len(NA)):
    args.save_dir = args_copy.save_dir + '/' + str(500)
    args.log_dir = args.save_dir + '/logs'
    args.num_agents = 50

    args.continue_training = True
    args.load_dir = './ep12200.pt'
    last_savedir = train(args, return_early=True)
