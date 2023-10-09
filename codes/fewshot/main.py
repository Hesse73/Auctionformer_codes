import torch
import random
import numpy as np
import os
import argparse
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from model import BertSolver, MLPSolver, BertValuedSolver
from dataloader import get_train_loader, get_test_loader
from train import default_train, dream_booth_train

parser = argparse.ArgumentParser()

# game setting
parser.add_argument('--mechanisms', nargs='+', default=['second'],
                    help="mechanisms in ['first', 'second']")
parser.add_argument('--max_player', type=int, default=10)
parser.add_argument('--valuation_range', type=int, default=21)
parser.add_argument('--max_entry', type=int, default=0)
# data setting
parser.add_argument('--distributions', nargs='+', default=['uniform'],
                    help="distribution in ['uniform', 'gaussian']")
parser.add_argument('--start_from_zero', type=int, default=1)
parser.add_argument('--dataset_size', type=int, default=2000)
parser.add_argument('--test_size', type=int, default=500)
parser.add_argument('--train_enlarge', type=int, default=1)
parser.add_argument('--test_enlarge', type=int, default=1)
parser.add_argument('--combine_zero', type=int, default=0)
parser.add_argument('--iterate_game_types', type=int, default=1)
parser.add_argument('--random_dataset', type=int, default=0)
parser.add_argument('--sample_num', type=int, default=10000)
parser.add_argument('--fix_player', type=int, default=0)
# model setting
parser.add_argument('--use_bert', type=int, default=1)
parser.add_argument('--requires_value', type=int, default=1)
parser.add_argument('--pretrain_requires_grad', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--mlp_layers', type=int, default=2)
parser.add_argument('--use_pos_emb', type=int, default=1)
parser.add_argument('--query_mlp', type=int, default=1)
parser.add_argument('--from_pretrain', type=int, default=1)
parser.add_argument('--data_parallel', type=int, default=0)
parser.add_argument('--fix_emb_q', type=int, default=0)
parser.add_argument('--entry_emb', type=int, default=4)
# train setting
parser.add_argument('--save_model', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--lr_decay', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--only_update_max', type=int, default=1)
parser.add_argument('--update_random', type=int, default=0)
parser.add_argument('--use_wandb', type=int, default=0)
parser.add_argument('--add_policy_loss', type=int, default=0)
parser.add_argument('--policy_loss', type=str, default='ce', choices=['l1', 'ce'])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--softmax_out', type=int, default=1)
parser.add_argument('--add_drop', type=int, default=0)
parser.add_argument('--log_results', type=int, default=0)
parser.add_argument('--random_tie', type=int, default=0)
parser.add_argument('--select_highest', type=int, default=1)
parser.add_argument('--detach_market', type=int, default=0)
parser.add_argument('--query_style', type=str, default='branch_add',
                    choices=['cat', 'branch_lcat', 'branch_rcat', 'branch_add'])
parser.add_argument('--detach_branch', type=int, default=1)
parser.add_argument('--target_mode', type=str, default='sp', choices=['number','sp','entry','compensate','sym'])
parser.add_argument('--target_number', type=int, default=15, choices=[15,20])
parser.add_argument('--target_dist', nargs='+', default=['uniform'], choices=['uniform', 'gaussian'])
parser.add_argument('--use_dream_booth', type=int, default=0)
parser.add_argument('--preserve_ratio', type=float, default=10)
parser.add_argument('--eval_freq', type=int, default=1)

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    # set seed
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.manual_seed(42)

    # different mode
    if args.target_mode == 'number':
        # mix trained, few-shot on player_num = 15
        # model_name = ('0.04809_f|s|e_u|g|nz_iter|+0_n=10_v=20_Bert+NV.add_on_Expl_'
        #               'batch=1024_lr=0.0001_update_max_epoch=300')
        # model_name = ('f|s|e_u|g|nz_iter|+0_n=10_v=20_Bert+NV.add_on_Expl_batch=1024'
        #               '_lr=0.0001_update_max_epoch=300')
        model_name = ('f|s|e_u|g|nz_iter|+0_n=10_v=20_bert+NV.add_on_Expl_batch=1024'
                      '_lr=0.0001_update_max_epoch=300')
        # old dataloader: all-mixed
        args.mechanisms, args.distributions = ['first', 'second'], ['uniform', 'gaussian']
        args.max_entry, args.start_from_zero = 3, 0
        old_train = get_train_loader(args)

        # load old test loader
        args.test_size = 12000  # 2000 * (zero + nonzero) * (3 split)
        old_test = get_test_loader(args, 'testset_2000.json')
        args.test_size = 500

        # target dataloader: number = 15/20
        default_args = args.max_player, args.fix_player
        args.max_player, args.fix_player = args.target_number, True
        target_train = get_train_loader(args)

        # load target test loader
        args.test_size = 4000  # 2000 * (zero + nonzero)
        target_test = get_test_loader(args, f'{args.target_number}_testset_2000.json')
        args.test_size = 500

        args.max_player, args.fix_player = default_args
    elif args.target_mode == 'sp':
        # train on fp, few-shot on sp
        if args.target_dist == ['uniform']:
            dist_name = 'u'
        elif args.target_dist == ['gaussian']:
            dist_name = 'g'
        else:
            dist_name = 'u|g'
        model_name = (f'f_{dist_name}|z_iter|x_n=10_v=20_bert+NV.add_on_Expl_'
                      'batch=1024_lr=1e-05_update_max_epoch=300')
        # old dataloader: fp, U0
        args.mechanisms, args.distributions = ['first'], args.target_dist
        args.max_entry, args.start_from_zero = 0, 1
        old_train = get_train_loader(args)

        # load old test loader
        args.test_size = 6000
        old_test = get_test_loader(args, 'zero_testset_2000.json')
        args.test_size = 500

        # target dataloader: sp
        default_args = args.mechanisms
        args.mechanisms = ['second']
        target_train = get_train_loader(args)

        # load target test loader
        args.test_size = 6000
        target_test = get_test_loader(args, 'zero_testset_2000.json')
        args.test_size = 500

        args.mechanisms = default_args
    elif args.target_mode == 'entry':
        # train on fp+sp, few-shot on fp+sp & entry
        model_name = ('f|s_u|z_iter|x_n=10_v=20_bert+NV.add_on_Expl_'
                      'batch=1024_lr=1e-05_update_max_epoch=300')
        # old dataloader: fp+sp, u0
        args.mechanisms, args.distributions = ['first','second'], ['uniform']
        args.max_entry, args.start_from_zero = 0, 1
        old_train, old_test = get_train_loader(args), get_test_loader(args)
        # target dataloader: +entry
        default_args = args.max_entry
        args.max_entry = 3
        target_train, target_test = get_train_loader(args), get_test_loader(args)
        args.max_entry = default_args
    elif args.target_mode == 'compensate':
        # a special one, compensate for SP+entry training
        model_name = ('0.04809_f|s|e_u|g|nz_iter|+0_n=10_v=20_Bert+NV.add_on_Expl_'
                      'batch=1024_lr=0.0001_update_max_epoch=300')
        # old dataloader: all-mixed
        args.mechanisms, args.distributions = ['first', 'second'], ['uniform', 'gaussian']
        args.max_entry, args.start_from_zero = 3, 0
        old_train, old_test = get_train_loader(args), get_test_loader(args)
        # target dataloader: SP+Entry
        default_args = args.mechanisms
        args.mechanisms = ['second']
        target_train, target_test = get_train_loader(args), get_test_loader(args)
        args.mechanisms = default_args
    elif args.target_mode == 'sym':
        model_name = ('f|s|e_u|g|nz_iter|+0_n=10_v=20_Bert+NV.add_on_Expl_batch=1024'
                      '_lr=0.0001_update_max_epoch=300')
        # old dataloader:
        args.max_epoch = 100
        args.mechanisms, args.distributions = ['first'], ['uniform']
        args.max_entry, args.start_from_zero = 0, 1
        old_train = None
        # old test
        args.test_size = 180
        old_test = get_test_loader(args, 'N=10_V=20_sym_games.json')
        # target train
        args.test_size = 0
        args.train_enlarge = 50
        target_train = get_train_loader(args, 'N=10_V=20_sym_games.json')
        args.train_enlarge = 1
        # target test
        args.test_size = 180
        target_test = get_test_loader(args, 'N=10_V=20_sym_games.json')
    else:
        raise NotImplementedError

    # set model
    if args.requires_value:
        if args.use_bert:
            model = BertValuedSolver(args).to(device)
        else:
            raise NotImplementedError
    else:
        if args.use_bert:
            model = BertSolver(args).to(device)
        else:
            model = MLPSolver(args).to(device)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    load_path = os.path.join('../models/', model_name)
    model.load_state_dict(torch.load(load_path))
    print('Successfully load model:', model_name)

    # set optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_decay > 0:
        scheduler = StepLR(optim, step_size=args.lr_decay, gamma=0.5)
    else:
        scheduler = None

    if args.use_dream_booth:
        base_model = deepcopy(model)
        base_model.requires_grad_(False)
        dream_booth_train(args, model, base_model, target_train, target_test, old_train, old_test, optim, scheduler, device)
    else:
        default_train(args, model, target_train, target_test, old_test, optim, scheduler, device)
