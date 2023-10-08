import torch
import random
import numpy as np
import os
import argparse
from torch.optim.lr_scheduler import StepLR
from model import BertSolver, MLPSolver, BertValuedSolver
from dataloader import get_train_loader, get_test_loader
from train import default_train

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
parser.add_argument('--dataset_size', type=int, default=100000)
parser.add_argument('--test_size', type=int, default=500)
parser.add_argument('--train_enlarge', type=int, default=1)
parser.add_argument('--test_enlarge', type=int, default=1)
parser.add_argument('--combine_zero', type=int, default=0)
parser.add_argument('--iterate_game_types', type=int, default=1)
parser.add_argument('--random_dataset', type=int, default=0)
parser.add_argument('--sample_num',type=int,default=10000)
# model setting
parser.add_argument('--use_bert', type=int, default=1)
parser.add_argument('--requires_value',type=int, default=1)
parser.add_argument('--pretrain_requires_grad', type=int, default=0)
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--mlp_layers', type=int, default=2)
parser.add_argument('--use_pos_emb', type=int, default=1)
parser.add_argument('--query_mlp', type=int, default=1)
parser.add_argument('--from_pretrain', type=int, default=1)
parser.add_argument('--data_parallel', type=int, default=0)
parser.add_argument('--final_mlp', type=int, default=1)
parser.add_argument('--model_name', type=str, default='bert', choices=['bert','bert_small','bert_large','gpt-2'])
parser.add_argument('--entry_emb', type=int, default=None)
# train setting
parser.add_argument('--save_model', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--only_update_max', type=int, default=1)
parser.add_argument('--update_random', type=int, default=0)
parser.add_argument('--use_wandb', type=int, default=0)
parser.add_argument('--add_policy_loss', type=int, default=0)
parser.add_argument('--policy_loss', type=str, default='ce', choices=['l1','ce'])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--softmax_out', type=int, default=1)
parser.add_argument('--add_drop', type=int, default=0)
parser.add_argument('--log_results', type=int, default=1)
parser.add_argument('--random_tie', type=int, default=0)
parser.add_argument('--select_highest', type=int, default=1)
parser.add_argument('--detach_market', type=int, default=0)
parser.add_argument('--query_style', type=str, default='branch_add',
                    choices=['cat', 'branch_lcat', 'branch_rcat', 'branch_add', 'test_raw'])
parser.add_argument('--detach_branch', type=int, default=1)

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')

    # set seed
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.manual_seed(42)

    # get dataloader
    train_loader, test_loader = get_train_loader(args), get_test_loader(args)

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

    # set optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_decay > 0:
        scheduler = StepLR(optim, step_size=args.lr_decay, gamma=0.5)
    else:
        scheduler = None

    default_train(args, model, train_loader, test_loader, optim, scheduler, device)
