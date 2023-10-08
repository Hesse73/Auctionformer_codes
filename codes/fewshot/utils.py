import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import wandb


def get_model_name(args):
    mechanism_name = ''
    # order: first second
    args.mechanisms.sort()
    for m in args.mechanisms:
        if m == 'first':
            mechanism_name += 'f|'
        elif m == 'second':
            mechanism_name += 's|'
        else:
            raise ValueError(f"Unknown mechanism name:{m}")
    if args.max_entry > 0:
        mechanism_name += 'e|'
    mechanism_name = mechanism_name[:-1]

    distribution_name = ''
    # order: uniform gaussian
    args.distributions.sort(reverse=True)
    for dist in args.distributions:
        if dist == 'uniform':
            distribution_name += 'u|'
        elif dist == 'gaussian':
            distribution_name += 'g|'
    distribution_name += 'z' if args.start_from_zero else 'nz'

    update_mode = 'update_max' if args.only_update_max else 'update_all'

    if args.requires_value:
        model = 'Bert+NV.' + args.query_style.split('_')[-1]
    else:
        model = 'Bert' if args.use_bert else 'MLP'

    loss_name = 'dExpl|' if args.detach_market else 'Expl|'
    if args.add_policy_loss:
        loss_name += 'ce|' if args.policy_loss == 'ce' else 'l1|'
    loss_name = loss_name[:-1]

    dataset_aug = 'iter|' if args.iterate_game_types else 'raw|'
    dataset_aug += '+0' if args.combine_zero else 'x'

    bert_onoff = 'on' if args.pretrain_requires_grad else 'off'
    fix_player = 'fix_' if args.fix_player else ''

    return f'ft_{mechanism_name}_{distribution_name}_{dataset_aug}_{fix_player}n={args.max_player}_v={args.valuation_range - 1}_{model}_{bert_onoff}_{loss_name}_batch={args.batch_size}_lr={args.lr}_{update_mode}_epoch={args.max_epoch}'


def get_model_path(args):
    return 'models/' + get_model_name(args)


def plot_strategy(valuation, strategy, path, filename='bid_strategy'):
    path = os.path.join('results/', path)
    if not os.path.exists(path):
        os.makedirs(path)
    # remove invalid values
    max_value = valuation.cumsum().argmax()
    strategy[strategy < 1e-4] = 0
    strategy = strategy[:max_value + 1]

    V, B = strategy.shape
    # plot heatmap with ticks
    fig, ax = plt.subplots()
    im = ax.imshow(strategy)

    ax.set_xticks(np.arange(V))
    ax.set_yticks(np.arange(B))

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('bid probability', rotation=-90, va='bottom')

    plt.ylabel('valuation')
    plt.xlabel('bid')

    plt.savefig(os.path.join(path, filename))

    plt.close()


def calculate_symmetric_acc(max_value, bidding_strategy, player_num, mechanism):
    all_count = 0
    hit_count = 0
    for player_id in range(player_num):
        cur_strategy = bidding_strategy[player_id]  # V*V
        for value in range(max_value + 1):
            bid = cur_strategy[value].argmax()
            if mechanism == 'first':
                true_bid = int(value * (player_num - 1.0) / player_num)
            else:
                true_bid = value
            if bid == true_bid:
                hit_count += 1
            all_count += 1
    return hit_count / all_count


def make_uniform_value_hist(agt_values, valuation_range, player_num):
    """
    value histogram of users
    input: valuation_range [V1, V2, ...]
    output: probability histogram [[a,a,a,a,0,0,0,], [b,b,b,0,0,0], ...]
    """
    all_value_hist = np.zeros([player_num, valuation_range], dtype=float)
    for id, value in enumerate(agt_values):
        v_range = value + 1
        all_value_hist[id][:v_range] = 1 / v_range
    return all_value_hist


def generate_test_strategy(y, cur_player_num, device):
    # random strategy
    random_y = torch.rand_like(y).softmax(dim=-1)
    # zero strategy
    zero_y = torch.zeros_like(y)  # B*N*V*V
    zero_y[:, :, :, 0] = 1
    # truthful strategy
    B, N, V = y.shape[:3]
    truthful_y = torch.arange(V).view(-1, 1) == torch.arange(V)  # V*V
    truthful_y = truthful_y.float().to(device).view(1, 1, V, V).repeat(B, N, 1, 1)
    # trivial strategy
    trivial_bid = (((cur_player_num - 1) / cur_player_num).view(-1, 1).repeat(1, V) * torch.arange(V).to(
        device)).int()  # B*V
    trivial_y = trivial_bid.unsqueeze(-1) == torch.arange(V).to(device)  # B*V*V
    trivial_y = trivial_y.float().view(B, 1, V, V).repeat(1, N, 1, 1)  # B*N*V*V

    return random_y, zero_y, truthful_y, trivial_y


def init_wandb(args):
    network = 'Bert' if args.use_bert else 'MLP'
    update_mode = 'max' if args.only_update_max else 'all'

    wandb.init(
        project='BertFewShot-v2',

        config={
            'mechanisms': args.mechanisms,
            'distributions': args.distributions,
            'architecture': network,
            "update_mode": update_mode,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.max_epoch,
            "player_num": args.max_player,
            'valuation_range': args.valuation_range,
            "saved_name": get_model_name(args),
            "gpu": args.gpu,
        }
    )
