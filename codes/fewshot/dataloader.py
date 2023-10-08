import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import json
from scipy.stats import norm


def get_dataset_name(fix_player, max_player, upper_value, dataset_num, start_from_zero):
    postfix = 'zero' if start_from_zero else 'nonzero'
    prefix = 'fix_' if fix_player else ''
    return f'{prefix}N={max_player}_V={upper_value}_{postfix}_{dataset_num}games.json'


def mechanism_encoding(mechanism, cur_entry):
    enc = {'first': 0, 'second': 1}
    return enc[mechanism] + (2 * int(cur_entry > 0))


class RandomHybridTrainDataset(Dataset):

    def __init__(self, args, requires_value=False):
        self.mechanisms = args.mechanisms
        self.distributions = args.distributions
        self.max_player = args.max_player
        self.valuation_range = args.valuation_range
        self.upper_value = self.valuation_range - 1
        self.max_entry = args.max_entry
        self.has_entry = args.max_entry > 0
        self.test_size = args.test_size
        self.fix_player = args.fix_player

        self.requires_value = requires_value

        # load test dataset infor for repetition check
        prefix = 'fix_' if self.fix_player else ''
        self.zero_dataset = json.load(open(f'./dataset/{prefix}N=10_V=20_zero_500game_info.json','r'))
        self.nonzero_dataset = json.load(open(f'./dataset/{prefix}N=10_V=20_nonzero_500game_info.json','r'))

        self.zeros = ['zero'] if args.start_from_zero else ['nonzero']
        if args.combine_zero and not args.start_from_zero:
            self.zeros = ['zero', 'nonzero']

        # train:, iterate over mechanisms / distributions / entries / u0 or u
        self.iteration = len(args.mechanisms) * len(args.distributions) * (args.max_entry + 1) * len(self.zeros)
        self.dataset_size = args.sample_num * self.iteration

        print(f'Successfully load random train dataset with size of {len(self)}, '
              f'iterated {self.iteration} times.')

    def __len__(self):
        return int(self.dataset_size)

    def __getitem__(self, idx):
        # select mechanism by idx
        game_id, game_type = idx % self.iteration, []
        for ntypes in [(self.max_entry + 1), len(self.distributions), len(self.mechanisms), len(self.zeros)]:
            game_type.append(game_id % ntypes)
            game_id = int(game_id / ntypes)
        cur_entry, cur_dist_type, cur_mechanism, cur_zero = (game_type[0], self.distributions[game_type[1]],
                                                   self.mechanisms[game_type[2]], self.zeros[game_type[3]])

        # random pick player_num & distribution
        is_repeated = True
        while is_repeated:
            if self.fix_player:
                cur_player_num = self.max_player
            else:
                cur_player_num = np.random.randint(2, self.max_player + 1)
            cur_value_hist, cur_value_ranges = [], []
            for _ in range(cur_player_num):
                player_val_size = np.random.randint(1, self.valuation_range)  # [1,200]
                if cur_zero == 'zero':
                    check_dataset = self.zero_dataset
                    player_lower_value = 0
                else:
                    check_dataset = self.nonzero_dataset
                    player_lower_value = np.random.randint(0, self.upper_value - player_val_size + 1)
                player_upper_value = player_lower_value + player_val_size
                player_value_range = [player_lower_value, player_upper_value]  # [a,b]
                player_value_hist = np.zeros(self.valuation_range)
                if cur_dist_type == 'uniform':
                    # uniform value hist
                    player_value_hist[player_lower_value:player_upper_value+1] = 1 / (player_val_size+1)
                else:
                    # gaussian value hist
                    player_possible_values = np.arange(player_lower_value,player_upper_value+1)
                    mean, size = (player_lower_value + player_upper_value) / 2, player_val_size
                    transform = lambda x: 6 / size * (x - mean)
                    gaussian_value_hist = norm.cdf(transform(player_possible_values + 0.5)) - norm.cdf(transform(player_possible_values - 0.5))
                    gaussian_value_hist[0] = norm.cdf(transform(player_lower_value + 0.5))
                    gaussian_value_hist[-1] = 1 - norm.cdf(transform(player_upper_value - 0.5))
                    player_value_hist[player_lower_value:player_upper_value+1] = gaussian_value_hist
                cur_value_hist.append(player_value_hist)
                cur_value_ranges.append(player_value_range)
            # check repetition
            is_repeated = cur_value_ranges in check_dataset[str(cur_player_num)]
        player_value_dists = np.zeros([self.max_player, self.valuation_range])
        player_value_dists[:cur_player_num] = cur_value_hist
        # if requires value, return N*V values
        if self.requires_value:
            player_values = torch.arange(self.valuation_range).view(-1, 1).repeat(1, self.max_player).flatten()  # (V*N)
            player_values = player_values.unsqueeze(-1) == torch.arange(self.valuation_range)  # (V*N) * V
            return mechanism_encoding(cur_mechanism,
                                      cur_entry), cur_entry, cur_player_num, player_value_dists, player_values
        else:
            return mechanism_encoding(cur_mechanism, cur_entry), cur_entry, cur_player_num, player_value_dists


class FixedHybridDataset(Dataset):

    def __init__(self, args, enlarge_times=1, is_test=False, requires_value=False, filename=None):
        self.mechanisms = args.mechanisms
        self.distributions = args.distributions
        self.max_player = args.max_player
        self.valuation_range = args.valuation_range
        self.upper_value = self.valuation_range - 1
        self.max_entry = args.max_entry
        self.has_entry = args.max_entry > 0
        self.test_size = args.test_size

        self.enlarge_times = enlarge_times
        self.is_test = is_test
        self.requires_value = requires_value

        if filename is not None:
            dataset_name = filename
        else:
            dataset_name = get_dataset_name(args.fix_player, self.max_player, self.upper_value, args.dataset_size, args.start_from_zero)
        self.dataset = json.load(open('./dataset/' + dataset_name, 'r'))

        if not is_test and args.start_from_zero == 0 and args.combine_zero and filename is None:
            # combine zero dataset
            zero_dataset_name = get_dataset_name(args.fix_player, self.max_player, self.upper_value, args.dataset_size, 1)
            zero_dataset = json.load(open('./dataset/' + zero_dataset_name, 'r'))
            print('Successfully load zero dataset to combine with current data.')
            for key in self.dataset:
                self.dataset[key] += zero_dataset[key]

        self.iteration = len(args.mechanisms) * len(args.distributions) * (args.max_entry + 1)
        if self.is_test:
            # test: first K items, iterate over mechanisms / distributions / entries
            self.dataset_size = self.test_size * self.iteration
        else:
            # train: data[K:], iterate over mechanisms / distributions / entries
            self.dataset_size = (len(self.dataset['number']) - self.test_size) * self.iteration

        prefix = 'test' if is_test else 'train'
        print(f'Successfully load {prefix} dataset:{dataset_name}, with size of {len(self)}, '
              f'iterated {self.iteration} times.')

    def __len__(self):
        return int(self.dataset_size * self.enlarge_times)

    def __getitem__(self, idx):
        idx = int(idx / self.enlarge_times)  # dataset enlarge
        cur_idx = int(idx / self.iteration)  # iteration over mechanisms
        if not self.is_test:
            cur_idx = cur_idx + self.test_size  # train start from K

        # select mechanism by idx
        game_id, game_type = idx % self.iteration, []
        for ntypes in [(self.max_entry + 1), len(self.distributions), len(self.mechanisms)]:
            game_type.append(game_id % ntypes)
            game_id = int(game_id / ntypes)
        cur_entry, cur_dist_type, cur_mechanism = (game_type[0], self.distributions[game_type[1]],
                                                   self.mechanisms[game_type[2]])

        cur_player_num = self.dataset['number'][cur_idx]
        # uniform_value_hist or gaussian_value_hist (already padded to V)
        cur_player_dist = self.dataset[f'{cur_dist_type}_value_hist'][cur_idx]  # N_cur *V
        player_value_dists = np.array(
            cur_player_dist + [np.zeros(self.valuation_range)] * (self.max_player - cur_player_num))

        # if requires value, return N*V values
        if self.requires_value:
            player_values = torch.arange(self.valuation_range).view(-1, 1).repeat(1, self.max_player).flatten()  # (V*N)
            player_values = player_values.unsqueeze(-1) == torch.arange(self.valuation_range)  # (V*N) * V
            return mechanism_encoding(cur_mechanism,
                                      cur_entry), cur_entry, cur_player_num, player_value_dists, player_values
        else:
            return mechanism_encoding(cur_mechanism, cur_entry), cur_entry, cur_player_num, player_value_dists


class HybridDataset(Dataset):
    """
    Hybrid mechanism & distribution dataset.

    The valuation distributions are generated by 'dataset/generate_game.py',
    including 'uniform' values and 'gaussian' values of different players,
    specified by args.distributions.

    While the mechanisms (first, second, first+entry and second+entry) are randomly picked
    each time the data is loaded, since it's independent of valuation distribution.
    """

    def __init__(self, args, enlarge_times=1, is_test=False, requires_value=False, filename=None):
        self.mechanisms = args.mechanisms
        self.distributions = args.distributions
        self.max_player = args.max_player
        self.valuation_range = args.valuation_range
        self.upper_value = self.valuation_range - 1
        self.max_entry = args.max_entry
        self.has_entry = args.max_entry > 0
        self.test_size = args.test_size

        self.enlarge_times = enlarge_times
        self.is_test = is_test
        self.requires_value = requires_value

        if filename is not None:
            dataset_name = filename
        else:
            dataset_name = get_dataset_name(args.fix_player, self.max_player, self.upper_value, args.dataset_size, args.start_from_zero)
        self.dataset = json.load(open('./dataset/' + dataset_name, 'r'))

        if not is_test and args.start_from_zero == 0 and args.combine_zero and filename is None:
            # combine zero dataset
            zero_dataset_name = get_dataset_name(args.fix_player, self.max_player, self.upper_value, args.dataset_size, 1)
            zero_dataset = json.load(open('./dataset/' + zero_dataset_name, 'r'))
            print('Successfully load zero dataset to combine with current data.')
            for key in self.dataset:
                self.dataset[key] += zero_dataset[key]

        if self.is_test:
            self.dataset_size = self.test_size
        else:
            self.dataset_size = len(self.dataset['number']) - self.test_size

        prefix = 'test' if is_test else 'train'
        print(f'Successfully load {prefix} dataset:{dataset_name}, with size of {len(self)}')

    def __len__(self):
        return int(self.dataset_size * self.enlarge_times)

    def __getitem__(self, idx):

        cur_idx = int(idx / self.enlarge_times)
        if not self.is_test:
            cur_idx = cur_idx + self.test_size

        # game type
        if not self.is_test:
            # train: random sample game types
            cur_mechanism = random.choice(self.mechanisms)
            cur_entry = random.randint(0, self.max_entry)
            cur_dist_type = random.choice(self.distributions)
        else:
            # test: select by idx
            game_id, game_type = idx % self.enlarge_times, []
            for ntypes in [(self.max_entry + 1), len(self.distributions), len(self.mechanisms)]:
                game_type.append(game_id % ntypes)
                game_id = int(game_id / ntypes)
            cur_entry, cur_dist_type, cur_mechanism = (game_type[0], self.distributions[game_type[1]],
                                                       self.mechanisms[game_type[2]])

        cur_player_num = self.dataset['number'][cur_idx]
        # uniform_value_hist or gaussian_value_hist (already padded to V)
        cur_player_dist = self.dataset[f'{cur_dist_type}_value_hist'][cur_idx]  # N_cur *V
        player_value_dists = np.array(
            cur_player_dist + [np.zeros(self.valuation_range)] * (self.max_player - cur_player_num))

        # if requires value, return N*V values
        if self.requires_value:
            player_values = torch.arange(self.valuation_range).view(-1, 1).repeat(1, self.max_player).flatten()  # (V*N)
            player_values = player_values.unsqueeze(-1) == torch.arange(self.valuation_range)  # (V*N) * V
            return mechanism_encoding(cur_mechanism,
                                      cur_entry), cur_entry, cur_player_num, player_value_dists, player_values
        else:
            return mechanism_encoding(cur_mechanism, cur_entry), cur_entry, cur_player_num, player_value_dists


def get_train_loader(args, filename=None):
    if args.iterate_game_types:
        if args.random_dataset:
            train_loader = DataLoader(RandomHybridTrainDataset(args,requires_value=args.requires_value),
                                      batch_size=args.batch_size, shuffle=True)
        else:
            train_loader = DataLoader(FixedHybridDataset(args, enlarge_times=args.train_enlarge, is_test=False,
                                                         requires_value=args.requires_value, filename=filename),
                                      batch_size=args.batch_size, shuffle=True)
    else:
        train_loader = DataLoader(HybridDataset(args, enlarge_times=args.train_enlarge, is_test=False,
                                                requires_value=args.requires_value, filename=filename),
                                  batch_size=args.batch_size, shuffle=True)
    return train_loader


def get_test_loader(args, filename=None):
    if args.iterate_game_types:
        test_loader = DataLoader(FixedHybridDataset(args, enlarge_times=1, is_test=True,
                                                    requires_value=args.requires_value, filename=filename),
                                 batch_size=args.batch_size, shuffle=False)
    else:
        # when testing, enlarge_times is exactly game types
        test_enlarge = len(args.mechanisms) * len(args.distributions) * (args.max_entry + 1)
        test_loader = DataLoader(HybridDataset(args, enlarge_times=test_enlarge, is_test=True,
                                               requires_value=args.requires_value, filename=filename),
                                 batch_size=args.batch_size, shuffle=False)
    return test_loader


def get_full_test_loader(args):
    from copy import deepcopy
    new_args = deepcopy(args)
    new_args.mechanisms = ['first', 'second']
    new_args.distributions = ['uniform', 'gaussian']
    new_args.max_entry = 3
    new_args.max_player = 10
    new_args.fix_player = 0
    return get_test_loader(new_args)