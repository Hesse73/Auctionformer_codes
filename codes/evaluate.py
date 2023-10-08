from main import *
from utils import plot_strategy, make_uniform_value_hist, calculate_symmetric_acc, get_model_name, get_model_path
from loss import calculate_exploit
from train import transpose_y0, generate_test_strategy
from metrics import *
from scipy.stats import norm
import time
import torch.utils.benchmark as benchmark


def generate_values(args, device):
    player_values = torch.arange(args.valuation_range).view(-1, 1).repeat(1, args.max_player).flatten()  # (V*N)
    player_values = player_values.unsqueeze(-1) == torch.arange(args.valuation_range)  # (V*N) * V
    return player_values.unsqueeze(0).float().to(device)  # 1*VN*2V


parser.add_argument('--ckpt_name', type=str, default='bert_bert_on')
parser.add_argument('--test_mode', type=str, default='symmetric',
                    choices=['symmetric', 'symmetric_time', 'distribution', 'number', 'mechanism', 'example',
                             'split', 'symmetric_number'])
parser.add_argument('--number_range', nargs='+', default=[11, 16])
parser.add_argument('--force_cpu', type=int, default=0)

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    if args.force_cpu:
        device = torch.device('cpu')

    model_name = args.ckpt_name
    # model_name = 'best_bert_off'
    # model_name = 'f|s|e_u|g|nz_iter|+0_n=10_v=20_Bert+NV.add_on_Expl_batch=1024_lr=0.0001_update_max_epoch=300'
    # model_name = 'f_u|z_n=3_v=20_bert=0_batch=1024_lr=0.0001_update_max_epoch=300'

    model_path = 'models/' + model_name

    args.batch_size = 256

    # set seed
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)
    torch.cuda.manual_seed(42)

    # get model
    if args.use_bert:
        if args.requires_value:
            model = BertValuedSolver(args).to(device)
        else:
            model = BertSolver(args).to(device)
    else:
        model = MLPSolver(args).to(device)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print('Successfully load model:', model_name)

    if args.test_mode in ['distribution', 'number', 'mechanism', 'split']:
        # test on different dataset
        if args.test_mode == 'distribution':
            # different distribution
            dataset_types = [(['uniform'], 1), (['gaussian'], 1), (['uniform', 'gaussian'], 1),
                             (['uniform'], 0), (['gaussian'], 0), (['uniform', 'gaussian'], 0)]
            datasets = []  # U_0, G_0, U+G
            for dist, zero_start in dataset_types:
                args.distributions, args.start_from_zero = dist, zero_start
                datasets.append(get_test_loader(args))
        elif args.test_mode == 'number':
            # different player_num
            if type(args.number_range[0]) is str:
                args.number_range = [int(x) for x in args.number_range]
            dataset_types = np.arange(*args.number_range)
            datasets = []
            for player_num in dataset_types:
                args.max_player = player_num
                args.test_size = 4000
                filename = f'{player_num}_testset_2000.json'
                print('loading file:', filename)
                datasets.append(get_test_loader(args, filename))
        elif args.test_mode == 'mechanism':
            mechanisms = [['first'], ['second'], ['first', 'second']]
            entries = [0, 3]
            dist_types = [(['uniform'], 1), (['gaussian'], 1), (['uniform'], 0), (['gaussian'], 0),
                          (['uniform', 'gaussian'], 0)]
            datasets = []  # U_0, G_0, U+G
            dataset_types = []
            for m in mechanisms:
                for e in entries:
                    for dist, zero_start in dist_types:
                        args.mechanisms = m
                        args.max_entry = e
                        args.distributions, args.start_from_zero = dist, zero_start
                        datasets.append(get_test_loader(args))
                        dataset_types.append((m, e, dist, zero_start))
        else:
            # iterate over all splited dataset & distribution & mechanism
            mechanisms = [['first'], ['second'], ['first', 'second']]
            entries = [0, 3]
            dist_types = [(['uniform'], 1), (['gaussian'], 1), (['uniform'], 0), (['gaussian'], 0),
                          (['uniform', 'gaussian'], None)]
            split_ranges = [(2, 4), (5, 7), (8, 10)]
            datasets = []  # U_0, G_0, U+G
            dataset_types = []
            for m in mechanisms:
                for e in entries:
                    for (lower, upper) in split_ranges:
                        for dist, zero_start in dist_types:
                            args.mechanisms = m
                            args.max_entry = e
                            args.distributions, args.start_from_zero = dist, zero_start
                            if zero_start == 1:
                                args.test_size = 2000
                                filename = f'zero_{lower}-{upper}_testset_2000.json'
                            elif zero_start == 0:
                                args.test_size = 2000
                                filename = f'no-zero_{lower}-{upper}_testset_2000.json'
                            else:
                                args.test_size = 4000
                                filename = f'{lower}-{upper}_testset_2000.json'
                            print('loading file:', filename)
                            datasets.append(get_test_loader(args, filename))
                            dataset_types.append((m, e, dist, zero_start, lower, upper))
            # add test all player on hybrid mechanism
            args.mechanisms, args.max_entry = ['first', 'second'], 3
            for dist, zero_start in dist_types:
                args.distributions, args.start_from_zero = dist, zero_start
                if zero_start == 1:
                    args.test_size = 6000
                    filename = f'zero_testset_2000.json'
                elif zero_start == 0:
                    args.test_size = 6000
                    filename = f'no-zero_testset_2000.json'
                else:
                    args.test_size = 12000
                    filename = f'testset_2000.json'
                print('loading file:', filename)
                datasets.append(get_test_loader(args, filename))
                dataset_types.append((m, e, dist, zero_start, None, None))

        avg_test_eps = [0, 0, 0, 0, 0]
        for test_type, test_loader in zip(dataset_types, datasets):
            print('test on:', test_type)
            test_eps = [0, 0, 0, 0, 0]
            with torch.no_grad():
                for X in test_loader:
                    if args.requires_value:
                        # B, B, B, B*N*V, B*NV
                        cur_mechanism, cur_entry, cur_player_num, value_dists, player_values = (data.to(device) for data
                                                                                                in X)
                        value_dists, player_values = value_dists.float(), player_values.float()
                        y0 = model((cur_mechanism, cur_entry, value_dists, player_values))  # B*N*V*V

                        y = transpose_y0(args, y0)
                    else:
                        # B, B, B, B*N*V
                        cur_mechanism, cur_entry, cur_player_num, value_dists = (data.to(device) for data in X)
                        value_dists = value_dists.float()
                        y = model((cur_mechanism, cur_entry, value_dists))  # B*N*V*V

                    random_y, zero_y, truthful_y, trivial_y = generate_test_strategy(y, cur_player_num, device)

                    for idx, strategy in enumerate([y, random_y, zero_y, truthful_y, trivial_y]):
                        exploits, _, _ = calculate_exploit(cur_mechanism, cur_entry, cur_player_num, value_dists,
                                                           strategy,
                                                           device, drop_mask_bn=None, overbid_punish=False,
                                                           random_tie=False)
                        max_exploits, _ = exploits.max(dim=-1)
                        test_eps[idx] += max_exploits.sum().cpu().item()
            eps_values = [eps / len(test_loader.dataset) / (args.valuation_range - 1) for eps in test_eps]
            predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = eps_values
            print(
                'Eps: [Predict | Random | Zero | Truthful | Trivial] = [%.2e | %.2e | %.2e | %.2e | %.2e]' % (
                    predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))
            # add to avg
            for idx, eps in enumerate(eps_values):
                avg_test_eps[idx] += eps
        # print avg eps across different dataset
        predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = (eps / len(datasets) for eps in avg_test_eps)
        print('AVG Eps: [Predict | Random | Zero | Truthful | Trivial] = [%.2e | %.2e | %.2e | %.2e | %.2e]' % (
            predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))

    elif args.test_mode == 'symmetric':
        # test on symmetric games (only support first/second price with no entry)
        enc = {'first': 0, 'second': 1, }
        for mechanism_name in args.mechanisms:
            mechanism = enc[mechanism_name]
            print('---------------------------------------------------------------')
            print(f"TEST ON SYMMETRIC {mechanism_name.upper()} PRICE GAMES")
            input_mechanism = torch.tensor([mechanism]).to(device)
            input_entry = torch.zeros_like(input_mechanism).to(device)
            test_eps, test_l2, test_interim = [[], [], [], [], []], [[], [], [], [], []], [[], [], [], [], []]
            for player_num in range(2, args.max_player + 1):
                player_max_value = [args.valuation_range - 1] * player_num + [0] * (args.max_player - player_num)
                symmetric_value_dists = torch.from_numpy(
                    make_uniform_value_hist(player_max_value, args.valuation_range, args.max_player))
                input_player_num = torch.tensor([player_num]).to(device)
                symmetric_value_dists[player_num:] = 0
                symmetric_value_dists = symmetric_value_dists.unsqueeze(0).float().to(device)  # 1*N*V
                with torch.no_grad():
                    # predicted policy
                    if args.requires_value:
                        # generate value realizations
                        player_values = generate_values(args, device)
                        y0 = model((input_mechanism, input_entry, symmetric_value_dists, player_values))
                        y = transpose_y0(args, y0)
                    else:
                        y = model((input_mechanism, input_entry, symmetric_value_dists))
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
                    trivial_y = ((player_num - 1) / player_num * torch.arange(V).view(-1, 1)).int() == torch.arange(
                        V)  # V*V
                    trivial_y = trivial_y.float().to(device).view(1, 1, V, V).repeat(B, N, 1, 1)
                    for idx, strategy in enumerate([y, random_y, zero_y, truthful_y, trivial_y]):
                        # calculate eps
                        exploits, _, _ = calculate_exploit(input_mechanism, input_entry, input_player_num,
                                                           symmetric_value_dists,
                                                           strategy, device, overbid_punish=False)
                        max_exploits, _ = exploits.max(dim=-1)
                        test_eps[idx].append(max_exploits.sum().cpu().item())
                        # # calculate acc
                        # acc = calculate_symmetric_acc(value, strategy[0], args.max_player, mechanism_name)
                        # test_acc[idx].append(acc)
                        # calculate L2
                        if mechanism_name == 'first':
                            analytic = ((player_num - 1) / player_num * torch.arange(V).float()).repeat(player_num,
                                                                                                        1).to(device)
                        else:
                            analytic = torch.arange(V).float().repeat(player_num, 1).to(device)
                        # max_l2, _ = torch.sqrt(torch.sum((strategy[0][:player_num] - target[0][:player_num])**2, dim=[1,2])).max(dim=-1)
                        # test_l2[idx].append(max_l2.cpu().item())
                        # l2_mean, l2_std = sample_l2_paper(strategy[0][:player_num], mechanism_name), torch.tensor(0)
                        l2_mean, l2_std = sample_l2_distance(strategy[0][:player_num], mechanism_name)
                        # l2_mean, l2_std = sample_l2_distance_old(symmetric_value_dists[0][:player_num],
                        #                                      strategy[0][:player_num], analytic)
                        test_l2[idx].append((l2_mean.cpu().item(), l2_std.cpu().item()))

                        interim_mean, interim_std = ex_interim_utility_loss(input_mechanism, input_entry,
                                                                            input_player_num, symmetric_value_dists,
                                                                            strategy, device)
                        test_interim[idx].append((interim_mean.cpu().item(), interim_std.cpu().item()))

                    # print results
                    predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = [eps_list[-1] for eps_list in
                                                                                    test_eps]
                    predict_l2, random_l2, zero_l2, truthful_l2, trivial_l2 = [l2_list[-1] for l2_list in
                                                                               test_l2]
                    predict_interim, random_interim, zero_interim, truthful_interim, trivial_interim = [interim_list[-1]
                                                                                                        for interim_list
                                                                                                        in test_interim]
                    fmt = lambda x: '%.5f (%.5f)' % (x[0], x[1])
                    print(f'Symmetric player num={player_num}')
                    print('Eps: [Predict | Random | Zero | Truthful | Trivial] = [%.5f | %.5f | %.5f | %.5f | %.5f]' % (
                        predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))
                    print(f'L2: [Predict | Random | Zero | Truthful | Trivial] = [{fmt(predict_l2)} | '
                          f'{fmt(random_l2)} | {fmt(zero_l2)} | {fmt(truthful_l2)} | {fmt(trivial_l2)}]')
                    print(f'interim loss: [Predict | Random | Zero | Truthful | Trivial] = [{fmt(predict_interim)} | '
                          f'{fmt(random_interim)} | {fmt(zero_interim)} | {fmt(truthful_interim)} | {fmt(trivial_interim)}]')
                    # plot strategy
                    plot_strategy(symmetric_value_dists[0, 0].cpu().numpy(), y[0, 0].cpu().numpy(), path=model_name,
                                  filename=f'{mechanism_name}_sym_player_num={player_num}_bid_strategy')
                    # plot_strategy(symmetric_value_dists[0].cpu().numpy(), target[0].cpu().numpy(), player_num, path=model_name,
                    #               filename=f'{mechanism_name}_sym_player_num={player_num}_target_strategy')
            # # print average results
            # print('---------------------------------------------------------------')
            # predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = [sum(eps_list) / len(eps_list) for eps_list
            #                                                                 in test_eps]
            # print(f'Average Results:')
            # print('Eps: [Predict | Random | Zero | Truthful | Trivial] = [%.5f | %.5f | %.5f | %.5f | %.5f]' % (
            #     predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))

    elif args.test_mode == 'symmetric_time':
        # test on symmetric games (only support first/second price with no entry)
        enc = {'first': 0, 'second': 1, }
        for mechanism_name in args.mechanisms:
            mechanism = enc[mechanism_name]
            print('---------------------------------------------------------------')
            print(f"TEST ON SYMMETRIC {mechanism_name.upper()} PRICE GAMES")
            input_mechanism = torch.tensor([mechanism]).to(device)
            input_entry = torch.zeros_like(input_mechanism).to(device)
            for player_num in range(2, args.max_player + 1):
                player_max_value = [args.valuation_range - 1] * player_num + [0] * (args.max_player - player_num)
                symmetric_value_dists = torch.from_numpy(
                    make_uniform_value_hist(player_max_value, args.valuation_range, args.max_player))
                input_player_num = torch.tensor([player_num]).to(device)
                symmetric_value_dists[player_num:] = 0
                symmetric_value_dists = symmetric_value_dists.unsqueeze(0).float().to(device)  # 1*N*V
                with torch.no_grad():
                    # predicted policy
                    if args.requires_value:
                        # generate value realizations
                        player_values = generate_values(args, device)

                        t = benchmark.Timer(
                            stmt='transpose_y0(args, model((input_mechanism, input_entry, symmetric_value_dists, player_values)))',
                            setup='from train import transpose_y0',
                            globals={
                                'args': args,
                                'model': model,
                                'input_mechanism': input_mechanism,
                                'input_entry': input_entry,
                                'symmetric_value_dists': symmetric_value_dists,
                                'player_values': player_values,
                            }
                        )
                        print('Using utils.benchmark.Timer')
                        print(t.timeit(10))

    elif args.test_mode == 'symmetric_number':
        # test on symmetric games (only support first/second price with no entry)
        enc = {'first': 0, 'second': 1, }
        for mechanism_name in args.mechanisms:
            mechanism = enc[mechanism_name]
            print('---------------------------------------------------------------')
            print(f"TEST ON SYMMETRIC {mechanism_name.upper()} PRICE GAMES")
            input_mechanism = torch.tensor([mechanism]).to(device)
            input_entry = torch.zeros_like(input_mechanism).to(device)
            if type(args.number_range[0]) is str:
                args.number_range = [int(x) for x in args.number_range]
            for player_num in range(*args.number_range):
                cur_max_player = max(player_num, args.max_player)
                print('current player:', player_num)

                upper_range = 20 + 1
                value_hist = torch.zeros(args.valuation_range)
                value_hist[:upper_range] = 1/upper_range
                symmetric_value_dists = value_hist.view(1,1,args.valuation_range).repeat(1, cur_max_player, 1)
                symmetric_value_dists[0, player_num:] = 0
                symmetric_value_dists = symmetric_value_dists.float().to(device)  # 1*N*V

                player_values = torch.arange(args.valuation_range).view(-1, 1).repeat(1, cur_max_player).flatten()  # (V*N)
                player_values = player_values.unsqueeze(-1) == torch.arange(args.valuation_range)  # (V*N) * V
                player_values = player_values.unsqueeze(0).float().to(device)  # 1*VN*2V

                # print(symmetric_value_dists[0,:,-1])
                # print(player_values.shape)

                with torch.no_grad():
                    # predicted policy
                    if args.requires_value:
                        y = transpose_y0(args, model((input_mechanism, input_entry, symmetric_value_dists, player_values)))
                        # sample bid to calculate v/bid
                        y = y[0, :player_num]  # n*V*V
                        bids = torch.distributions.Categorical(probs=y).sample(torch.Size([100]))  # k*n*V
                        bids = bids.float().transpose(0, 2).transpose(0, 1)  # k*n*V -> V*n*k -> n*V*k
                        bid_per_v = bids.mean(dim=-1)  # n*V
                        # import matplotlib.pyplot as plt
                        # plt.plot(torch.arange(args.valuation_range), bid_per_v.mean(dim=0).cpu())
                        # plt.savefig(f'{player_num}.png')
                        # plt.close()
                        # v/bid
                        division_values = torch.arange(args.valuation_range).unsqueeze(0).repeat(player_num,1).to(device)  # n*V
                        division_values[:, 0] = 1  # v=0 --> v=1 (for division)
                        n_ratio = bid_per_v / division_values
                        print((player_num-1)/player_num, n_ratio.mean())

    elif args.test_mode == 'example':
        # fp, G[0,10], G[0,5], G[0,5]
        player_num = 3
        upper_values = [20, 10, 10]
        player_value_hist = []
        for player_upper_value in upper_values:
            player_lower_value, player_val_size = 0, player_upper_value
            player_possible_values = np.arange(player_lower_value, player_upper_value + 1)
            mean, size = (player_lower_value + player_upper_value) / 2, player_val_size
            transform = lambda x: 6 / size * (x - mean)
            tmp = norm.cdf(transform(player_possible_values + 0.5)) - norm.cdf(
                transform(player_possible_values - 0.5))
            tmp[0] = norm.cdf(transform(player_lower_value + 0.5))
            tmp[-1] = 1 - norm.cdf(transform(player_upper_value - 0.5))
            gaussian_value_hist = np.zeros(args.valuation_range)
            gaussian_value_hist[player_lower_value:player_upper_value + 1] = tmp
            player_value_hist.append(gaussian_value_hist)
        print("Generated value hist:\n", player_value_hist)
        value_dists = np.array(player_value_hist + [np.zeros(args.valuation_range)] * (args.max_player - player_num))
        input_value_dists = torch.from_numpy(value_dists).unsqueeze(0).float().to(device)
        input_mechanism, input_entry = torch.tensor([0]).to(device), torch.tensor([0]).to(device)
        input_player_num = torch.tensor([player_num]).to(device)
        player_values = generate_values(args, device)
        with torch.no_grad():
            start = time.time()
            if args.requires_value:
                y0 = model((input_mechanism, input_entry, input_value_dists, player_values))
                y = transpose_y0(args, y0)
            else:
                y = model((input_mechanism, input_entry, input_value_dists))
            end = time.time()
            print('using time:', end - start)
            # calculate max exploit
            exploits, _, _ = calculate_exploit(input_mechanism, input_entry, input_player_num,
                                               input_value_dists,
                                               y, device, overbid_punish=False)
            max_exploits, _ = exploits.max(dim=-1)
            print('Max Exploit:', max_exploits[0].cpu())
            # dump strategy
            print('saving strategy of p0')
            np.save(os.path.join('results', 'strategy'), y[0, 0].cpu().numpy())
            # plot strategy
            for pid in range(player_num):
                plot_strategy(input_value_dists[0, pid].cpu().numpy(), y[0, pid].cpu().numpy(), path=model_name,
                              filename=f'paper_player_id={pid}_bid_strategy')


