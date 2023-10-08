import torch
import wandb
import os
from tqdm import tqdm
from utils import get_model_path, init_wandb, generate_test_strategy
from loss import calculate_exploit


def transpose_y0(args, y0):
    if args.softmax_out:
        y = y0.softmax(dim=-1)  # softmax over bids
    else:
        y = torch.abs(y0)  # abs
        y = y / torch.sum(y, dim=-1, keepdim=True)  # normalize

    return y


def default_train(args, model, train_loader, test_loader, optim, scheduler, device):
    # prepare for model saving
    if not os.path.exists('./models'):
        os.mkdir('./models')
    print('model will be saved at:', get_model_path(args))

    if args.use_wandb:
        init_wandb(args)
    best_eps = 1000000
    CE_flat_loss = torch.nn.CrossEntropyLoss(reduction='none')

    for epoch in range(1, args.max_epoch + 1):
        print('-------------------------------------------------------')
        model.train()
        train_loss = 0.0
        train_bar = tqdm(enumerate(train_loader))
        for b_id, X in train_bar:
            if args.requires_value:
                # B, B, B, B*N*V, B*NV
                cur_mechanism, cur_entry, cur_player_num, value_dists, player_values = (data.to(device) for data in X)
                value_dists, player_values = value_dists.float(), player_values.float()

                y0 = model((cur_mechanism, cur_entry, value_dists, player_values))  # B*N*V*V

                y = transpose_y0(args, y0)
            else:
                # B, B, B, B*N*V
                cur_mechanism, cur_entry, cur_player_num, value_dists = (data.to(device) for data in X)
                value_dists = value_dists.float()
                y = model((cur_mechanism, cur_entry, value_dists))  # B*N*V*V

            # valid player mask & dro-pout mask
            valid_mask = cur_player_num.view(-1, 1) > torch.arange(args.max_player).to(device)  # B*N
            if args.add_drop:
                drop_mask = (torch.rand(value_dists.shape[:2]) < 0.05).to(device)
                drop_mask = drop_mask * valid_mask
            else:
                drop_mask = torch.zeros_like(valid_mask)

            # player's exploits: B*N
            exploits, opt_strategy, dropped_strategy = calculate_exploit(cur_mechanism, cur_entry, cur_player_num,
                                                                         value_dists, y, device,
                                                                         drop_mask_bn=drop_mask, overbid_punish=True,
                                                                         random_tie=args.random_tie,
                                                                         select_highest=args.select_highest,
                                                                         detach_market=args.detach_market)

            # loss = max exploit or sum exploit
            update_mask_bn = (~drop_mask) * valid_mask
            if args.only_update_max:
                max_exploits, max_ids = exploits.max(dim=-1)  # B
                player_mask = max_ids.view(-1, 1) == torch.arange(args.max_player).to(device)  # mask:B*N=>B
                update_mask = update_mask_bn[player_mask]  # B
                loss = torch.sum(update_mask * max_exploits) / torch.sum(update_mask)
            elif args.update_random:
                update_ids = torch.distributions.Categorical(probs=update_mask_bn.float()).sample()  # B*N--sample-->B
                player_mask = update_ids.view(-1, 1) == torch.arange(args.max_player).to(device)  # mask:B*N=>B
                rand_exploits = exploits[player_mask]  # B
                update_mask = update_mask_bn[player_mask]  # B
                loss = torch.sum(update_mask * rand_exploits) / torch.sum(update_mask)
            else:
                player_mask = torch.ones_like(update_mask_bn)  # B*N
                loss = torch.sum(update_mask_bn * exploits) / torch.sum(update_mask_bn)

            if args.add_policy_loss:
                if args.policy_loss == 'ce':
                    # CE loss's class is on dim-1, so transpose to B*V_b*V*N, then loss across bid classes: B*V*N
                    ce_loss = CE_flat_loss(dropped_strategy.transpose(1,3), opt_strategy.transpose(1,3)).transpose(1,2)
                    loss += args.gamma * torch.sum(ce_loss.sum(dim=-1) * update_mask_bn * player_mask) / torch.sum(
                        update_mask_bn * player_mask)

                elif args.policy_loss == 'l1':
                    # l1_strategy: B*N*V*V --sum over last 2 dim-> B*N
                    l1_per_bidder = torch.abs(opt_strategy - dropped_strategy).sum(dim=[-1, -2])
                    loss += args.gamma * 0.5 * torch.sum(l1_per_bidder * update_mask_bn * player_mask) / torch.sum(
                        update_mask_bn * player_mask)

            # y.register_hook(lambda grad: print('y:',grad))
            loss.backward()
            optim.step()
            optim.zero_grad()

            # show results
            loss_value = loss.cpu().item()
            train_loss += loss_value
            avg_train_loss = train_loss / (b_id + 1)
            train_bar.set_description("Epoch %d Avg Loss %.5f" % (epoch, avg_train_loss))
            if args.use_wandb:
                wandb.log({'loss': loss_value, 'avg_loss': avg_train_loss})
        if scheduler is not None:
            scheduler.step()

        if args.log_results:
            # # print || y - truthful ||
            # B, N, V = y.shape[:3]
            # truthful_y = torch.arange(V).view(-1, 1) == torch.arange(V)  # V*V
            # truthful_y = truthful_y.float().to(device).view(1, 1, V, V).repeat(B, N, 1, 1)
            # print('||y - truthful|| =',torch.abs(truthful_y - y).sum(dim=[1,2,3]).mean().cpu())
            # print last bidding
            torch.set_printoptions(precision=5, sci_mode=False)
            player_value_range = torch.cumsum(value_dists[0], dim=-1).argmax(dim=-1) + 1  # N
            player_id = player_value_range.argmax()  # id with max range
            print(
                f'mechanism:{cur_mechanism[0].cpu()}, entry:{cur_entry[0].cpu()}, player_num:{cur_player_num[0].cpu()}')
            print(f'player {player_id} value distribution:{value_dists[0][player_id].cpu()}')
            print(f'player {player_id} bidding strategy:{y[0][player_id][:player_value_range[player_id]].cpu()}')
            print(f'player {player_id} exploit:{exploits[0][player_id].cpu()}')
            torch.set_printoptions(profile='default')

        # test
        model.eval()
        test_eps = [0, 0, 0, 0, 0]
        with torch.no_grad():
            for X in test_loader:
                if args.requires_value:
                    # B, B, B, B*N*V, B*NV
                    cur_mechanism, cur_entry, cur_player_num, value_dists, player_values = (data.to(device) for data in X)
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
                    exploits, _, _ = calculate_exploit(cur_mechanism, cur_entry, cur_player_num, value_dists, strategy,
                                                    device, drop_mask_bn=None, overbid_punish=False, random_tie=False)
                    max_exploits, _ = exploits.max(dim=-1)
                    test_eps[idx] += max_exploits.sum().cpu().item()
        predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = (eps / len(test_loader.dataset) for eps in
                                                                        test_eps)
        print(
            'Eps on test data: [Predict | Random | Zero | Truthful | Trivial] = [%.5f | %.5f | %.5f | %.5f | %.5f]' % (
                predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))
        if args.save_model and predict_eps < best_eps:
            best_eps = predict_eps
            # save best model
            print('current best eps={}, saving model...'.format(best_eps))
            torch.save(model.state_dict(), get_model_path(args))

        if args.use_wandb:
            wandb.log({'Test Eps': predict_eps})

    if args.use_wandb:
        wandb.finish()
