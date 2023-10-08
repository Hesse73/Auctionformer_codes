import torch
import wandb
import os
from tqdm import tqdm
from utils import get_model_path, init_wandb, generate_test_strategy
from loss import calculate_exploit
from itertools import cycle


def transpose_y0(args, y0):
    if args.softmax_out:
        y = y0.softmax(dim=-1)  # softmax over bids
    else:
        y = torch.abs(y0)  # abs
        y = y / torch.sum(y, dim=-1, keepdim=True)  # normalize

    return y

def evaluate_on_dataloader(args, model, dataloader, device):
    model.eval()
    test_eps = [0, 0, 0, 0, 0]
    with torch.no_grad():
        for X in dataloader:
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
    predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = (eps / len(dataloader.dataset) for eps in
                                                                    test_eps)
    return predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps


def default_train(args, model, target_train, target_test, old_test, optim, scheduler, device):
    # prepare for model saving
    if not os.path.exists('./models'):
        os.mkdir('./models')
    print('model will be saved at:', get_model_path(args))

    if args.use_wandb:
        init_wandb(args)
    CE_flat_loss = torch.nn.CrossEntropyLoss(reduction='none')

    # print eps before fine-tuning
    model.eval()
    with torch.no_grad():
        train_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = [eps_val/20 for eps_val in
                                                                              evaluate_on_dataloader(args, model,
                                                                                                     target_train,
                                                                                                     device)]
        print(
            'TARGET TRAIN EPS: [Predict | Random | Zero | Truthful | Trivial] = [%.2e | %.2e | %.2e | %.2e | %.2e]' % (
                train_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))
        test_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = [eps_val/20 for eps_val in
                                                                             evaluate_on_dataloader(args, model,
                                                                                                    target_test,
                                                                                                    device)]
        print(
            'TARGET TEST EPS: [Predict | Random | Zero | Truthful | Trivial] = [%.2e | %.2e | %.2e | %.2e | %.2e]' % (
                test_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))
        old_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = [eps_val/20 for eps_val in
                                                                            evaluate_on_dataloader(args, model,
                                                                                                   old_test,
                                                                                                   device)]
        print(
            'OLD TEST EPS: [Predict | Random | Zero | Truthful | Trivial] = [%.2e | %.2e | %.2e | %.2e | %.2e]' % (
                old_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))

    best_test_eps, best_old_eps = test_predict_eps, old_predict_eps
    if args.use_wandb:
        wandb.log({'Target Eps': test_predict_eps, 'Old Eps': old_predict_eps,})

    # train
    for epoch in range(1, args.max_epoch + 1):
        print('-------------------------------------------------------')
        model.train()
        train_loss = 0.0
        train_bar = tqdm(enumerate(target_train))
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
            B,N,V = value_dists.shape
            valid_mask = cur_player_num.view(-1, 1) > torch.arange(N).to(device)  # B*N
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
                player_mask = max_ids.view(-1, 1) == torch.arange(N).to(device)  # mask:B*N=>B
                update_mask = update_mask_bn[player_mask]  # B
                loss = torch.sum(update_mask * max_exploits) / torch.sum(update_mask)
            elif args.update_random:
                update_ids = torch.distributions.Categorical(probs=update_mask_bn.float()).sample()  # B*N--sample-->B
                player_mask = update_ids.view(-1, 1) == torch.arange(N).to(device)  # mask:B*N=>B
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
        if epoch % args.eval_freq == 0 or epoch == args.max_epoch:
            model.eval()
            with torch.no_grad():
                test_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = [eps_val/20 for eps_val in
                                                                             evaluate_on_dataloader(args, model,
                                                                                                    target_test,
                                                                                                    device)]
                print(
                    'Eps on TARGET test data: [Predict | Random | Zero | Truthful | Trivial] = [%.2e | %.2e | %.2e | %.2e | %.2e]' % (
                        test_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))
                # test on full dataloader
                old_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = [eps_val/20 for eps_val in
                                                                             evaluate_on_dataloader(args, model,
                                                                                                    old_test,
                                                                                                    device)]
                print(
                    'Eps on OLD test data: [Predict | Random | Zero | Truthful | Trivial] = [%.2e | %.2e | %.2e | %.2e | %.2e]' % (
                        old_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))

            if args.use_wandb:
                wandb.log({'Target Eps': test_predict_eps, 'Old Eps': old_predict_eps, })

            if args.save_model and test_predict_eps < best_test_eps and old_predict_eps < best_old_eps:
                best_test_eps = test_predict_eps
                best_old_eps = old_predict_eps
                # save best model
                print(f'current best eps={best_test_eps}, old eps={best_old_eps}, saving model...')
                torch.save(model.state_dict(), get_model_path(args))

    if args.use_wandb:
        wandb.finish()



def dream_booth_train(args, target_model, base_model, target_train, target_test, old_train, old_test, optim, scheduler, device):
    # prepare for model saving
    if not os.path.exists('./models'):
        os.mkdir('./models')
    print('model will be saved at:', get_model_path(args))

    if args.use_wandb:
        init_wandb(args)

    # print eps before fine-tuning
    target_model.eval()
    with torch.no_grad():
        train_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = [eps_val/20 for eps_val in
                                                                              evaluate_on_dataloader(args, target_model,
                                                                                                     target_train,
                                                                                                     device)]
        print(
            'TARGET TRAIN EPS: [Predict | Random | Zero | Truthful | Trivial] = [%.2e | %.2e | %.2e | %.2e | %.2e]' % (
                train_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))
        target_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = [eps_val/20 for eps_val in
                                                                              evaluate_on_dataloader(args, target_model,
                                                                                                     target_test,
                                                                                                     device)]
        print(
            'TARGET TEST EPS: [Predict | Random | Zero | Truthful | Trivial] = [%.2e | %.2e | %.2e | %.2e | %.2e]' % (
                target_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))
        train_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = [eps_val/20 for eps_val in
                                                                              evaluate_on_dataloader(args, target_model,
                                                                                                     old_train,
                                                                                                     device)]
        print(
            'OLD TRAIN EPS: [Predict | Random | Zero | Truthful | Trivial] = [%.2e | %.2e | %.2e | %.2e | %.2e]' % (
                train_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))
        old_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = [eps_val/20 for eps_val in
                                                                              evaluate_on_dataloader(args, target_model,
                                                                                                     old_test,
                                                                                                     device)]
        print(
            'OLD TEST EPS: [Predict | Random | Zero | Truthful | Trivial] = [%.2e | %.2e | %.2e | %.2e | %.2e]' % (
                old_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))

    best_target_eps, best_old_eps = target_predict_eps, old_predict_eps
    if args.use_wandb:
        wandb.log({'Target Eps': target_predict_eps, 'Old Eps': old_predict_eps,})

    # train
    dataset_size = max(len(target_train.dataset), len(old_train.dataset))
    if len(target_train) < len(old_train):
        target_train = cycle(target_train)
    else:
        old_train = cycle(old_train)
    for epoch in range(1, args.max_epoch + 1):
        print('-------------------------------------------------------')
        target_model.train()
        train_loss, target_train_eps = 0.0, 0.0
        train_bar = tqdm(enumerate(zip(target_train, old_train)))
        for b_id, X in train_bar:
            assert args.requires_value
            # B, B, B, B*N*V, B*NV, B*N*V*V
            target_data, old_data = X

            # target data, reconstruction loss
            cur_mechanism, cur_entry, cur_player_num, value_dists, player_values = (data.to(device) for data in target_data)
            value_dists, player_values = value_dists.float(), player_values.float()

            target_y_gt = transpose_y0(args, target_model((cur_mechanism, cur_entry, value_dists, player_values)))

            B,N,V = value_dists.shape
            valid_mask = cur_player_num.view(-1, 1) > torch.arange(N).to(device)  # B*N

            exploits, _, _ = calculate_exploit(cur_mechanism, cur_entry, cur_player_num,
                                                                         value_dists, target_y_gt, device,
                                                                         drop_mask_bn=None, overbid_punish=True,
                                                                         random_tie=args.random_tie,
                                                                         select_highest=args.select_highest,
                                                                         detach_market=args.detach_market)

            # loss = max exploit or sum exploit
            max_exploits, max_ids = exploits.max(dim=-1)  # B
            player_mask = max_ids.view(-1, 1) == torch.arange(N).to(device)  # mask:B*N=>B
            update_mask = valid_mask[player_mask]  # B
            reconstruction_loss = torch.sum(update_mask * max_exploits) / torch.sum(update_mask)
            # get train eps
            target_train_eps += max_exploits.detach().sum().cpu().item()

            # old data, preservation loss
            cur_mechanism, cur_entry, cur_player_num, value_dists, player_values = (data.to(device) for data in old_data)
            value_dists, player_values = value_dists.float(), player_values.float()

            target_y_old = transpose_y0(args, target_model((cur_mechanism, cur_entry, value_dists, player_values)))
            base_y_old = transpose_y0(args, base_model((cur_mechanism, cur_entry, value_dists, player_values)))

            B,N,V = value_dists.shape
            valid_mask = cur_player_num.view(-1, 1) > torch.arange(N).to(device)  # B*N
            valid_mask_bnvv = valid_mask.view(B,N,1,1).repeat(1,1,V,V)
            preservation_loss = torch.sum((target_y_old - base_y_old).abs() * valid_mask_bnvv) / torch.sum(valid_mask_bnvv)

            loss = reconstruction_loss + args.preserve_ratio * preservation_loss
            loss.backward()
            optim.step()
            optim.zero_grad()

            loss_value = loss.cpu().item()
            train_loss += loss_value
            avg_train_loss = train_loss / (b_id + 1)
            train_bar.set_description("Epoch %d Avg Loss %.5f" % (epoch, avg_train_loss))
            if args.use_wandb:
                wandb.log({'loss': loss_value, 'avg_loss': avg_train_loss})

        # print train eps
        print('eps on train data:', target_train_eps/dataset_size)

        if scheduler is not None:
            scheduler.step()

        if epoch % args.eval_freq == 0 or epoch == args.max_epoch:
            # test
            target_model.eval()
            with torch.no_grad():
                target_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = [eps_val/20 for eps_val in
                                                                              evaluate_on_dataloader(args, target_model,
                                                                                                     target_test,
                                                                                                     device)]
                print(
                    'Eps on TARGET test data: [Predict | Random | Zero | Truthful | Trivial] = [%.2e | %.2e | %.2e | %.2e | %.2e]' % (
                        target_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))
                # test on full dataloader
                old_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps = [eps_val/20 for eps_val in
                                                                              evaluate_on_dataloader(args, target_model,
                                                                                                     old_test,
                                                                                                     device)]
                print(
                    'Eps on OLD test data: [Predict | Random | Zero | Truthful | Trivial] = [%.2e | %.2e | %.2e | %.2e | %.2e]' % (
                        old_predict_eps, random_eps, zero_eps, truthful_eps, trivial_eps))

            if args.use_wandb:
                wandb.log({'Target Eps': target_predict_eps, 'Old Eps': old_predict_eps, })

            if args.save_model and target_predict_eps < best_target_eps and old_predict_eps < best_old_eps:
                best_target_eps = target_predict_eps
                best_old_eps = old_predict_eps
                # save best model
                print(f'current best target eps={best_target_eps}, best raw eps={best_old_eps}, saving model...')
                torch.save(target_model.state_dict(), get_model_path(args))

    if args.use_wandb:
        wandb.finish()

