import torch


def calculate_exploit(mechanism, entry, player_num, valuation_dist, bidding_strategy, device,
                      drop_mask_bn=None, overbid_punish=True, random_tie=False,
                      select_highest=False, detach_market=False):
    """

        Args:
            mechanism: B
            entry: B
            player_num: B
            valuation_dist: B*N*V
            bidding_strategy: B*N*V*V
            device: torch device
            drop_mask_bn: drop some players' bid to rand-bid, a B*N mask
            overbid_punish: whether to use overbid punishment
            random_tie: random tie-breaking or zero tie-breaking
            select_highest: select highest bid as the opt bid or lowest
            detach_market: whether to detach market price's calculation

        Returns: player's exploit (B*N), opt strategy (B*N*V*V), dropped strategy (B*N*V*V)

    """
    B, N, V = valuation_dist.shape

    ####################################
    # 1. get market price distribution #
    ####################################
    # valid mask: B*N or B*N*V
    valid_player_mask_bn = player_num.view(-1, 1) > torch.arange(N).to(device)
    valid_player_mask_bnv = valid_player_mask_bn.unsqueeze(-1).repeat(1, 1, V)

    # drop strategy
    if drop_mask_bn is not None:
        dropped_strategy = torch.rand(bidding_strategy.shape).softmax(dim=-1).to(device)  # B*N*V*V
        drop_mask_bnvv = drop_mask_bn.view(B,N,1,1).repeat(1,1,V,V)
        dropped_strategy[~drop_mask_bnvv] = bidding_strategy[~drop_mask_bnvv]
    else:
        drop_mask_bn = torch.zeros_like(valid_player_mask_bn)
        dropped_strategy = bidding_strategy

    # get each player's marginal bid (B*N*1*V @ B*N*V*V => B*N*1*V) --> B*N*V
    marginal_bid = torch.matmul(valuation_dist.unsqueeze(-2), dropped_strategy).squeeze(-2)
    cumulative_bid = torch.cumsum(marginal_bid, dim=-1)  # B*N*V
    # set invalid player with zero bid (i.e. cum_bid = [1,1,1,1,...])
    cumulative_bid[~valid_player_mask_bnv] = 1

    # get each player's other players' bid
    # B*N*V --> B*1*N*V --> repeat to B*N*N*V
    others_cum_bid = cumulative_bid.unsqueeze(1).repeat(1, N, 1, 1)
    # set self with zero bid (i.e. cum_bid = [1,1,1,1,...])
    self_mask = (torch.arange(N).view(-1, 1) == torch.arange(N)).view(1, N, N, 1).repeat(B, 1, 1, V).to(
        device)  # B*N*N*V
    others_cum_bid[self_mask] = 1

    # market price cdf for each bidder, prod on other player's dim (B*N*N*V -- > B*N*V)
    market_cdf = torch.prod(others_cum_bid, dim=-2)

    # market price pdf (B*N*V)
    tmp = torch.zeros_like(market_cdf)
    tmp[:, :, 1:] = market_cdf[:, :, :-1]
    market_pdf = market_cdf - tmp

    # check precision
    if market_pdf.min() < -1e-5:
        raise ValueError('Calculated pdf has negative values < -1e-5!')
    else:
        # market_pdf = torch.abs(market_pdf)
        market_pdf[market_pdf < 0] = 0

    ###############################
    # 2. calculate utility matrix #
    ###############################
    # utility matrix V*V*V (given different value|market|bid)
    value = torch.arange(V).view(-1, 1, 1).repeat(1, V, V).float().to(device)
    market = torch.arange(V).view(1, -1, 1).repeat(V, 1, V).float().to(device)
    bid = torch.arange(V).view(1, 1, -1).repeat(V, V, 1).float().to(device)

    fp_utility_v_m_b = (value - bid) * (bid >= market)
    sp_utility_v_m_b = (value - market) * (bid >= market)

    # batched utility (B*V*V*V)
    fp_utility = fp_utility_v_m_b.repeat(B, 1, 1, 1)
    sp_utility = sp_utility_v_m_b.repeat(B, 1, 1, 1)

    # random tie-breaking
    tie_breaking_mask = (bid == market).repeat(B, 1, 1, 1)
    if random_tie:
        # valid tie-breaking num range: [1,n-1] for each game (B*N)
        valid_tie_num = player_num.view(-1, 1) > torch.arange(N).to(device)  # (B*N)
        valid_tie_num[:, 0] = 0
        # random pick a market num (B*V*V)
        random_tie_num = torch.distributions.Categorical(probs=valid_tie_num).sample(torch.Size([V,V])).T
        # including itself
        random_tie_num += 1

        # random tie-breaking
        fp_utility[tie_breaking_mask] /= random_tie_num.flatten()
        sp_utility[tie_breaking_mask] /= random_tie_num.flatten()
    else:
        # zero tie-breaking
        fp_utility[tie_breaking_mask] = 0
        sp_utility[tie_breaking_mask] = 0


    # mechanism
    is_first = (mechanism % 2 == 0).view(-1, 1, 1, 1).repeat(1, V, V, V)

    # entrance fee:
    entries = entry.view(-1, 1, 1, 1).repeat(1, V, V, V)
    entries[:, :, :, 0] = 0  # bid=0's entry is 0

    # batched reward matrix, given different mechanism & entry (B*V*V*V)
    utility_v_m_b = (fp_utility * is_first + sp_utility * (~is_first)) - entries

    # overbid punishment
    if overbid_punish:
        # donot apply for SP+Entry
        is_overbid = (bid + entries > value) & ~((entries > 0) & (~is_first))
        utility_v_m_b[is_overbid] = -10
    #################################
    # 3. calculate expected utility #
    #################################
    if detach_market:
        market_pdf = market_pdf.detach()
    # expectation on each player's market price (each player's expected utility under different value & bid)
    utility_v_b = torch.matmul(market_pdf.view(B, N, 1, 1, V), utility_v_m_b.view(B, 1, V, V, V))  # B*N*V*1*V
    utility_v_b = utility_v_b.squeeze(-2)  # B*N*V*V
    # # check matmul results
    # torch.set_printoptions(profile='full')
    # # check 1
    # market_pdf_1 = market_pdf.view(B,N,1,1,V).repeat(1,1,V,1,1)  # B*N*V*1*V
    # utility_v_m_b_1 = utility_v_m_b.view(B,1,V,V,V).repeat(1,N,1,1,1)  # B*N*V*V*V
    # utility_v_b_1 = torch.matmul(market_pdf_1, utility_v_m_b_1)  # B*N*V*[1*V] @ B*N*V*[V*V] = B*N*V*[1*V]
    # utility_v_b_1 = utility_v_b_1.squeeze(-2)  # B*N*V*V
    # assert (utility_v_b_1.detach() == utility_v_b.detach()).min() == True
    # print('pass check 1')
    #
    # # check 2
    # utility_v_b_2 = torch.zeros_like(bidding_strategy)  # B*N*V*V
    # for i in range(N):
    #     for j in range(V):
    #         # player {i}'s utility when value = {j}
    #         # market B*1*V_m @ utility_matrix B*V_m*V_b
    #         utility_i_j = torch.matmul(market_pdf[:,i,:].unsqueeze(-2), utility_v_m_b[:,j,:,:])  # B*1*V_b
    #         utility_v_b_2[:, i, j, :] = utility_i_j.squeeze(-2)  # B*V_b
    # assert (utility_v_b_2.detach() == utility_v_b.detach()).min() == True
    # print('pass check 2')

    # calculate current policy's expected utility on each player's value (B*N*V)
    utility_v = (dropped_strategy * utility_v_b).sum(dim=-1)

    # each player's optimal utility under different value (B*N*V)
    if select_highest:
        # get the highest bid's utility (which is different in torch.grad)
        opt_bids_v = utility_v_b == utility_v_b.max(dim=-1, keepdims=True)[0]  # B*N*V*V
        opt_bid_v = opt_bids_v.cumsum(dim=-1).argmax(dim=-1)  # select the highest opt bid, B*N*V
        opt_bid_idx = opt_bid_v.unsqueeze(-1) == torch.arange(V).to(device)  # B*N*V*1 == V --> B*N*V*V
        opt_utility_v = utility_v_b[opt_bid_idx].view(B,N,V)
    else:
        opt_utility_v, opt_bid_v = utility_v_b.max(dim=-1)

    # expectation on each [valid] player's valuation distribution (B*N)
    utility = (~drop_mask_bn) * valid_player_mask_bn * (valuation_dist * utility_v).sum(dim=-1)
    opt_utility = (~drop_mask_bn) * valid_player_mask_bn * (valuation_dist * opt_utility_v).sum(dim=-1)

    expl = opt_utility - utility

    opt_strategy = torch.nn.functional.one_hot(opt_bid_v, num_classes=V).float()  # B*N*V*V
    # l1_strategy = torch.abs(opt_strategy - dropped_strategy)

    return expl, opt_strategy, dropped_strategy

