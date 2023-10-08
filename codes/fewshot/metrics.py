import torch


def ex_interim_utility_loss(mechanism: torch.Tensor,
                            entry: torch.Tensor,
                            player_num: torch.Tensor,
                            valuation_dist: torch.Tensor,
                            bidding_strategy: torch.Tensor,
                            device: torch.device):
    r"""
        Calculate max interim utility loss :math:`\ell_max`

        .. math::
            \max_{v_i \in V_i} \max_{b_i^* \in A_i} E_{v_{-i}|v_i} [u(v_i, b_i^*, b_{-i}(v_{-i})) - u(v_i, b_i, b_{-i}(v_{-i}))]

        Args:
            mechanism: B
            entry: B
            player_num: B
            valuation_dist: B*N*V
            bidding_strategy: B*N*V*V
            device: torch device

        Returns:
            :math:`\ell_max` mean, std of each bidder
    """
    B, N, V = valuation_dist.shape
    assert B == 1

    # 1. get market price distribution
    # valid mask: B*N or B*N*V
    valid_player_mask_bn = player_num.view(-1, 1) > torch.arange(N).to(device)
    valid_player_mask_bnv = valid_player_mask_bn.unsqueeze(-1).repeat(1, 1, V)

    # get each player's marginal bid (B*N*1*V @ B*N*V*V => B*N*1*V) --> B*N*V
    marginal_bid = torch.matmul(valuation_dist.unsqueeze(-2), bidding_strategy).squeeze(-2)
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

    # 2. calculate utility matrix
    # utility matrix V*V*V (given different value|market|bid)
    value = torch.arange(V).view(-1, 1, 1).repeat(1, V, V).float().to(device)
    market = torch.arange(V).view(1, -1, 1).repeat(V, 1, V).float().to(device)
    bid = torch.arange(V).view(1, 1, -1).repeat(V, V, 1).float().to(device)

    fp_utility_v_m_b = (value - bid) * (bid > market)
    sp_utility_v_m_b = (value - market) * (bid > market)

    # batched utility (B*V*V*V)
    fp_utility = fp_utility_v_m_b.repeat(B, 1, 1, 1)
    sp_utility = sp_utility_v_m_b.repeat(B, 1, 1, 1)

    # mechanism
    is_first = (mechanism % 2 == 0).view(-1, 1, 1, 1).repeat(1, V, V, V)

    # entrance fee:
    entries = entry.view(-1, 1, 1, 1).repeat(1, V, V, V)
    entries[:, :, :, 0] = 0  # bid=0's entry is 0

    # batched reward matrix, given different mechanism & entry (B*V*V*V)
    utility_v_m_b = (fp_utility * is_first + sp_utility * (~is_first)) - entries

    # 3. calculate expected utility
    # expectation on each player's market price (each player's expected utility under different value & bid)
    utility_v_b = torch.matmul(market_pdf.view(B, N, 1, 1, V), utility_v_m_b.view(B, 1, V, V, V))  # B*N*V*1*V
    utility_v_b = utility_v_b.squeeze(-2)  # B*N*V*V

    # calculate current policy's expected utility on each player's value (B*N*V)
    utility_v = (bidding_strategy * utility_v_b).sum(dim=-1)
    # each player's optimal utility under different value (B*N*V)
    opt_utility_v, opt_bid_v = utility_v_b.max(dim=-1)

    # utility loss max_v{opt_b{(utility* - utility)}}, B*N
    valuation_mask = valuation_dist > 0  # B*N*V
    utility_loss_v = valid_player_mask_bnv * valuation_mask * (opt_utility_v - utility_v)  # B*N*V
    utility_loss, _ = utility_loss_v.max(dim=-1)  # B*N

    utility_loss = utility_loss[0][:player_num[0]]  # n

    return torch.mean(utility_loss), torch.std(utility_loss)


def relative_utility_loss(mechanism: torch.Tensor,
                          entry: torch.Tensor,
                          player_num: torch.Tensor,
                          valuation_dist: torch.Tensor,
                          bidding_strategy: torch.Tensor,
                          bne_strategy: torch.Tensor,
                          device: torch.device):
    r"""
        Calculate relative utility loss :math:`L`

        .. math::
            1 - u(i,b_i^*,b_{-i})/u(i,b_i,b_{-i})

        Args:
            mechanism: 0: first, 1: second
            entry: entrance fee
            player_num: number of players (n)
            valuation_dist: n*V
            bidding_strategy: n*V*V
            bne_strategy: n*V*V
            device: torch device

        Returns:
            :math:`L` mean, std of each bidder
    """
    pass


def sample_l2_distance(value_hist: torch.Tensor,
                       y: torch.Tensor,
                       analytic: torch.Tensor,
                       k=100):
    r"""
        Calculate L2 distance :math:`L_2` by sampling mixed strategy:

        .. math::
            1/n\sum_{i<n}\sqrt{\sum_{v_i<V}p_i(v_i)*(b_i^*(v_i) - b_i(v_i))^2}

        Args:
           value_hist: value distribution histogram: n*V
           y: model predicted strategy: n*V*V
           analytic: analytic strategy: n*V
           k: sample num, default 10

        Returns:
            :math:`L_2`
    """

    predict = torch.distributions.Categorical(y).sample(torch.Size([k]))  # k*n*V
    predict = predict.float().transpose(0, 2).transpose(0, 1)  # k*n*V -> V*n*k -> n*V*k
    analytic = analytic.unsqueeze(-1).repeat(1, 1, k)  # n*V*k

    square_dist = ((predict - analytic) ** 2).mean(dim=-1)  # n*V
    l2_dist = (value_hist * square_dist).mean(dim=-1)  # n

    return torch.mean(l2_dist), torch.std(l2_dist)
