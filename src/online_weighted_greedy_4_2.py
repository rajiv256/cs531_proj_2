import numpy as np

from data_utils import create_data_vars


def discount(B_i, M_i):
    return 1 - np.exp(M_i / (B_i + 1e-9) - 1)


def online_weighted_greedy_step(B, M, W, n, kw_num):
    optimal_ad_num = -1
    optimal_bid = 0

    for i in range(n):
        # 0 means no bid.
        # print(f'i: {i} | kw_num: {kw_num}| discount: {discount(B[i], M[i])} | wij: {W[i][kw_num]} | bid: {discount(B[i], M[i])*W[i][kw_num]}')
        if W[i][kw_num] == 0:
            continue
        if W[i][kw_num] <= (B[i] - M[i]):

            if optimal_bid < discount(B[i], M[i]) * W[i][kw_num]:
                optimal_bid = W[i][kw_num]
                optimal_ad_num = i
    # print(f'selected ad_num: {optimal_ad_num}')
    return optimal_ad_num, optimal_bid


def online_weighted_greedy(B, W, n, r, m, kw_nums):
    """

    Args:
        B:
        W:
        n:
        r:
        kw_nums:

    Returns:

    """
    M = [0] * n
    revenue = 0
    Q = [-1] * m

    for t in range(m):
        kw_num = kw_nums[t]
        ad_num, bid = online_weighted_greedy_step(B, M, W, n, kw_num)
        if ad_num == -1:
            continue
        M[ad_num] += bid
        revenue += bid
        Q[t] = ad_num
    return Q, revenue


def get_results(data_alias='ds0'):
    """
    This function makes it easy to get numbers of slides. This is a common function in all problem files.
    Args:
        data_alias:

    Returns:
        results: A dictionary with keys as query assignments `Q` and revenue `revenue`.
    """
    data = create_data_vars('ds0')
    n = data['n']
    m = data['m']
    W = data['W']
    B = data['B']
    kw_nums = data['kw_nums']
    r = data['r']
    Q, revenue = online_weighted_greedy(B, W, n, r, m, kw_nums)
    results = {
        'Q': Q,
        'revenue': revenue
    }
    return results


if __name__ == "__main__":
    # When testing, substitute the variables n, m, W, B with appropriate values.
    print(get_results('ds0'))
    print(get_results('ds0')['revenue'])
