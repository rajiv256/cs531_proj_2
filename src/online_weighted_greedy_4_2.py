import random

import numpy as np

from src.data_utils import create_data_vars


def discount(B_i, M_i):
    return 1 - np.exp((M_i / (B_i+1e-9)) - 1)


def online_weighted_greedy_step(B, M, W, n, kw_num):
    optimal_ad_num = -1
    disc_bid = 0
    optimal_bid = 0

    for i in range(n):
        # 0 means no bid.
        # # print(f'i: {i} | kw_num: {kw_num}| discount: {discount(B[i], M[i])} | wij: {W[i][kw_num]} | bid: {discount(B[i], M[i])*W[i][kw_num]}')
        if W[i][kw_num] == 0:
            # print(f'ad: {i} | bid: 0 | skipping')
            continue
        if W[i][kw_num] <= (B[i] - M[i]):
            disc = discount(B[i], M[i])
            # print(f'ad: {i} | disc: {disc} | bid: {W[i][kw_num]} | val: {W[i][kw_num] * disc}')
            if disc_bid < disc * W[i][kw_num]:
                disc_bid = disc * W[i][kw_num]
                optimal_bid = W[i][kw_num]
                optimal_ad_num = i
    # print(f'selected ad_num: {optimal_ad_num}')
    # print('================STEP OVER=======================')
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
        # print(f'iter: {t} | B: {B} | M: {M}')
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
    data = create_data_vars(data_alias)
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


def get_results_avg(data_alias='ds1'):
    data = create_data_vars(data_alias)
    print(data)
    n = data['n']
    m = data['m']
    W = data['W']
    B = data['B']
    kw_nums = data['kw_nums']
    r = data['r']
    revenues = []
    for i in range(100):
        random.shuffle(kw_nums)
        Q, revenue = online_weighted_greedy(B, W, n, r, m, kw_nums)
        revenues.append(revenue)
    print(np.mean(revenues), np.std(revenues))

if __name__ == "__main__":
    # When testing, substitute the variables n, m, W, B with appropriate values.

    print(get_results('ds3')['revenue'])
    # print(get_results_avg('ds1_mini'))