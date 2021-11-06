import os
import numpy as np
import random


def discount(b, m):
    return 1 - np.exp(m / b - 1)


def online_weighted_greedy_step(B, M, W, n, kw_num):
    optimal_ad_num = -1
    optimal_bid = 0
    for i in range(n):
        # 0 means no bid.
        if W[i][kw_num] == 0:
            continue
        if W[i][kw_num] <= (B[i] - M[i]):
            if optimal_bid < discount(B[i], M[i]) * W[i][kw_num]:
                optimal_bid = W[i][kw_num]
                optimal_ad_num = i
    return optimal_ad_num, optimal_bid


def online_weighted_greedy(B, W, n, r, kw_nums):
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
    m = len(kw_nums)
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


if __name__ == "__main__":
    # When testing, substitute the variables n, m, W, B with appropriate values.
    x = 0