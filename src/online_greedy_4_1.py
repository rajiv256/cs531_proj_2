import os
import numpy as np
import random


# In the online setting, "m" value is not known. We do know "r" which is the no. of unique key words.

def online_greedy_step(B, M, W, n, kw_num):
    """

    Args:
        B:
        M:
        W:
        n:
        kw_num: keyword number.

    Returns:
        optimal_ad_num: which advertiser is mapped.
        optimal_bid: bid value of the matched advertiser
    """
    optimal_ad_num = -1
    optimal_bid = 0
    for i in range(n):
        # 0 means no bid.
        if W[i][kw_num] == 0:
            continue
        if W[i][kw_num] <= (B[i]-M[i]):
            if optimal_bid <= W[i][kw_num]:
                optimal_bid = W[i][kw_num]
                optimal_ad_num = i
    return optimal_ad_num, optimal_bid


def online_greedy(B, W, n, r, kw_nums):
    """
    Args:
        B: (n), budgets
        W: (n*r) i-th advertiser's bid for j-th keyword
        n: # advertisers
        r: # keywords
        kw_nums: (m) number of the keyword in each of the queries

    Returns:
        Q: (m) query assignment
        revenue: Revenue accrued
    """
    M = [0] * n
    m = len(kw_nums)
    revenue = 0
    Q = [-1]*m

    for t in range(len(kw_nums)):
        kw_num = kw_nums[t]
        ad_num, bid = online_greedy_step(B, M, W, n, kw_num)
        if ad_num == -1:
            continue
        M[ad_num] += bid
        revenue += bid
        Q[t] = ad_num
    return Q, revenue


if __name__ == "__main__":
    # When testing, substitute the variables n, m, W, B with appropriate values.
    x = 0