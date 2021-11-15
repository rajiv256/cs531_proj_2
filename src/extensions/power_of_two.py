#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random

import numpy as np

from src.data_utils import create_data_vars


def weighted_rand_ad(weights, n, treshold=0.90, slope=300):
    leader_bid = max(weights)
    if leader_bid == 0:
        return -1
    altered_weights = [0] * n
    for i in range(len(weights)):
        w = weights[i] / leader_bid
        altered_weights[i] = 1 / (1 + np.exp(-slope * (w - treshold)))
    return altered_weights


def online_rand_power_step(B, M, W, n, kw_num):
    optimal_ad_num = -1
    disc_bid = 0
    optimal_bid = 0
    chosen_bid = 0

    advertisers = [*range(n)]
    weights = [0] * n

    for i in range(n):
        # 0 means no bid.
        # print(f'i: {i} | kw_num: {kw_num}| discount: {discount(B[i], M[i])} | wij: {W[i][kw_num]} | bid: {discount(B[i], M[i])*W[i][kw_num]}')
        if W[i][kw_num] == 0:
            # print(f'ad: {i} | bid: 0 | skipping')
            continue
        if W[i][kw_num] <= (B[i] - M[i]):
            weights[i] = W[i][kw_num]

        # print(f'ad: {i} | disc: {disc} | bid: {W[i][kw_num]} | val: {W[i][kw_num] * disc}')

    altered_weights = weighted_rand_ad(weights, n)
    if altered_weights == -1:
        return (-1, -1)
    first_ad_num = random.choices(advertisers, altered_weights, k=1)[0]
    secon_ad_num = random.choices(advertisers, altered_weights, k=1)[0]
    
    chosen_ad_num = first_ad_num
    
    #if (B[first_ad_num] - M[first_ad_num])/ B[first_ad_num] < (B[secon_ad_num] - M[secon_ad_num])/ B[secon_ad_num]:
    #    chosen_ad_num = secon_ad_num
    
    chosen_bid = W[chosen_ad_num][kw_num]

    # print(f'selected ad_num: {optimal_ad_num}')
    # print('================STEP OVER=======================')
    return chosen_ad_num, chosen_bid


def online_rand_power(B, W, n, r, m, kw_nums):
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
        #print(f'iter: {t} | B: {B} | M: {M}')
        kw_num = kw_nums[t]
        ad_num, bid = online_rand_power_step(B, M, W, n, kw_num)
        if ad_num == -1:
            continue
        M[ad_num] += bid
        revenue += bid
        Q[t] = ad_num
        
    mean_alloc, std_alloc = assignment(B,M)

    return Q, revenue, mean_alloc, std_alloc


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
    Q, revenue, mean_alloc, std_alloc = online_rand_power(B, W, n, r, m, kw_nums)
    results = {
        'Q': Q,
        'revenue': revenue,
        'mean_alloc': mean_alloc,
        'std_alloc': std_alloc
    }

    return results

def preprocess(data_alias='ds0'):
    data = create_data_vars(data_alias)
    n = data['n']
    m = data['m']
    W = data['W']
    B = data['B']
    r = data['r']
    kw_nums = data['kw_nums']

    return n, m, W, B, r, kw_nums

def assignment(B,M):
    allocation = []
    for i in range(len(B)):
        allocation.append(float(M[i]/B[i]))
    mean = "{:.2f}".format(np.average(allocation))
    std = "{:.2f}".format(np.std(allocation))
    return mean, std

def get_statistics(data_alias='ds0'):
    results = []
    mean_allocs = []
    std_allocs = []
    n, m, W, B, r, kw_nums = preprocess(data_alias)


    for i in range(100):

        #if (i > 0):
        #    random.shuffle(kw_nums)

        Q, revenue, mean_alloc, std_alloc = online_rand_power(B, W, n, r, m, kw_nums)
        results.append(revenue)
        mean_allocs.append(float(mean_alloc))
        std_allocs.append(float(std_alloc))
    mean = "{:.2f}".format(np.average(results))
    std = "{:.2f}".format(np.std(results))
    mean_alloc = "{:.2f}".format(np.average(mean_allocs))
    std_alloc = "{:.2f}".format(np.average(std_allocs))
    return mean, std, mean_alloc, std_alloc

if __name__ == "__main__":
    # When testing, substitute the variables n, m, W, B with appropriate values.
    #print(get_results('ds1'))
    #print(get_results('ds1')['revenue'])
    #print(get_statistics('ds1'))
    print(get_statistics('ds3'))

