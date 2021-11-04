import os
import numpy as np
import random

def discount(b, m):
    return 1-np.exp(m/b - 1)


def online_weighted_greedy_step(B, M, W, n, kw_num):

    optimal_ad_num = -1
    optimal_bid = 0
    for i in range(n):
        if W[i][kw_num] <= B[i]-M[i]:
            if optimal_bid < discount(B[i], M[i])*W[i][kw_num]:
                optimal_bid = W[i][kw_num]
                optimal_ad_num = i
    return optimal_ad_num, optimal_bid


def online_weighted_greedy(B, W, n, m):

    M = [0]*n
    revenue = 0
    Q = [-1]*m

    for i in range(m):

        ad_num, bid = online_weighted_greedy_step(B, M, W, n, i)
        M[ad_num] += bid
        revenue += bid
        Q[i] = ad_num
    return Q, revenue


if __name__=="__main__":
    # When testing, substitute the variables n, m, W, B with appropriate values.
    n = 4
    m = 2
    W = np.random.randint(1, 10, (n, m))
    B = np.random.randint(2, 10, (n))
