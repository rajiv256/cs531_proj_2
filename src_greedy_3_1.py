import os
import random
import numpy as np

def greedy(B, W):
    '''
    Args:
        B: Budget of the i-th advertiser
        W: 2D array. W[i][j] refers to the bid value of i-th advertiser to j-th keyword.
    Returns:
        Q: i-th keyword is mapped to Q[i]-th advertiser
        revenue: Total revenue obtained.
    '''
    n = len(B)
    m = len(W[0])
    M = [0]*n
    revenue = 0
    Q = [-1]*len(W[0]) # advertiser that bid for i-th query.
    sortedW = []
    for i in range(n):
        for j in range(m):
            sortedW.append([i, j, W[i][j]])
    sortedW.sort(key=lambda x: x[2], reverse=True)
    for w_ij in sortedW:

        ad_num = w_ij[0]
        kw_num = w_ij[1]
        bid = w_ij[2]

        # Bid is possible when
        # - Advertiser has enough money left.
        # - The keyword was previously unassigned.
        if B[ad_num] >= bid and Q[kw_num] == -1:
            revenue += bid
            B[ad_num] -= bid
            M[ad_num] += bid
            Q[kw_num] = ad_num
    return Q, revenue


if __name__== "__main__":
    Alen = 3
    Qlen = 2
    W = np.random.randint(4, 10, size=(Alen, Qlen))
    B = [2, 3, 30]
    print(f'W: {W}')
    print(f'B: {B}')
    Q, revenue = greedy(B, W)
    print(Q)
    print(revenue)
