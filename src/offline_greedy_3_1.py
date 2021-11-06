import os
import random
import numpy as np
import pickle
from data_utils import create_data_vars


def greedy(B, W, n, m):
    '''
    Args:
        B: Budget of the i-th advertiser
        W: 2D array. W[i][j] refers to the bid value of i-th advertiser to j-th keyword.
    Returns:
        Q: i-th keyword is mapped to Q[i]-th advertiser. Q[i] = -1 means that the i-th query is unassigned.
        revenue: Total revenue obtained.
    '''
    M = [0]*n
    revenue = 0
    Q = [-1]*len(W[0]) # advertiser that bid for i-th query.
    sortedW = []
    for i in range(n):
        for j in range(m):
            # O weight means no bid.
            if W[i][j] == 0:
                continue
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
    Q, revenue = greedy(B, W, n, m)
    results = {
        'Q': Q,
        'revenue': revenue
    }
    return results


if __name__== "__main__":
    data = create_data_vars('ds0')
    n = data['n']
    m = data['m']
    W = data['W']
    B = data['B']
    Q, revenue = greedy(B, W, n, m)

    print(f'Q:\n{Q}\nrevenue:\n{revenue}')
