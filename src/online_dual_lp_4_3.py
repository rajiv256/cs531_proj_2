"""
In this sub-section, the objective function would be the total revenue. Which is bid_ij multiplied by x_ij.
Here, j is the query number. However, we only have bid_ij to be based on keywords and not queries. Therefore, we might
need to expand the W matrix to duplicate the queries.
"""
import math

import numpy as np
from scipy.optimize import linprog

from data_utils import create_data_vars


def min_lp_solver(c, A_ub, b_ub, bounds):
    """
    Args:
        b_ub:
        A_ub:
        c: Co-efficients for each variable.
        bounds: (min, max) bounds of each of the x_i. Set default to (0, 1.0)
    Returns:
         con: array([], dtype=float64)
         fun: -21.999999840824927
     message: 'Optimization terminated successfully.'
         nit: 6
       slack: array([3.89999997e+01, 8.46872599e-08])
      status: 0
     success: True
           x: array([ 9.99999989, -2.99999999])
    """
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    return res


def get_adword_dual(c, A, b):
    """

    Args:
        c:
        A:
        b:

    Returns:
        c_new: (m+n, 1)
        A_new: (n*m, m+n)
        b_new: (n*m, 1)
    """
    c_new = b  # (m+n, 1)
    A_new = -A.T
    b_new = -c
    return c_new, A_new, b_new


# def fill_A(n, m, W):
#     """
#     Creates a co-efficient matrix for the constraints.
#     Args:
#         n:
#         m:
#         W:
#
#     Returns:
#
#     """
#     A = np.zeros((n + m, n * m))
#
#     # First m rows for the ∑_i x_ij <= 1 constraints.
#     for i in range(m):
#         A[i, i::m] = 1
#
#     # Next n rows will have ∑_j w_ij*x_ij <= B_i for i in 1..n
#     for i in range(n):
#         for j in range(m):
#             A[m + i, i * m + j] = W[i][j]
#
#     # A has a total of m + n rows and m*n columns.
#     return A


def fill_A(n, m, W, kw_nums):
    '''
    Args:
        n: #advertisers
        m: #keywords
    '''
    A = np.zeros((n + m, n * m))
    # First m rows for the ∑_i x_ij <= 1 constraints.
    for i in range(m):
        A[i, i::m] = 1
    # Next n rows will have ∑_j w_ij*x_ij <= B_i for i in 1..n
    for i in range(n):
        for j in range(m):
            kw_num = kw_nums[j]
            A[m + i, i * m + j] = W[i][kw_num]
    # A has a total of m + n rows and m*n columns.
    return A

def fill_b(n, m, B):
    b = np.zeros((m + n, 1))
    b[:m] = 1
    for i in range(n):
        b[m + i] = B[i]
    return b


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


def online_greedy(B, W, n, r, m, kw_nums):
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
    return M, Q, revenue


def online_weighted_greedy_step(B, M, W, alphas, n, kw_num):
    optimal_ad_num = -1
    optimal_bid = 0

    for i in range(n):
        # 0 means no bid.
        # print(f'i: {i} | kw_num: {kw_num}| discount: {discount(B[i], M[i])} | wij: {W[i][kw_num]} | bid: {discount(B[i], M[i])*W[i][kw_num]}')
        if W[i][kw_num] == 0:
            continue
        if W[i][kw_num] <= (B[i] - M[i]):

            if optimal_bid < (1 - alphas[i]) * W[i][kw_num]:
                optimal_bid = W[i][kw_num]
                optimal_ad_num = i
    # print(f'selected ad_num: {optimal_ad_num}')
    return optimal_ad_num, optimal_bid


def expand_W(W, kw_nums):
    """Transforms W from containing bids for keywords to contain bids for queries. This is done using
    np.take(https://numpy.org/doc/stable/reference/generated/numpy.take.html). Now we can directly apply the weights
    like we did in the greedy setting.
    Args:
        W:
        kw_nums:
    Returns:

    """
    W_new = np.take(W, indices=kw_nums, axis=1)
    return W_new


def online_dual_lp(B, W, n, r, m, kw_nums, eps):
    """

    Args:
        B:
        W:
        n:
        r:
        m:
        kw_nums:
        eps:

    Returns:

    """
    eps_m = int(eps * m)
    M, Q, revenue = online_greedy(B, W, n, r, eps_m, kw_nums[:eps_m])
    c = np.array(expand_W(W, kw_nums))[:, :eps_m].flatten()
    A = fill_A(n, eps_m, W, kw_nums[:eps_m])
    b = fill_b(n, eps_m, B)
    print(f'eps_m: {eps_m}')
    print(f'A: {len(A), len(A[0])} | c: {len(c)} | b: {len(b)}')
    c_du, A_du, b_du = get_adword_dual(c, A, b)
    c_du = np.concatenate([c_du[:eps_m], eps * c_du[eps_m:]], axis=0)
    bounds = [(0, math.inf)] * (eps_m + n)

    res = min_lp_solver(c_du, A_du, b_du, bounds)
    alphas = res['x'][eps_m:]

    for t in range(eps_m, m):
        ad_num, bid = online_weighted_greedy_step(B, M, W, alphas, n, kw_nums[t])
        if ad_num == -1:
            Q.append(ad_num)
            continue
        M[ad_num] += bid
        revenue += bid
        Q.append(ad_num)
    msum = sum(M)
    return Q, msum  # revenue also gives the same answer


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
    Q, revenue = online_dual_lp(B, W, n, r, m, kw_nums, eps=0.1)
    results = {
        'Q': Q,
        'revenue': revenue
    }
    return results




if __name__ == "__main__":
    # When testing, substitute the variables n, m, W, B with appropriate values.
    # do something
    print(get_results('ds1')['revenue'])
