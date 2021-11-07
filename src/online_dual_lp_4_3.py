"""
In this sub-section, the objective function would be the total revenue. Which is bid_ij multiplied by x_ij.
Here, j is the query number. However, we only have bid_ij to be based on keywords and not queries. Therefore, we might
need to expand the W matrix to duplicate the queries.
"""

import numpy as np
import pulp as pl

import configs
from src.data_utils import create_data_vars
from src.pulp_utils import optimize_lp


def min_lp_solver(c, A_ub, b_ub, bounds):
    """
    Args:
        b_ub:
        A_ub:
        c: Co-efficients for each variable.
        bounds: (min, max) bounds of each of the x_i. Set default to (0, 1.0)
    Returns:

    """
    solver = pl.getSolver(configs.SOLVER_TYPE)
    obj_value, values = optimize_lp(c, A_ub, b_ub, objective=pl.LpMinimize, solver=solver, bounds=bounds)
    return obj_value, values


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
        if W[i][kw_num]==0:
            continue
        if W[i][kw_num] <= (B[i] - M[i]):
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
    Q = [-1] * m

    for t in range(len(kw_nums)):
        kw_num = kw_nums[t]
        ad_num, bid = online_greedy_step(B, M, W, n, kw_num)
        if ad_num==-1:
            continue
        M[ad_num] += bid
        revenue += bid
        Q[t] = ad_num
    return M, Q, revenue


def online_weighted_greedy_step(B, M, W, alphas, n, kw_num):
    optimal_ad_num = -1
    optimal_bid = 0
    disc_bid = 0

    for i in range(n):
        # 0 means no bid.
        if W[i][kw_num]==0:
            # print(f'ad: {i} | bid: 0 | skipping')
            continue
        disc = (1 - alphas[i])
        # print(f'alphas: {alphas}')
        # print(f'ad: {i} | disc: {disc} | bid: {W[i][kw_num]} | val: {W[i][kw_num] * disc} | B[i]: {B[i]} | M[i]: {M[i]}')
        if W[i][kw_num] <= (B[i] - M[i]):
            if disc_bid <= disc * W[i][kw_num]:
                disc_bid = disc * W[i][kw_num]
                optimal_bid = W[i][kw_num]
                optimal_ad_num = i
    # print(f'selected ad_num: {optimal_ad_num}')
    # print('================STEP OVER=======================')
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
    c_du, A_du, b_du = get_adword_dual(c, A, b)
    c_du = np.concatenate([c_du[:eps_m], eps * c_du[eps_m:]], axis=0)
    bounds = [(0, 1e9)] * eps_m + [(0, 1)] * n

    obj_value, values = min_lp_solver(c_du, A_du, b_du, bounds)
    alphas = values[eps_m:]

    for t in range(eps_m, m):
        # print(f'iter: {t} | B: {B} | M: {M}')
        ad_num, bid = online_weighted_greedy_step(B, M, W, alphas, n,
            kw_nums[t])
        if ad_num==-1:
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


if __name__=="__main__":
    # When testing, substitute the variables n, m, W, B with appropriate values.
    # do something
    print(get_results('ds1')['revenue'])
