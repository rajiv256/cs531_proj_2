"""
In this sub-section, the objective function would be the total revenue. Which is bid_ij multiplied by x_ij.
Here, j is the query number. However, we only have bid_ij to be based on keywords and not queries. Therefore, we might
need to expand the W matrix to duplicate the queries.
"""
from scipy.optimize import linprog
import numpy as np


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


def fill_A(n, m, W):
    """
    Creates a co-efficient matrix for the constraints.
    Args:
        n:
        m:
        W:

    Returns:

    """
    A = np.zeros((n + m, n * m))

    # First m rows for the ∑_i x_ij <= 1 constraints.
    for i in range(m):
        A[i, i::m] = 1

    # Next n rows will have ∑_j w_ij*x_ij <= B_i for i in 1..n
    for i in range(n):
        for j in range(m):
            A[m + i, i * m + j] = W[i][j]

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


def online_dual_lp(B, W, n, m, eps):
    eps_m = int(eps * m)
    M, Q, revenue = online_greedy(B, W, n, eps_m)
    W_trunc = W[:, :eps_m]
    c_trunc = W_trunc.flatten()  # As mentioned in the  problem.
    A_trunc = fill_A(n, eps_m, W_trunc)
    b_trunc = fill_b(n, eps_m, B)

    c_du, A_du, b_du = get_adword_dual(c_trunc, A_trunc, b_trunc)
    c_du = np.concatenate([c_du[:eps_m], eps * c_du[eps_m:]], axis=0)
    bounds = [(0, 1.0)] * (eps_m + n)

    res = min_lp_solver(c=c_du, A_ub=A_du, b_ub=b_du, bounds=bounds)

    alphas = res['x']  # (eps_m + n)
    alphas = alphas[eps_m:]  # (n)

    for t in range(eps_m, m):

        ad_num, bid = online_weighted_greedy_step(B, M, W, alphas, n, t)
        if ad_num == -1:
            Q.append(ad_num)
            continue

        M[ad_num] += bid
        revenue += bid
        Q.append(ad_num)

    return Q, revenue


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


if __name__ == "__main__":
    # When testing, substitute the variables n, m, W, B with appropriate values.
    n = 4
    r = 2
    m = 10

    W = np.random.randint(1, 10, (n, r))
    B = np.random.randint(2, 10, n)
    kw_nums = np.random.randint(0, r, m)
    bids = expand_W(W, kw_nums) # (n*m)

    c = bids.flatten()
    A = fill_A(n, m, bids)
    b = fill_b(n, m, B)
    eps = 0.1
    Q, revenue = online_dual_lp(B, bids, n, m, eps)
    print(Q, revenue)
