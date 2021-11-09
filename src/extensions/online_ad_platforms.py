"""In this instantiation of the online AdWords problem, the #variables becomes
n*m*(#slots)*
"""
import numpy as np
import pulp as pl

import configs
from src.data_utils import create_data_vars
from src.pulp_utils import optimize_lp


SLOTS = 3
DISCOUNT_FACTOR = 0.5


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
    A = np.zeros([m + n, m * n * SLOTS])

    # First m rows for the ∑_i ∑_s x_ijs <= 1 constraints
    for i in range(m):
        start = i * SLOTS
        while start + SLOTS < len(A[i]):
            A[i, start:start + SLOTS] = 1
            start += m * SLOTS

    # Next n rows will have ∑_j ∑_s w_ijs x_ijs d_ijs <= B_i
    for i in range(n):
        for j in range(m):
            kw_num = kw_nums[j]
            for s in range(SLOTS):
                A[m + i, i * m * SLOTS + j * SLOTS + s] = (
                                                                      DISCOUNT_FACTOR ** s) * \
                                                          W[i][kw_num]

    # A has a total of m + n rows and m*n*SLOTS columns.
    return A


def fill_b(n, m, B):
    print(f'B: {B}')
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
    optimal_ad_num = [-1] * SLOTS
    optimal_bid = [0] * SLOTS
    for slot in range(SLOTS):
        slot_discount = DISCOUNT_FACTOR ** slot
        for i in range(n):

            # 0 means no bid.
            if W[i][kw_num]==0:
                continue
            if slot_discount * W[i][kw_num] <= (B[i] - M[i]):
                if optimal_bid[slot] <= slot_discount * W[i][kw_num]:
                    optimal_bid[slot] = slot_discount * W[i][kw_num]
                    optimal_ad_num[slot] = i
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
    Q = [[-1] * SLOTS] * m

    for t in range(len(kw_nums)):
        kw_num = kw_nums[t]
        ad_nums, bids = online_greedy_step(B, M, W, n, kw_num)
        for slot, (slot_ad_num, slot_bid) in enumerate(zip(ad_nums, bids)):
            if slot_ad_num==-1:
                continue
            M[slot_ad_num] += slot_bid
            revenue += slot_bid
            Q[t][slot] = slot_ad_num
    return M, Q, revenue


def online_weighted_greedy_step(B, M, W, alphas, n, kw_num):
    optimal_ad_nums = [-1] * SLOTS
    optimal_bids = [0] * SLOTS
    disc_bids = [-1] * SLOTS
    for slot in range(SLOTS):
        slot_discount = (DISCOUNT_FACTOR ** slot)
        for i in range(n):

            # 0 means no bid.
            if W[i][kw_num]==0:
                continue
            disc = (1 - alphas[i])
            # print(f'alphas: {alphas}')
            # print(f'ad: {i} | disc: {disc} | bid: {W[i][kw_num]} | val: {W[i][kw_num] * disc} | B[i]: {B[i]} | M[i]: {M[i]}')
            slot_discounted_bid = slot_discount * W[i][kw_num]
            if slot_discounted_bid <= (B[i] - M[i]):
                if disc_bids[slot] < disc * W[i][
                    kw_num]:  # TODO(rajiv): This is correct. It could just be off by a constant factor.
                    disc_bids[slot] = disc * W[i][kw_num]
                    optimal_bids[slot] = slot_discounted_bid
                    optimal_ad_nums[slot] = i
    # print(f'selected ad_num: {optimal_ad_num}')
    # print('================STEP OVER=======================')
    return optimal_ad_nums, optimal_bids


def expand_W_withslots(W, kw_nums):
    """First executes np.take to get the n*r to n*m.
    Then expands the final dim to accommodate for the slot values.
    Args:
        W:
        kw_nums:

    Returns:

    """
    W_new = np.take(W, indices=kw_nums, axis=1)
    W_slots = np.expand_dims(W_new, axis=2)  # n*m*1
    W_slots = np.repeat(W_slots, repeats=SLOTS, axis=2)  # n*m*SLOTS
    discounting = [(DISCOUNT_FACTOR ** s) for s in range(SLOTS)]
    W_slots = W_slots * discounting
    return W_slots


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
    c = np.array(expand_W_withslots(W, kw_nums))[:, :eps_m, :].flatten()
    A = fill_A(n, eps_m, W, kw_nums[:eps_m])
    b = fill_b(n, eps_m, B)
    print(A)
    print(
        f'c: {c.shape} | A: {A.shape} | b: {b.shape}'
    )
    c_du, A_du, b_du = get_adword_dual(c, A, b)
    print(
        f'c_du: {c_du.shape} | A_du: {A_du.shape} | b_du: {b_du.shape}'
    )
    c_du = np.concatenate([c_du[:eps_m], eps * c_du[eps_m:]], axis=0)
    bounds = [(0, 1e9)] * eps_m + [(0, 1)] * n
    obj_value, values = min_lp_solver(c_du, A_du, b_du, bounds)
    alphas = values[eps_m:]

    for t in range(eps_m, m):
        # print(f'iter: {t} | B: {B} | M: {M}')
        Q.append([])
        ad_nums, bids = online_weighted_greedy_step(B, M, W, alphas, n,
            kw_nums[t])
        for slot, (ad_num, bid) in enumerate(zip(ad_nums, bids)):
            if ad_num==-1:
                Q[t].append(ad_num)
                continue
            M[ad_num] += bid
            revenue += bid
            Q[t].append(ad_num)
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
    print(get_results('ds2')['revenue'])
    # B = [20, 20]
    # W = [[1, 1, 1],
    #      [2, 2, 2],
    #      [3, 3, 3],
    #      [4, 4, 4]]
    # kw_nums = [0, 1, 2]
    # A = fill_A(len(B), len(kw_nums), W, kw_nums)
    # print(A)
