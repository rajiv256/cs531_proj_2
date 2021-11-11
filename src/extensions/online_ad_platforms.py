"""In this instantiation of the online AdWords problem, the #variables becomes
n*m*(#slots)*
"""

import numpy as np
import pulp as pl

import configs
from src.data_utils import create_data_vars
from src.pulp_utils import optimize_lp


SLOTS = 3
DISCOUNT_FACTOR = 0.1


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
    """
    Args:
        n: #advertisers
        m: #keywords
        kw_nums: number of the keyword coming to the queries.
    """
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
                A[m + i, i * m * SLOTS + j * SLOTS + s] = (1-s*DISCOUNT_FACTOR) * W[i][kw_num]

    # A has a total of m + n rows and m*n*SLOTS columns.
    return A


def fill_b(n, m, B):
    print(f'B: {B}')
    b = np.zeros((m + n, 1))
    b[:m] = 1
    for i in range(n):
        b[m + i] = B[i]
    return b


def min_max_norm(arr):
    min_ = arr[0]
    max_ = arr[0]
    for i in range(len(arr)):
        if min_ > arr[i]:
            min_ = arr[i]
        if max_ < arr[i]:
            max_ = arr[i]
    new_arr = [(arr[i]-min_)/(max_-min_ + 1e-9) for i in range(len(arr))]
    return new_arr


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
        slot_discount = 1 - slot*DISCOUNT_FACTOR
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
    freqs = [0]*r
    for t in range(len(kw_nums)):
        kw_num = kw_nums[t]

        # Maintaining the keyword frequency.
        freqs[kw_num] += 1
        ad_nums, bids = online_greedy_step(B, M, W, n, kw_num)
        for slot, (slot_ad_num, slot_bid) in enumerate(zip(ad_nums, bids)):
            if slot_ad_num==-1:
                continue
            M[slot_ad_num] += slot_bid
            revenue += slot_bid
            Q[t][slot] = slot_ad_num
    return M, Q, revenue, freqs


def online_weighted_greedy_step_with_dynamic_slots(B, M, W, alphas, n, kw_num, freqs_norm):

    # TODO(rajiv): Check the validity again.

    slots = SLOTS + int(freqs_norm[kw_num]*SLOTS)


    optimal_ad_nums = [-1] * slots
    optimal_bids = [0] * slots
    disc_bids = [-1] * slots
    for slot in range(slots):
        slot_discount = (1 - DISCOUNT_FACTOR*slot)
        for i in range(n):

            # 0 means no bid.
            if W[i][kw_num] == 0:
                continue

            disc = (1 - alphas[i])
            slot_discounted_bid = slot_discount * W[i][kw_num]
            if slot_discounted_bid <= (B[i] - M[i]):
                if disc_bids[slot] < disc * W[i][kw_num]:  # TODO(rajiv): This is correct. It could just be off by a constant factor.
                    disc_bids[slot] = disc * W[i][kw_num]
                    optimal_bids[slot] = slot_discounted_bid
                    optimal_ad_nums[slot] = i
    return optimal_ad_nums, optimal_bids


def online_weighted_greedy_step(B, M, W, alphas, n, kw_num):

    optimal_ad_nums = [-1] * SLOTS
    optimal_bids = [0] * SLOTS
    disc_bids = [-1] * SLOTS
    for slot in range(SLOTS):
        slot_discount = (1 - DISCOUNT_FACTOR*slot)
        for i in range(n):

            # 0 means no bid.
            if W[i][kw_num] == 0:
                continue

            disc = (1 - alphas[i])
            slot_discounted_bid = slot_discount * W[i][kw_num]
            if slot_discounted_bid <= (B[i] - M[i]):
                if disc_bids[slot] < disc * W[i][kw_num]:  # TODO(rajiv): This is correct. It could just be off by a constant factor.
                    disc_bids[slot] = disc * W[i][kw_num]
                    optimal_bids[slot] = slot_discounted_bid
                    optimal_ad_nums[slot] = i
    return optimal_ad_nums, optimal_bids


def expand_W_with_slots_kwprobs(W, kw_nums, kw_probs):
    """First executes np.take to get the n*r to n*m.
    Then expands the final dim to accommodate for the slot values.
    Args:
        W:
        kw_nums:

    Returns:

    """
    W_kwprobs = np.multiply(W, kw_probs)
    W_new = np.take(W_kwprobs, indices=kw_nums, axis=1)
    W_slots = np.expand_dims(W_new, axis=2)  # n*m*1
    W_slots = np.repeat(W_slots, repeats=SLOTS, axis=2)  # n*m*SLOTS
    discounting = [(1 - s*DISCOUNT_FACTOR) for s in range(SLOTS)]
    W_slots = W_slots * discounting
    return W_slots


def online_dual_lp(B, W, n, r, m, kw_nums, kw_probs, eps):
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
    M, Q, revenue, freqs = online_greedy(B, W, n, r, eps_m, kw_nums[:eps_m])

    freqs_norm = min_max_norm(freqs)

    c = np.array(expand_W_with_slots_kwprobs(W, kw_nums, kw_probs))[:, :eps_m, :].flatten()
    A = fill_A(n, eps_m, W, kw_nums[:eps_m])
    b = fill_b(n, eps_m, B)
    c_du, A_du, b_du = get_adword_dual(c, A, b)
    c_du = np.concatenate([c_du[:eps_m], eps * c_du[eps_m:]], axis=0)
    bounds = [(0, 1e9)] * eps_m + [(0, 1)] * n
    obj_value, values = min_lp_solver(c_du, A_du, b_du, bounds)
    alphas = values[eps_m:]

    for t in range(eps_m, m):
        Q.append([])
        ad_nums, bids = online_weighted_greedy_step_with_dynamic_slots(B, M, W, alphas, n, kw_nums[t], freqs_norm)
        # ad_nums, bids = online_weighted_greedy_step(B, M, W, alphas, n, kw_nums[t])

        for slot, (ad_num, bid) in enumerate(zip(ad_nums, bids)):
            if ad_num == -1:
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

    #TODO(rajiv) Draw kw_nums from a gaussian
    mu = r // 2
    sigma = r // 8
    kw_nums = np.random.standard_normal(size=m) * sigma + mu
    kw_nums = [int(kw_num) for kw_num in kw_nums]
    kw_nums = [max(0, min(r-1, item)) for item in kw_nums]
    print(kw_nums)

    kw_probs = [0.2 for i in range(r)]
    Q, revenue = online_dual_lp(B, W, n, r, m, kw_nums, kw_probs, eps=0.1)
    results = {
        'Q': Q,
        'revenue': revenue
    }
    return results


if __name__=="__main__":
    # When testing, substitute the variables n, m, W, B with appropriate values.
    # do something
    print(get_results('ds3')['revenue'])
    # B = [20, 20]
    # W = [[1, 1, 1],
    #      [2, 2, 2],
    #      [3, 3, 3],
    #      [4, 4, 4]]
    # kw_nums = [0, 1, 2]
    # A = fill_A(len(B), len(kw_nums), W, kw_nums)
    # print(A)

