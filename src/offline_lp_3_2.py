import numpy as np
import pulp as pl

import configs
from data_utils import create_data_vars
from pulp_utils import optimize_lp


def max_lp_solver(c, A_ub, b_ub, bounds):
    """

    Args:
        c: The below linprog minimizes the objective function. So flip  the sign in c.
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
    solver = pl.getSolver(configs.SOLVER_TYPE)
    obj_value, values = optimize_lp(c, A_ub, b_ub, objective=pl.LpMaximize, solver=solver)

    print(obj_value, values)
    return obj_value, values


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
        b[m+i] = B[i]
    return b


def post_process(X, B, W, n, r, m, kw_nums):
    '''
    Args:
        X: n*m size array. (i*n + m)-th index contains the probability with which i-th advertiser gets the j-th keyword.
        n: #advertisers
        m: #keywords.
    Returns:
        Q: selected bids
        revenue: Total revenue
    '''
    revenue = 0.0
    M = [0.0]*n
    Q = [-1]*m
    sortedX = [(index, x) for index, x in enumerate(X)]
    sortedX.sort(key=lambda x: x[1], reverse=True)
    for index, prob in sortedX:
        ad_num = int(index / m)
        q_num = index % m
        kw_num = kw_nums[q_num]
        bid = W[ad_num][kw_num]
        # 0 means there is no bid.
        if bid == 0:
            continue
        if Q[q_num] == -1 and bid <= B[ad_num]:
            M[ad_num] += bid
            B[ad_num] -= bid
            Q[q_num] = ad_num
            revenue += bid
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


def naive_lp_solver(B, W, n, r, m, kw_nums):
    A = fill_A(n, m, W, kw_nums)
    b = fill_b(n, m, B)
    bounds = [(0, 1.0)] * (n * m)

    # Need to expand this to match n*m size of c.
    # Here we just expand the columns of W to match the `kw_nums` items.
    c = np.array(expand_W(W, kw_nums)).flatten()

    obj_value, values = max_lp_solver(c, A, b, bounds)
    X = values
    Q, revenue = post_process(X, B, W, n, r, m, kw_nums)
    return Q, revenue


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
    Q, revenue = naive_lp_solver(B, W, n, r, m, kw_nums)
    results = {
        'Q': Q,
        'revenue': revenue
    }
    return results


if __name__ == "__main__":
    # When testing, substitute the variables n, m, W, B with appropriate values.
    print(get_results('ds0')['revenue'])
