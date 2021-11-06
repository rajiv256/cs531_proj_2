from scipy.optimize import linprog
import numpy as np
import math

import configs
from pulp_utils import optimize_lp
import pulp as pl
from data_utils import create_data_vars


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
    # model = pl.LpProblem("Example", pl.LpMaximize)
    # solver_type='PULP_CBC_CMD'
    # solver = pl.getSolver(solver_type)
    # var_names = ['x_'+str(i) for i in range(len(c))]
    # variables = [pl.LpVariable(name=var_names[i]) for i in range(len(var_names))]
    # constraints = [create_constraint(A_ub[i], variables, sense=pl.LpConstraintLE, name='co_'+str(i), rhs=b_ub[i]) for i in range(len(A_ub))]
    # for constraint in constraints:
    #     model.addConstraint(constraint)
    # obj = pl.lpDot(c, variables)
    # model += obj
    solver = pl.getSolver(configs.SOLVER_TYPE)
    obj_value, values = optimize_lp(c, A_ub, b_ub, objective=pl.LpMaximize, solver=solver)

    print(obj_value, values)

def fill_A(n, m, W):
    '''
    Args:
        n: #advertisers
        m: #keywords
    '''
    A = np.zeros((n+m, n*m))
    # First m rows for the ∑_i x_ij <= 1 constraints.
    for i in range(m):
        A[i, i::m] = 1
    # Next n rows will have ∑_j w_ij*x_ij <= B_i for i in 1..n
    for i in range(n):
        for j in range(m):
            A[m+i, i*m + j] = W[i][j]
    # A has a total of m + n rows and m*n columns.
    return A


def fill_b(n, m, B):
    b = np.zeros((m + n, 1))
    b[:m] = 1
    for i in range(n):
        b[m+i] = B[i]
    return b


def naive_lp(X, W, B, n, m):
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
        ad_num = int(index/m)
        kw_num = index%m
        bid = W[ad_num][kw_num]
        # 0 means there is no bid.
        if bid == 0:
            continue
        if Q[kw_num] == -1 and bid <= B[ad_num]:
            M[ad_num] += bid
            B[ad_num] -= bid
            Q[kw_num] = ad_num
            revenue += bid
    return Q, revenue

def test_naive_lp_solver(W, B, n, m):
    A = fill_A(n, m, W)
    b = fill_b(n, m, B)
    bounds = [(0, 1.0)]*(n*m)
    c = W.flatten()
    res = max_lp_solver(c, A, b, bounds)
    # X = res["x"]
    # Q, revenue = naive_lp(X, W, B, n, m)
    # return Q, revenue

if __name__=="__main__":
    # When testing, substitute the variables n, m, W, B with appropriate values.
    data = create_data_vars('ds0')
    n = data['n']
    m = data['m']
    W = data['W']
    B = data['B']

    _ = test_naive_lp_solver(W, B, n, m)