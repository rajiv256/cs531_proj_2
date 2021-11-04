from scipy.optimize import linprog
import numpy as np


def min_lp_solver(c, A_ub, b_ub, bounds):
    '''
    Args:
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
    '''
    # Negative sign to `c` as this function minimizes by default.
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    return res


def get_adword_dual(c, A, b):
    '''
    Args:
        c: (n*m, 1)
        A: (m+n, n*m)
        b: (m+n, 1)
    Returns:
        c_new: (m+n, 1)
        A_new: (n*m, m+n)
        b_new: (n*m, 1)
    '''
    c_new = b #(m+n, 1)
    A_new = -A.T
    b_new = -c
    return c_new, A_new, b_new


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


def online_greedy_step(B, M, W, n, kw_num):

    optimal_ad_num = -1
    optimal_bid = 0
    for i in range(n):
        if W[i][kw_num] <= B[i]-M[i]:
            if optimal_bid < W[i][kw_num]:
                optimal_bid = W[i][kw_num]
                optimal_ad_num = i
    return optimal_ad_num, optimal_bid


def online_greedy(B, W, n, m):

    M = [0]*n
    revenue = 0
    Q = [-1]*m

    for i in range(m):

        ad_num, bid = online_greedy_step(B, M, W, n, i)
        M[ad_num] += bid
        revenue += bid
        Q[i] = ad_num
    return M, Q, revenue


def online_weighted_greedy_step(B, M, W, alphas, n, kw_num):

    optimal_ad_num = -1
    optimal_bid = 0
    for i in range(n):
        if W[i][kw_num] <= B[i]-M[i]:
            if optimal_bid < (1-alphas[i])*W[i][kw_num]:
                optimal_bid = W[i][kw_num]
                optimal_ad_num = i
    return optimal_ad_num, optimal_bid


def online_dual_lp(B, W, n, m, eps):

    eps_m = int(eps*m)
    M, Q, revenue = online_greedy(B, W, n, eps_m)
    W_trunc = W[:, :eps_m]
    c_trunc = W_trunc.flatten() # As mentioned in the  problem.
    A_trunc = fill_A(n, eps_m, W_trunc)
    b_trunc = fill_b(n, eps_m, B)

    c_du, A_du, b_du = get_adword_dual(c_trunc, A_trunc, b_trunc)
    c_du = np.concatenate([c_du[:eps_m], eps*c_du[eps_m:]], axis=0)
    bounds = [(0, 1.0)]*(eps_m+n)

    res = min_lp_solver(c=c_du, A_ub=A_du, b_ub=b_du, bounds=bounds)

    alphas = res['x'] # (eps_m + n)
    alphas = alphas[eps_m:] # (n)

    for t in range(eps_m, m):

        ad_num, bid = online_weighted_greedy_step(B, M, W, alphas, n, t)
        if ad_num == -1:
            Q.append(ad_num)
            continue

        M[ad_num] += bid
        revenue += bid
        Q.append(ad_num)

    return Q, revenue



if __name__=="__main__":
    # When testing, substitute the variables n, m, W, B with appropriate values.
    n = 4
    m = 2
    W = np.random.randint(1, 10, (n, m))
    B = np.random.randint(2, 10, (n))
    c = W.flatten()
    A = fill_A(n, m, W)
    b = fill_b(n, m, B)
    eps = 0.1
    Q, revenue = online_dual_lp(B, W, n, m, eps)
    print(Q, revenue)
