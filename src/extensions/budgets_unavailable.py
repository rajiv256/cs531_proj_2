import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import binom

from src.data_utils import create_data_vars

BETA = 1.15
# np.random.seed(256)


def bar_plot(X):
    x = range(len(X))
    x = [str(i) for i in x]
    data = {'x': x, 'y': X}
    g = sns.barplot(x='x', y='y', data=data)
    plt.savefig('fig.png')

def discount(beta, x):
    return 1 - np.exp(beta*(x-1))


def discount_with_budget(M_i, B_i):
    # print(M_i, B_i)
    return 1 - np.exp(M_i/(B_i + 1e-9) - 1)


def online_weighted_greedy_step(B, M, W, n, kw_num, B_guess):
    optimal_ad_num = -1
    disc_bid = 0
    optimal_bid = 0

    for i in range(n):
        # 0 means no bid.
        # # print(f'i: {i} | kw_num: {kw_num}| discount: {discount(B[i], M[i])} | wij: {W[i][kw_num]} | bid: {discount(B[i], M[i])*W[i][kw_num]}')
        if W[i][kw_num] == 0:
            continue
        if W[i][kw_num] <= (B[i] - M[i]):
            disc = discount_with_budget(M[i], B_guess[i])
            # print(f'ad: {i} | disc: {disc} | bid: {W[i][kw_num]} | val: {W[i][kw_num] * disc}')
            if disc_bid < disc * W[i][kw_num]:
                disc_bid = disc * W[i][kw_num]
                optimal_bid = W[i][kw_num]
                optimal_ad_num = i
    # print(f'selected ad_num: {optimal_ad_num}')
    # print('================STEP OVER=======================')
    return optimal_ad_num, optimal_bid


def adwords_with_unknown_budgets_step(B, M, W, n, kw_num, Y):
    optimal_bid = 0
    disc_bid = -1
    optimal_ad_num = -1
    for i in range(n):
        if W[i][kw_num] == 0:
            continue
        r_i = 1  # TODO(rajiv): See if this is OK!
        disc = discount(BETA, Y[i])  # 1 - e^{BETA*(Y[i]-1)

        if W[i][kw_num] <= (B[i] - M[i]):
            if disc_bid < disc * W[i][kw_num]*r_i:
                disc_bid = disc * W[i][kw_num]*r_i
                optimal_bid = W[i][kw_num]
                optimal_ad_num = i
    return optimal_ad_num, optimal_bid


def adwords_with_unknown_budgets(B, W, n, r, m, kw_nums, B_guess):
    M = [0]*n
    revenue = 0
    Q = [-1]*m

    # Y = np.random.uniform(low=0, high=1, size=n)
    # Y = np.random.binomial(1.0, p=0.1, size=n)
    Y = [binom.pmf(i, n, 0.5) for i in range(n)]
    print(Y)
    for t in range(m):
        kw_num = kw_nums[t]
        # ad_num, bid = online_weighted_greedy_step(B, M, W, n, kw_num, B_guess)
        ad_num, bid = adwords_with_unknown_budgets_step(B, M, W, n, kw_num, Y)
        if ad_num == -1:
            continue
        # print(f'ad_num: {ad_num}')
        M[ad_num] += bid
        revenue += bid
        Q[t] = ad_num
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

    rands = np.random.random(n)
    B_guess = np.array(W).max(axis=1)*m
    # print(B_guess)
    # print(B)
    # bar_plot(B_guess)
    r = data['r']

    Q, revenue = adwords_with_unknown_budgets(B, W, n, r, m, kw_nums, B_guess)
    results = {
        'Q': Q,
        'revenue': revenue
    }
    return results


if __name__ == "__main__":
    # When testing, substitute the variables n, m, W, B with appropriate values.
    revenues = {}
    dss = ['ds0', 'ds1', 'ds2', 'ds3']
    for ds in dss:
        revenues[ds] = []
        for i in range(100):
            results = get_results(ds)
            rev = results['revenue']
            revenues[ds].append(rev)
    for ds in dss:
        if ds not in revenues:
            continue
        print(f'{round(np.mean(revenues[ds]),2)} ({round(np.std(revenues[ds]), 2)})')