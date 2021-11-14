import os
import pickle

import numpy as np

from configs import ROOT_DIR


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


def construct_data_file_from_alias(data_alias='ds0'):
    path = os.path.join(ROOT_DIR, 'data', data_alias+'.pkl')
    return path


def create_data_vars(data_alias='ds0'):
    """Use aliases like ds0, ds1, ds2, ds3 to initialize the required matrices.
    NOTE: The returned W contains expanded dims with bids of dimension n*m inplace of the original n*r,
    Args:
        data_alias:ds0|ds1|ds2|ds3

    Returns: Dictionary of data variables.
    """
    filepath = construct_data_file_from_alias(data_alias)
    assert os.path.exists(filepath), f"Pickle filepath: {filepath} doesn't exist"
    B, W, kw_nums = pickle.load(open(filepath, 'rb'))
    n = len(B)
    m = len(kw_nums)
    r = len(W[0])
    # print(f'W: {len(W), len(W[0])} | n: {n} | m: {m} | r: {r}')

    data = {
        'n': n,
        'm': m,
        'r': r,
        'B': B,
        'W': W,
        'kw_nums': kw_nums
    }
    return data

# print(create_data_vars('ds0'))