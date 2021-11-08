import os
import pickle

from src.improvements.offline_lp_3_2_withB import \
    get_results as get_results_3_2_withB

from configs import DATA_DIR
from src.offline_greedy_3_1 import get_results as get_results_3_1
from src.online_dual_lp_4_3 import get_results as get_results_4_3
from src.online_greedy_4_1 import get_results as get_results_4_1
from src.online_weighted_greedy_4_2 import get_results as get_results_4_2


def create_test(data_alias='ds_test'):
    B = [60, 80]
    kw_nums = [0, 1]
    W = [[61, 50], [30, 31]]
    obj = B, W, kw_nums
    pickle.dump(obj, open(os.path.join(DATA_DIR, data_alias + '.pkl'), 'wb'))


if __name__=="__main__":
    fns = [get_results_3_1, get_results_3_2_withB, get_results_4_1,
           get_results_4_2,
           get_results_4_3]
    data_alias = 'ds_test'
    create_test(data_alias)
    A = []
    for fn in fns[:2]:
        A.append(fn(data_alias=data_alias))
        print(fn(data_alias=data_alias))
    print(A)
