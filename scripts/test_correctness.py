import os
import pickle

from configs import DATA_DIR
from data.gen import ds0_gen, ds1_gen, ds2_gen, ds3_gen
from src.offline_lp_3_2 import get_results as get_results_3_2


def create_test(data_alias='ds_test'):
    B = [60, 80]
    W = [[61, 50],
         [30, 31]]
    kw_nums = [0, 1]
    obj = B, W, kw_nums
    pickle.dump(obj, open(os.path.join(DATA_DIR, data_alias + '.pkl'), 'wb'))


def ds0_gen_min(m=10):
    B, W, kw_nums = ds0_gen(10)
    obj = B, W, kw_nums
    pickle.dump(obj, open(os.path.join(DATA_DIR, 'ds0_mini.pkl'), 'wb'))


def ds1_gen_min(n=3, B=10):
    B, W, kw_nums = ds1_gen(n, B)
    obj = B, W, kw_nums
    pickle.dump(obj, open(os.path.join(DATA_DIR, 'ds1_mini.pkl'), 'wb'))


def ds2_gen_min(n=3, B=10):
    B, W, kw_nums = ds2_gen(n, B)
    obj = B, W, kw_nums
    pickle.dump(obj, open(os.path.join(DATA_DIR, 'ds2_mini.pkl'), 'wb'))


def ds3_gen_min(n=3, expo=1.3, f=10):
    W, kw_nums = ds3_gen(n, expo, f)
    B = [10**(i+1) for i in range(n)]
    obj = B, W, kw_nums
    pickle.dump(obj, open(os.path.join(DATA_DIR, 'ds3_mini.pkl'), 'wb'))


if __name__=="__main__":
    # fns = [get_results_3_1, get_results_3_2, get_results_4_1,
    #        get_results_4_2,
    #        get_results_4_3]
    # descriptors = ['3_1', '3_2', '4_1', '4_2', '4_3']
    # # descriptors = descriptors[1:2]
    # # fns = fns[1:2]
    # # ds0_gen_min()
    # # results = get_results_4_1('ds0_mini')
    # # print(results['revenue'])
    # # print(results['Q'])
    # # ds2_gen_min(n=6, B=5)
    # # ds1_gen_min(n=20, B=20)
    # # ds3_gen_min(2, 1.3, 100)
    # # create_test()
    # data_alias = 'ds3'
    # results_lst = []
    # for desc, fn in zip(descriptors, fns):
    #     results = fn(data_alias)
    #     results_lst.append(results)
    #
    # print("Printing Results ############## ")
    # for desc, results in zip(descriptors, results_lst):
    #     vars = create_data_vars(data_alias)
    #     print(f'For Descriptor: {desc} ######')
    #     print(vars)
    #     print(f'Q\n{results["Q"]}\nrevenue\n{results["revenue"]}')
    #     print("###########################################")
    #
    # print("Just Revenues ###################")
    # for desc, results in zip(descriptors, results_lst):
    #     print(f'desc: {desc} -> {results["revenue"]}')

    create_test('ds_test')
    results = get_results_3_2('ds_test')
    print(results['revenue'], results['Q'])