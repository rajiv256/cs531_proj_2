from src.offline_greedy_3_1 import get_results as get_results_3_1
from src.offline_lp_3_2 import get_results as get_results_3_2
from src.online_dual_lp_4_3 import get_results as get_results_4_3
from src.online_greedy_4_1 import get_results as get_results_4_1
from src.online_weighted_greedy_4_2 import get_results as get_results_4_2

if __name__=="__main__":
    fns = [get_results_3_1, get_results_3_2, get_results_4_1, get_results_4_2,
           get_results_4_3]
    data_alias = 'ds0_mini'
    A = []
    for fn in fns[3:4]:
        print(fn(data_alias=data_alias))
    print(A)
