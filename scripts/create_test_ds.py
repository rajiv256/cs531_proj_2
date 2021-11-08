import os
import pickle

from configs import DATA_DIR


B = [90]
kw_nums = [0, 1, 2]
W = [[70, 40, 40]]
n = 1
m = 3
r = 3

obj = B, W, kw_nums
pickle.dump(obj, open(os.path.join(DATA_DIR, 'ds_test.pkl'), 'wb'))
