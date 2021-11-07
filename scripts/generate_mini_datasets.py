import os
import pickle

from configs import DATA_DIR
from data.gen import ds0_gen

m = 10
data = ds0_gen(10)
pickle.dump(data, open(os.path.join(DATA_DIR, 'ds0_mini.pkl'), 'wb'))
# data = ds1_gen(2, [100, 50])
# pickle.dump(data, open(os.path.join(DATA_DIR, 'ds1_mini.pkl'), 'wb'))
# data = ds2_gen(2, [100, 50])
# pickle.dump(data, open(os.path.join(DATA_DIR, 'ds2_mini.pkl'), 'wb'))
# data = ds3_gen(2, expo=1, f=1)
# pickle.dump(data, open(os.path.join(DATA_DIR, 'ds3_mini.pkl'), 'wb'))
