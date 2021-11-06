"""
Utils to manipulate the data. This is not source code. Refer to the src/ for the source code.
"""
import os
import pickle
from configs import ROOT_DIR


def test_0(pklFilePath):
    obj = pickle.load(open(pklFilePath, 'rb'))
    print(obj)


if __name__ == "__main__":
    test_0(os.path.join(ROOT_DIR, 'data/ds0.pkl'))
