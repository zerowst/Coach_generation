import numpy as np
import pickle

with open('gcot', 'rb') as f:
    re = pickle.load(f)
print(re)