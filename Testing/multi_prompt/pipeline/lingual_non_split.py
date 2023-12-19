import pickle
import numpy as np
import csv

with open('../../annotation/human_detection_all', 'rb') as f:
    human_detection = pickle.load(f)

with open('../../annotation/human_correction_all', 'rb') as f:
    human_correction = pickle.load(f)

eval_key = np.load('../../annotation/eval_keys.npy')

lingual = {}
non_lingual = {}
n_i = 0
l_i = 0
for key in eval_key:
    lingual[key] = []
    non_lingual[key] = []
    for i, (det, cor) in enumerate(zip(human_detection[key], human_correction[key])):
        if det == 'None' and cor == 'None':
            non_lingual[key].append(i)
            n_i += 1
        else:
            lingual[key].append(i)
            l_i += 1

with open('non_lingual_pos', 'wb') as f:
    pickle.dump(non_lingual, f)

with open('lingual_pos', 'wb') as f:
    pickle.dump(lingual, f)

print(n_i, lingual)
