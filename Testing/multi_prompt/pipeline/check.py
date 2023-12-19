import pickle
import numpy as np

with open('../coach_gen/vannnila_gen', 'rb') as f:
    vanni = pickle.load(f)
# np.save('../generated_coach/old/vannila_generated.npy', vanni)
#
# va = np.load('../generated_coach/old/vannila_generated.npy', allow_pickle=True).item()
# gcot = np.load('../generated_coach/old/gcot_generated.npy', allow_pickle=True).item()

print(len(dis))

