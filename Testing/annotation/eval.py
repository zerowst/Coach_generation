from evaluate import load
from Testing.annotation.metrics import qa_f1_score, qa_f1_zh_score
from datasets import load_metric
import numpy as np
import pickle
import jieba



###-----------------------------------------------------------------------------------------
# Measure the number of human labels in each case

# Filter 300 cases

# Find cases with incorrect labels

# In human label file, the following cases have problems: 110 has different number of labels
# [6, 32, 97, 147, 160, 163, 170, 216, 216, 216, 216, 233, 278, 325, 365, 402, 428, 430, 456, 473, 474, 477] + 110
#
# Load selected key
# Given annotation file of different model and selected key, save detection and correction dict(300 elements)


# gpt coach detection: EM 0.156 F1 0.202 correction: EM 0.094 F1 0.138

# delete none list {'exact_match': 0.07504078303425775} # f1: 0.07612833061446438
# correction# {'exact_match': 0.13539967373572595}# f1: 0.14139413242186488

# llama lora result:
# {'exact_match': 0.5791190864600326} # f1: 0.5791190864600326
# correction # {'exact_match': 0.5301794453507341} # f1: 0.532626427406199


### Both are not None 546/1315

# llama lora
# {'exact_match': 0.25274725274725274}# f1: 0.35309084365322263
# correction# {'exact_match': 0.08424908424908426} # f1: 0.24424431135373628

# llama no lora
# {'exact_match': 0.0347985347985348} # f1: 0.059994546899308804
# correction # {'exact_match': 0.009157509157509158} # f1: 0.02349694781837291

# gpt
# {'exact_match': 0.27106227106227104}# # f1: 0.3756535768175444
# correction# {'exact_match': 0.054945054945054944}# f1: 0.13646170072309555

### Not both None 702/1315

# llama lora
# {'exact_match': 0.2492877492877493} # f1: 0.33151808528882454
# correction # {'exact_match': 0.08547008547008547} # f1: 0.22515639810798355

# llama not lora
# {'exact_match': 0.13817663817663817} # f1: 0.1588927875964913
# correction # {'exact_match': 0.04843304843304843} # f1: 0.05958594516927579

# gpt
# {'exact_match': 0.2264957264957265} # f1: 0.3118348196845723
# correction# {'exact_match': 0.0584045584045584}# f1: 0.13490129179854063











with open('human_detection_all', 'rb') as f:
    human_detection = pickle.load(f)

with open('human_correction_all', 'rb') as f:
    human_correction = pickle.load(f)

with open('machine_detection_all', 'rb') as f:
    lora_detection = pickle.load(f)

with open('machine_correction_all', 'rb') as f:
    lora_correction = pickle.load(f)

eval_key = np.load('eval_keys.npy')
zero_key = np.load('zero_case.npy')

train_key = [k for k in range(1, 500) if k not in np.append(eval_key, zero_key)]


def extract(keys, dic):
    list_out = []
    for key in keys:
        list_out += dic[key]

    return list_out

def extract_from_dic(keys):
    human_detection_ = extract(keys, human_detection)
    human_correction_ = extract(keys, human_correction)
    detection = extract(keys, lora_detection)
    correction = extract(keys, lora_correction)
    return human_detection_, human_correction_, detection, correction


# human_detection, human_correction, detection, correction = extract_from_dic(train_key)
#
#
# print('...')
# not_none_list = []
# for i, (det, cor) in enumerate(zip(human_detection, human_correction)):
#     if not (det == 'None' and cor == 'None'):
#         not_none_list.append(i)
# # print(h_det_none/len(human_detection))
# print(len(not_none_list), len(human_detection))
#
#
# wrong_detection_cases = []
# wrong_correction_cases = []
# for i, (h_d, d) in enumerate(zip(human_detection, detection)):
#     if h_d != d:
#         wrong_detection_cases.append(i)
#
# for i, (h_d, d) in enumerate(zip(human_correction, correction)):
#     if h_d != d:
#         wrong_correction_cases.append(i)
#
# wrong_cases = np.unique(wrong_detection_cases+wrong_correction_cases)



#
# exit()

# human_detection = [human_detection[d] for d in not_none_list]
# human_correction = [human_correction[c] for c in not_none_list]
#
# detection = [detection[d] for d in not_none_list]
# correction = [correction[c] for c in not_none_list]





def f1_score(pre, ref):
    score = 0
    lens = len(pre)
    if len(pre) == len(ref):
        for pre, ref in zip(pre, ref):
            score += qa_f1_zh_score(pre, ref)

    return score/lens


print(len(human_detection), len(lora_detection))
# print(human_detection[:5])
# print(lora_detection[:5])
# print(human_correction[:5])
# print(lora_correction[:5])


### check number of wrong detection and correction cases
wrong_detection_cases = {}
wrong_correction_cases = {}


for key in eval_key:
    wrong_detection_cases[key] = []
    wrong_correction_cases[key] = []
    for i, (h_d, d) in enumerate(zip(human_detection[key], lora_detection[key])):
        if h_d != d:
            wrong_detection_cases[key].append(i)

    for i, (h_d, d) in enumerate(zip(human_correction[key], lora_correction[key])):
        if h_d != d:
            wrong_correction_cases[key].append(i)

wrong_detection_cases = {k: v for k, v in wrong_detection_cases.items() if len(v) > 0}
wrong_correction_cases = {k: v for k, v in wrong_correction_cases.items() if len(v) > 0}
wrong_cases = list(set(wrong_detection_cases.keys()) | set(wrong_correction_cases.keys()))

wrong_dict = {key: list(set((wrong_detection_cases.get(key, []) + wrong_correction_cases.get(key, [])))) for key in wrong_cases}


print(wrong_dict)
lens = [len(v) for v in wrong_dict.values()]
print(sum(lens))






exact_match_metric = load("exact_match")
results = exact_match_metric.compute(predictions=detection, references=human_detection)
#
print(results)
print('f1:', f1_score(detection, human_detection))
#
print('correction')
#
results = exact_match_metric.compute(predictions=correction, references=human_correction)
#
print(results)
print('f1:', f1_score(correction, human_correction))


### training set wrong machine label cases
"""
{6: [1, 3, 6, 8, 9, 10], 20: [2], 27: [1], 29: [4], 32: [0, 4, 5], 38: [3], 42: [0], 
45: [6], 304: [4], 49: [0], 61: [1], 63: [5], 64: [0], 323: [3], 325: [0], 73: [9, 10, 6], 
76: [4, 6], 340: [1], 87: [1], 343: [4], 344: [3], 102: [0, 3], 365: [2], 110: [3, 4], 
366: [1, 2], 113: [1], 371: [3], 118: [1], 377: [2, 3], 121: [3], 381: [2], 395: [3], 
402: [5, 6], 147: [0], 405: [2], 157: [1, 2], 158: [0], 163: [0], 164: [3, 4], 166: [2, 4], 
422: [0], 174: [2], 432: [1], 186: [6], 442: [1, 3], 446: [6], 195: [2], 456: [0], 216: [4], 
473: [3], 474: [1], 477: [0], 225: [2], 230: [0], 233: [0], 495: [1], 498: [2], 499: [1]}
77

"""

### testing set wrong machine label cases
"""
{1: [0, 2, 3, 5], 2: [0], 4: [0], 5: [1], 261: [0], 7: [5], 264: [1], 8: [0, 3], 10: [0, 2], 
11: [0, 1, 3, 4], 268: [3], 17: [2], 273: [2], 21: [5], 277: [2], 22: [3], 26: [0], 30: [0], 
288: [1], 34: [1], 37: [2], 40: [4, 5], 53: [0], 312: [1], 57: [0], 56: [1], 59: [0, 1, 2, 3], 
60: [0], 320: [4], 65: [0], 66: [1, 3], 75: [0, 1, 3], 77: [0], 79: [0], 80: [0], 81: [1, 2], 
82: [2, 3], 84: [1], 88: [5, 7], 89: [0], 90: [0, 3], 349: [2], 98: [1, 4, 5], 99: [2], 101: [0, 6], 
105: [3], 112: [1, 2, 3, 4], 369: [3], 114: [0, 4], 115: [1, 4, 6], 117: [0], 375: [4], 376: [8, 1], 
379: [0], 123: [0], 383: [3], 127: [0, 3, 4, 5], 129: [3], 136: [4], 394: [1], 397: [1, 2], 150: [0], 
413: [1], 418: [2], 168: [1], 178: [0], 437: [10], 440: [2, 4], 192: [5], 455: [1], 201: [1], 468: [0], 
213: [0], 472: [3], 481: [3], 227: [0], 255: [3]}"""