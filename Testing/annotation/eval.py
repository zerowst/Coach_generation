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











with open('human_detection', 'rb') as f:
    human_detection = pickle.load(f)

with open('human_correction', 'rb') as f:
    human_correction = pickle.load(f)

with open('machine_detection', 'rb') as f:
    lora_detection = pickle.load(f)

with open('machine_correction', 'rb') as f:
    lora_correction = pickle.load(f)

eval_key = np.load('eval_keys.npy')


def extract(keys, dic):
    list_out = []
    for key in keys:
        list_out += dic[key]

    return list_out

human_detection = extract(eval_key, human_detection)
human_correction = extract(eval_key, human_correction)
detection = extract(eval_key, lora_detection)
correction = extract(eval_key, lora_correction)

# detection = ['None'] * len(detection)
# correction = ['None'] * len(correction)

print('...')
none_list = []
for i, (det, cor) in enumerate(zip(human_detection, human_correction)):
    if not (det == 'None' and cor == 'None'):
        none_list.append(i)
# print(h_det_none/len(human_detection))
print(len(none_list), len(human_detection))
#
# exit()

human_detection = [human_detection[d] for d in none_list]
human_correction = [human_correction[c] for c in none_list]

detection = [detection[d] for d in none_list]
correction = [correction[c] for c in none_list]





def f1_score(pre, ref):
    score = 0
    lens = len(pre)
    if len(pre) == len(ref):
        for pre, ref in zip(pre, ref):
            score += qa_f1_zh_score(pre, ref)

    return score/lens

print(len(human_detection), len(detection))
print(human_detection[:5])
print(detection[:5])
print(human_correction[:5])
print(correction[:5])





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


