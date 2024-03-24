import openai
import numpy as np
import pickle
import pandas as pd

with open('../parsed_2020t', 'rb') as f:
    dic = pickle.load(f)

label = ['医生：', '病人：']

### Keys with dialogue and 'Diagnosis and suggestions'
dic_both = []
for key in dic.keys():
    if 'Dialogue' in dic[key].keys() and 'Diagnosis and suggestions' in dic[key].keys():
        dic_both.append(key)


diagnosis = []
for key in dic_both[:100]:
    pre = dic[key]['Diagnosis and suggestions'][0]
    diag = dic[key]['Diagnosis and suggestions'][1]
    diagnosis.append(diag)









