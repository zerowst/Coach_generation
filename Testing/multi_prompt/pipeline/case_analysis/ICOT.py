import numpy as np
import pickle
import re
from Testing.multi_prompt.pipeline.run import extract_det_cor
import csv


### all info of conversation
with open('../../../human_label_data/all_cases_human_doc', 'rb') as f:
    all_conversation = pickle.load(f)

### slot_filling first phase result
slot_first_result = np.load('../../coach_gen/slot_decode.npy', allow_pickle=True).item()

### slot_filling second phase result, labeled by gpt4, changed into detection:xxx corrcetion:xxx
slot_second_result = np.load('../lingual_dic/slot_second.npy', allow_pickle=True).item()

### slot_filling generated coach sentence
slot_second_generated = np.load('../../generated_coach/slot_second_generated.npy', allow_pickle=True).item()


### human labels
with open('../../../human_label_data/human_detection_all_old', 'rb') as f:
    human_detection = pickle.load(f)
with open('../../../human_label_data/human_correction_all_old', 'rb') as f:
    human_correction = pickle.load(f)
with open('../lingual_pos', 'rb') as f:
    lingual_pos = pickle.load(f)

### processing slot second into {key: [det, det]} {key: [cor, cor]}
def process(dic):
    det_dic = {}
    cor_dic = {}
    for key, value in dic.items():
        det_dic[key] = []
        cor_dic[key] = []
        for v in value:
            det, cor = extract_det_cor(v)
            det_dic[key].append(det)
            cor_dic[key].append(cor)
    return det_dic, cor_dic

### find  positions of truth!=pre
def error_case_location(pos_dic, truth, pred):
    error_dic = {}
    for key in pos_dic:
        error_dic[key] = []
        for i, (tru, pre) in enumerate(zip(truth[key], pred[key])):
            if tru != pre:
                error_dic[key].append(i)

    error_dic = {key: value for key, value in error_dic.items() if value != []}
    return error_dic


def error_case_info(error_pos, human, slot, first, second):
    info = {}
    info_list = []
    for key, value in error_pos.items():
        info[key] = []
        for idx in value:
            info_dic = {'human': human[key][idx], 'doctor': all_conversation[key]['con']['d'][idx],
                        'coach': all_conversation[key]['con']['c'][idx], 'slot_coach': slot[key][idx],
                        'patient': all_conversation[key]['con']['p'][idx],
                        'slot_first': first[key][idx], 'slot_second': second[key][idx]}
            info[key].append(info_dic)

    for key in info.keys():
        info_list += info[key]
    return info_list



slot_second_det, slot_second_cor = process(slot_second_result)

detection_wrong_pos = error_case_location(lingual_pos, human_detection, slot_second_det)
detection_wrong_info = error_case_info(detection_wrong_pos, human_detection, slot_second_generated, slot_first_result, slot_second_result)

correction_wrong_pos = error_case_location(lingual_pos, human_correction, slot_second_cor)
correction_wrong_info = error_case_info(correction_wrong_pos, human_correction, slot_second_generated, slot_first_result, slot_second_result)

with open('detection_info.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ['human', 'doctor', 'coach', 'slot_coach', 'patient', 'slot_first', 'slot_second']
    writer.writerow(header)
    for error in detection_wrong_info:

        row = list(error.values())
        writer.writerow(row)

with open('correction_info.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ['human', 'doctor', 'coach', 'slot_coach', 'patient', 'slot_first', 'slot_second']
    writer.writerow(header)
    for error in correction_wrong_info:
        row = list(error.values())
        writer.writerow(row)

print(detection_wrong_info)

















