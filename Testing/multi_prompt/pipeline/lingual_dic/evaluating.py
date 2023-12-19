import openai
import pickle
import numpy as np
import csv
import os
import re
import glob
from evaluate import load
import jieba
from Testing.annotation.metrics import qa_f1_zh_score, qa_f1_score, normalize_zh_answer, normalize_answer
from Testing.multi_prompt.pipeline.run import det_cor_process, Accuracy, Exact_Match, f1_score,MT_metrics



def lingual_evaluation(det_cor_dict, filename):

    with open('../../../human_label_data/all_cases', 'rb') as f:
        all_cases = pickle.load(f)

    # loading lingual pos
    with open('../../../human_label_data/lingual_pos', 'rb') as f:
        lingual_pos = pickle.load(f)

    # loading relative pos
    with open('relative_pos', 'rb') as f:
        relative_pos = pickle.load(f)

    lingual_human_detection = []
    lingual_human_correction = []
    for k, v in lingual_pos.items():
        k = str(k)
        for i in v:
            lingual_human_detection.append(all_cases[k]['detection'][i])
            lingual_human_correction.append(all_cases[k]['correction'][i])

    # # other methods
    special_set = ['bcot.npy', 'cot2.npy']
    if filename not in special_set:
        new_det_cor = {}
        for k, v in relative_pos.items():
            if len(v) > 0:
                new_det_cor[k] = [det_cor_dict[int(k)][p] for p in relative_pos[str(k)]]

    else:
        # bcot only
        new_det_cor = det_cor_dict



    processed = det_cor_process(new_det_cor)
    print(len(processed['det']))

    det_none_num = 0
    cor_none_num = 0
    for d, c in zip(processed['det'], processed['cor']):
        if d == 'None':
            det_none_num += 1
        if c == 'None':
            cor_none_num += 1

    print(f'filename: {filename}')
    print(f'detection none case: {det_none_num}/300, correction none case: {cor_none_num}/300')

    ### ACC
    # det_acc = Accuracy(processed['det'], lingual_human_detection)
    #
    #
    #
    # cor_acc = Accuracy(processed['cor'], lingual_human_correction)
    #
    # # print('det acc:', det_acc)
    # # print('cor acc:', cor_acc)
    #
    # ### EM
    # # exact_match_metric = load("exact_match")
    # # results = exact_match_metric.compute(predictions=processed['det'], references=lingual_human_detection)
    # det_EM = Exact_Match(processed['det'], lingual_human_detection)
    # # print('det EM: ', det_EM)
    # # print(det_EM)
    #
    # cor_EM = Exact_Match(processed['cor'], lingual_human_correction)
    # # print('cor EM: ', cor_EM)
    # # print(cor_EM)
    #
    # ### F1
    # det_f1 = f1_score(processed['det'], lingual_human_detection)
    # # print('det f1:', det_f1)
    # #
    # cor_f1 = f1_score(processed['cor'], lingual_human_correction)
    # # print('cor f1:', cor_f1)
    #
    # eval_dict = {'detection EM': det_EM, 'detection F1': det_f1, 'detection ACC': det_acc, 'correction EM': cor_EM,
    #              'correction F1': cor_f1, 'correction ACC': cor_acc}
    # print(eval_dict)
    #
    # ### MT metrics
    # det_mt_results = MT_metrics(processed['det'], lingual_human_detection)
    # cor_mt_results = MT_metrics(processed['cor'], lingual_human_detection)
    # print('det mt:', det_mt_results)
    # print('cor_mt:', cor_mt_results)


if __name__ == '__main__':
    files = glob.glob('*.npy')

    for f in files:
        print(f'filename {f}\n\n')
        lingual_dict = np.load(f, allow_pickle=True).item()
        lingual_evaluation(lingual_dict, f)











