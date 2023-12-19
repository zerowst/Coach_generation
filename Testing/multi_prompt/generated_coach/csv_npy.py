import csv
import numpy as np
import pickle
from sklearn.utils import shuffle


def loading_coach(filename):
    FILENAME = filename + '.csv'
    coach = []
    with open(FILENAME, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                coach.append(row)
    coach = coach[1:]
    return coach


def process_all(coach, file):
    eval_key = np.load('../../human_label_data/eval_keys.npy')
    with open('../../human_label_data/human_detection_all_old', 'rb') as f:
        sample = pickle.load(f)

    new_coach = {}
    start_idx = 0
    end_idx = 0
    for key, v in sample.items():
        l = len(v)
        end_idx += l
        new_coach[key] = coach[start_idx:end_idx]
        start_idx = end_idx

    new_coach = {k: new_coach[k] for k in eval_key}
    np.save(file + '_generated.npy', new_coach)

def process_test(coach, file):
    eval_key = np.load('../../human_label_data/eval_keys.npy')
    with open('../../human_label_data/human_detection_all_old', 'rb') as f:
        sample = pickle.load(f)
    sample = {k: sample[k] for k in eval_key}
    new_coach = {}
    start_idx = 0
    end_idx = 0
    for key, v in sample.items():
        l = len(v)
        end_idx += l
        new_coach[key] = coach[start_idx:end_idx]
        start_idx = end_idx

    np.save(file + '_generated.npy', new_coach)

if __name__ == '__main__':
    FILE = 'lora_lr4'

    coach_dic = loading_coach(FILE)

    if len(coach_dic) == 2170:
        process_all(coach_dic, FILE)
    else:
        print(len(coach_dic))
        process_test(coach_dic, FILE)

    exit()
