import csv
import openai
import pickle
import numpy as np
import pandas as pd
import openai
import os


from Testing.multi_prompt.coach_gen.generate import *


def slot_filling_second(text, dictionary):
    prompt = str(text[0]) + \
    """\n<dictionary>:\n""" + str(dictionary) + """\n\nYour response (Generated response in Chinese):
    """
    return prompt




def process_input_slot_second(input_file):
    ### loading first phase
    slot_dict = np.load('../generated_coach/slot_generated.npy', allow_pickle=True).item()

    ### get lens of eval key
    with open('../../annotation/non_lingual/data/coach_dict/no_lora_coach_dict', 'rb') as f:
        dic = pickle.load(f)

    eval_key = np.load('../../human_label_data/eval_keys.npy')
    all_lens = {key: len(value) for key, value in dic.items()}

    input_dict = {}
    start_index = 0
    end_idx = 0
    for key in all_lens.keys():
        lens = all_lens[key]
        end_idx += lens

        input_dict[key] = input_file[start_index:end_idx]

        start_index = end_idx

    input_eval_dict = {}
    for k in eval_key:
        input_eval_dict[k] = []
        for t, d in zip(input_dict[k], slot_dict[k]):
            new_input = slot_filling_second(t, d)
            input_eval_dict[k].append([new_input])

    # input_eval_dict = {k: [slot_filling_second(t, slot_dict[k]) for t, d in zip(input_dict[k], slot_dict[k])] for k in eval_key}

    return input_eval_dict

if __name__ == '__main__':
    ### jzc
    openai.api_key = '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'

    FILENAME = 'slot_second'

    INPUT_FILE = '../coach_data/' + FILENAME + '_coach.csv'
    SAVE_PATH = '../generated_coach/' + FILENAME + '_generated.npy'

    input_list = load_input_file(INPUT_FILE)

    input_dict = process_input_slot_second(input_list)

    generate_coach(input_dict, SAVE_PATH)


