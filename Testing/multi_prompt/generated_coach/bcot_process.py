import csv
import openai
import pickle
import numpy as np
import pandas as pd
import openai
import os
import re

from Testing.multi_prompt.pipeline.run import det_cor_gen, gpt_call4, lingual_detect_prompt


# old model and revised prompt
def filtering(sen):
    lines = [l for l in sen.splitlines() if l.strip()]
    coach_idx = []
    for i, line in enumerate(lines):
        if line.startswith('教练:') or line.startswith('教练：') or line.startswith('教练反馈'):
            coach_idx.append(i)

    if len(coach_idx) > 0:
        coach_i = coach_idx[-1]
        # if not last line and ends with }
        if lines[coach_i].endswith('}'):
            if coach_i < len(lines)-1:
                # '教练：{coach response}' '教练:{coach response}' '教练：{教练的回应}'
                coach_response = lines[coach_i+1]
            elif coach_i == len(lines) - 1 and len(coach_idx) > 1:
                coach_response = lines[coach_idx[-2]]
            else:
                coach_response = lines[coach_i]
        else:
            coach_response = lines[coach_i]
    else:
        coach_response = 'None'

    if coach_response.strip().endswith('}'):
        coach_response = 'None'

    return coach_response

# new model with original prompt filtering
def filtering_ori(sen):
    sen_split = re.split('[:：]', sen)
    if len(sen_split) != 0:
        return sen_split[-1]

    else:
        lines = [l for l in sen.splitlines() if l.strip()]
        return lines[-1]

def filtering_none(sen):
    lines = [l for l in sen.splitlines() if l.strip()]
    if len(lines) < 12 or ('P' not in sen):
        return 'None'
    else:
        return lines[-1]



if __name__ == "__main__":
    bcot = np.load('bcot_generated.npy', allow_pickle=True).item()

    with open('../../human_label_data/lingual_pos', 'rb') as f:
        new_lingual_pos = pickle.load(f)

    new_lingual_pos = {int(key): value for key, value in new_lingual_pos.items()}


    bcot_coach = {}
    bcot_list = []
    none_num = 0
    for key, v_list in bcot.items():
        # key = int(key)
        bcot_coach[key] = []
        for v in v_list:

            # sen = bcot[key][v]
            sen = v
            coach = filtering_ori(sen)
            if coach == 'None':   # P not in sen: 366;  len < 12: 417
                none_num += 1
                print(key, v)
            bcot_coach[key].append(coach)
            bcot_list.append(coach)

    ## 163
    openai.api_key = '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'

    FILENAME = 'bcot'

    # FILE_PATH = '../generated_coach/' + FILENAME + '_generated.npy'
    COACH_DIC = bcot_coach

    ### lingual
    LINGUAL_DICT = det_cor_gen(coach_dict=COACH_DIC, filename=FILENAME)


    print(bcot)


