import openai
import pickle
import numpy as np
import csv
import os
import re
from evaluate import load
from Testing.annotation.metrics import qa_f1_zh_score, qa_f1_score
from Testing.multi_prompt.pipeline.run import *


### results




if __name__ == '__main__':
    ### jzc
    openai.api_key = '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'

    FILENAME = 'vannila'

    FILE_PATH = '../generated_coach/' + FILENAME + '_generated.npy'
    COACH_DIC = np.load(FILE_PATH, allow_pickle=True).item()

    ### lingual
    LINGUAL_DICT = det_cor_gen(coach_dict=COACH_DIC, filename=FILENAME)
    lingual_evaluation(LINGUAL_DICT, filename=FILENAME)


    ### non lingual
    NON_LINGUAL_DICT = non_lingual_filtering(coach_dict=COACH_DIC, filename=FILENAME)
    non_lingual_evaluation(filter_result=NON_LINGUAL_DICT, filename=FILENAME)

    exit()