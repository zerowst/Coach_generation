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
# gcot abnormal cases:
# wrong case: key 98 value True, False
# False, False
# True, False
# True, False
# Processing: TF+1
# wrong case: key 197 value False, False
# True, False
# False, True
# False, False
# False, False
# False, True
# Processing: TT+1
# wrong case: key 436 value True, False
# False, True
# False, True
# TT TF FT FF
# Processing: TT+1
# final result[27, 270, 108, 208] 613
# Encouragement T ACC: (TT+TF)/ALL 300/616
# Medical advice: T ACC: (TT+FT)/ALL 137/288



# {'exact_match': 0.0844062947067239}

# det f1: 0.08668091168091167
# correction
# {'exact_match': 0.034334763948497854}
# cor f1: 0.039370662680490505



if __name__ == '__main__':
    ## gmail slot
    openai.api_key = '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'

    FILENAME = 'icl_slot'

    FILE_PATH = '../generated_coach/' + FILENAME + '_generated.npy'
    COACH_DIC = np.load(FILE_PATH, allow_pickle=True).item()

    ### lingual
    LINGUAL_DICT = det_cor_gen(coach_dict=COACH_DIC, filename=FILENAME)
    lingual_evaluation(LINGUAL_DICT, filename=FILENAME)


    ### non lingual
    NON_LINGUAL_DICT = non_lingual_filtering(coach_dict=COACH_DIC, filename=FILENAME)
    non_lingual_evaluation(filter_result=NON_LINGUAL_DICT, filename=FILENAME)

    exit()