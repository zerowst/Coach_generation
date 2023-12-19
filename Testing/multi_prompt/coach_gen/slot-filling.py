import numpy as np
import pickle
import json
import re
from Testing.multi_prompt.pipeline.run import Exact_Match, f1_score


FILE = 'cot_slot'

LOAD_FILE = '../generated_coach/' + str(FILE) + '_generated.npy'
slot_re = np.load(LOAD_FILE, allow_pickle=True).item()
# print(slot_re[184][4])

### slot_generated
# slot_re[184][4] = """{
#   "medical_words": ["玉泽皮肤屏障修复乳", "雅漾三重滋润霜"],
#   "no_medical_words_detected": false,
#   "alignment": false,
#   "incorrect_symptoms": [],
#   "incorrect_disease": [],
#   "incorrect_medications": ["玉泽皮肤屏障修复乳", "雅漾三重滋润霜"],
#   "correct_symptoms": ["湿疹", "皮疹", "皮肤瘙痒", "皮肤异常", "皮肤干燥", "脱皮", "鳞屑或粗糙", "皮肤损伤", "咳嗽", "过敏反应", "痤疮或粉刺", "皮肤肿胀", "头皮出现不规则", "皮肤刺激", "疣", "湿疹或通常称为特应性皮炎（来自希腊语 ἔκζεμα，沸腾）是一种皮炎的形式，或表皮（皮肤的外层）的炎症。"],
#   "correct_disease": [],
#   "correct_medications": ["湿疹", "皮疹", "皮肤瘙痒", "皮肤异常", "皮肤干燥", "脱皮", "鳞屑或粗糙", "皮肤损伤", "咳嗽", "过敏反应", "痤疮或粉刺", "皮肤肿胀", "头皮出现不规则", "皮肤刺激", "疣", "湿疹或通常称为特应性皮炎（来自希腊语 ἔκζεμα，沸腾）是一种皮炎的形式，或表皮（皮肤的外层）的炎症。"]
# }"""


slot_decode = {}
for key, v in slot_re.items():
    slot_decode[key] = []

    for i, json_string in enumerate(v):
        pattern = r'\\x[0-9a-f]{2}'
        cleaned_json_string = re.sub(pattern, '', json_string)

        dict_pattern = r"\{.*?\}"
        dictionary_output = re.findall(dict_pattern, cleaned_json_string, re.DOTALL)
        try:
            dictionary_output = dictionary_output[0] if isinstance(dictionary_output, list) else dictionary_output
        except:
            print(key, i)
            print(json_string)
            print(dictionary_output)
        # Replace these sequences with an empty string
        try:
            slot_decode[key].append(json.loads(dictionary_output))
        except:
            slot_decode[key].append({})
            print(key, i)
            print(dictionary_output)

# np.save('slot_decode', slot_decode)

key_list = slot_decode[1][0].keys()
incorrect_keys = ['incorrect_symptoms', 'incorrect_disease', 'incorrect_medications']
correct_keys = ['correct_symptoms', 'correct_disease', 'correct_medications']


def filtering_non_incorrect(decode):
    incorrect_items = {}
    correct_items = {}
    for key, v in decode.items():
        incorrect_items[key] = []
        correct_items[key] = []

        for i, json_dic in enumerate(v):

            incorrect_item = ''
            correct_item = ''
            for in_k in incorrect_keys:
                if in_k in json_dic.keys():
                    json_incorrect_item = json_dic[in_k]
                    if isinstance(json_incorrect_item, list) and len(json_incorrect_item) > 0:
                        incorrect_string = ''.join(json_incorrect_item)
                        incorrect_item += incorrect_string
                    if isinstance(json_incorrect_item, str):
                        incorrect_string = json_incorrect_item
                        incorrect_item += incorrect_string

            for cor_k in correct_keys:
                if cor_k in json_dic.keys():
                    json_correct_item = json_dic[cor_k]
                    if isinstance(json_correct_item, list) and len(json_correct_item) > 0:
                        correct_string = ''.join(json_correct_item)
                        correct_item += correct_string
                    if isinstance(json_correct_item, str):
                        correct_string = json_correct_item
                        correct_item += correct_string

            incorrect_items[key].append(incorrect_item)
            correct_items[key].append(correct_item)

    return incorrect_items, correct_items

def filling_none(incorrect_dic):
    new_dic = {}
    for key, v in incorrect_dic.items():
        new_dic[key] = []
        for item in v:
            if item == '':
                new_dic[key].append('None')
            else:
                new_dic[key].append(item)

    return new_dic

def testing_with_human(incorrect_items, correction_items):
    with open('../../human_label_data/human_detection_all_old', 'rb') as f:
        human_detection = pickle.load(f)
    with open('../../human_label_data/human_correction_all_old', 'rb') as f:
        human_correction = pickle.load(f)
    with open('../pipeline/lingual_pos', 'rb') as f:
        lingual_pos = pickle.load(f)

    print('testing...')

    lingual_human_detection = []
    lingual_human_correction = []
    detect_items = []
    correct_items = []
    for k, v in lingual_pos.items():
        for i in v:
            lingual_human_detection.append(human_detection[k][i])
            lingual_human_correction.append(human_correction[k][i])
            detect_items.append(incorrect_items[k][i])
            correct_items.append(correction_items[k][i])

    symbols = r'[：。，,.、（）\n\s\[\]\.\\\'\'\'"或者,]'
    detect_items = [re.sub(symbols, '', str_det) for str_det in detect_items]
    correct_items = [re.sub(symbols, '', str_cor) for str_cor in correct_items]

    # lingual_human_detection = [re.sub(symbols, '', det) for det in lingual_human_detection]
    # lingual_human_correction = [re.sub(symbols, '', cor) for cor in lingual_human_correction]

    det_EM = Exact_Match(detect_items, lingual_human_detection)
    print('det EM')
    print(det_EM)
    cor_EM = Exact_Match(correct_items, lingual_human_correction)
    print('cor EM')
    print(cor_EM)

    ### F1
    det_f1 = f1_score(detect_items, lingual_human_detection)
    print('det f1:', det_f1)
    #
    cor_f1 = f1_score(correct_items, lingual_human_correction)
    print('cor f1:', cor_f1)




slot_incorrect_items, slot_correct_items = filtering_non_incorrect(slot_decode)
new_slot_incorrect_items = filling_none(slot_incorrect_items)
new_slot_correct_items = filling_none(slot_correct_items)

testing_with_human(new_slot_incorrect_items, new_slot_correct_items)

exit()















