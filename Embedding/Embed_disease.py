import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from openpyxl import load_workbook
import torch
import openai
import pickle

### gmail
# openai.api_key = ''


def get_embedding(text, model="text-embedding-ada-002", retry = 1, max_retry = 5):
    text = text.replace("\n", " ")
    try:
        embedding = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
        # print('Embedding successfully.')
        return embedding
    except openai.OpenAIError as e:
        if retry < max_retry:
            print(f'Embedding error: {e}, retry: {retry}')
            return get_embedding(text, model="text-embedding-ada-002", retry=retry+1)
        else:
            print(f'Embedding error retrying has reached limit.')
            exit(1)

cols = [1]
disease = pd.read_csv('disease.csv', header=None)
disease = disease.iloc[:, cols].astype(str).apply(' '.join, axis=1)
disease = disease[1:]

with open('../Dialogue_gen/coach/key_extraction/match_key_disease_dic20', 'rb') as f:
    match_dic = pickle.load(f)
match_disease = list(match_dic.values())
match_disease = np.unique(match_disease)
print(match_disease.shape)

#---------------------------------------------disease_name embedding----------------------------------------------------
# disease_embedding = []
# i = 0
# for t in disease:
#     disease_embedding.append(get_embedding(text=t))
#     i += 1
#     if i % 10 == 0:
#         print(i)
# disease_embedding = np.array(disease_embedding)
# np.save('disease_embedding_1', disease_embedding)


match_disease_embedding = []
i = 0
for d in match_disease:
    match_disease_embedding.append(get_embedding(text=d))
    i += 1
    if i % 10 == 0:
        print(i)

np.save('match_disease_embedding', match_disease_embedding)






