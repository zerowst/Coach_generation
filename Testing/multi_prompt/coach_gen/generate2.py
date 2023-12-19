import csv
import openai
import pickle
import numpy as np
import pandas as pd
import openai
import os
from Testing.multi_prompt.coach_gen.generate import load_input_file, process_input_file, generate_coach


def gpt_call(prompt, retry=0):
    try:
        coach_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            temperature=0.85
        )
        response = coach_response['choices'][0]['message']['content']

        return response

    except openai.OpenAIError as e:
        print(e)
        print('retry:', retry)
        return gpt_call(prompt, retry=retry+1)



if __name__ == '__main__':
    ## gmail slot
    openai.api_key = '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'

    FILENAME = 'cot2'
    SAVE_PATH = '../generated_coach/' + FILENAME + '_generated.npy'
    INPUT_FILE = '../coach_data/' + FILENAME + '_coach.csv'

    input_list = load_input_file(INPUT_FILE)

    input_dict = process_input_file(input_list)

    generate_coach(input_dict, SAVE_PATH, FILENAME)












