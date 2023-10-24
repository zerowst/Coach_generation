import numpy as np
import pickle
import csv


llama_coach = []
csv_file = 'gpt_coach.csv'
annotation_file = 'annotation_data/gpt_coach.npy'

with open('coach_data/'+csv_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)

    for i, row in enumerate(reader):
        if row:
            llama_coach.append(row)

llama_coach = llama_coach[1:]
# llama_coach = llama_coach[:500]
# print()

annotation = np.load(annotation_file)
print(len(annotation))
# annotation = annotation

with open('annotation_data/'+csv_file, 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    header = ['coach', 'annotation']
    writer.writerow(header)

    for coach, annota in zip(llama_coach, annotation):
        writer.writerow([coach[0], annota])


