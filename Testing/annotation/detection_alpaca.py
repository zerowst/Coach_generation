import numpy as np
import csv
import os
import pickle
import re
from docx import Document

###-----------------------------------------------------------------------------------------
# Extracting testing results from llama alpaca and separate labels into detection and correction





###-----------------------------------------------------------------------------------------



file = 'annotation_data/annotation_alpaca700.txt'

with open(file, 'r') as f:
    lines = f.readlines()

lines = [line for line in lines if line.strip()]

human_label = []
machine_label = []
line3 = 0
for i, line in enumerate(lines):
    if line.startswith('机器标注'):
        detect_correct = re.split('机器标注：|修改为', line)
        ### try to split line into three parts
        if len(detect_correct) == 3:
            human_label.append([detect_correct[1], detect_correct[2]])
            line3 += 1
        ### line has less than 3 elements
        elif len(detect_correct) < 3:

            ### human label is empty, move to next line. Do the same check and split
            if lines[i+1].startswith('错误术语'):
                detect_correct = re.split('错误术语：|修改为', lines[i+1])
                human_label.append([detect_correct[1], detect_correct[2]])

            else:
                print('short')
                print(i)
                print(lines[i - 1])
                print(line)
                print(lines[i + 1])

        else:
            print(i)
            print(lines[i - 1])
            print(line)
            print(lines[i + 1])
            print('long line', line)

print('line3', line3)
# print(human_label)
detection_label = []
correction_label = []
print(len(human_label))

symbols = r'[：。，、（）]|\b错误术语\b|\n'
for label in human_label:
    det_label = re.sub(symbols, '', label[0]).replace(' ', '')
    cor_label = re.sub(symbols, '', label[1]).replace(' ', '')
    detection_label.append(det_label)
    correction_label.append(cor_label)

print(detection_label[:10])
print(correction_label[:10])
print(len(detection_label))

# np.save('detection_label', detection_label)
# np.save('correction_label', correction_label)


### exact match F1




















