import numpy as np
import csv
import os
import pickle
import re
from docx import Document
###-----------------------------------------------------------------------------------------
# Extracting human label and separate labels into detection and correction





###-----------------------------------------------------------------------------------------


file_human = 'annotation_data/annotation_human.docx'
human_file = Document(file_human).paragraphs
human_lines = [para.text for para in human_file if para.text.strip()]

human_label = []
machine_label = []

for i, line in enumerate(human_lines):
    if line.startswith('人工标注'):
        detect_correct = re.split('人工标注：|修改为', line)
        ### try to split line into three parts
        if len(detect_correct) == 3:
            human_label.append([detect_correct[1], detect_correct[2]])

        ### line has less than 3 elements
        elif len(detect_correct) < 3:

            ### human label is empty, move to next line. Do the same check and split
            if human_lines[i+1].startswith('错误术语'):
                print(i)
                print(line)
                print(human_lines[i + 1])
                print('short line', line, human_lines[i + 1], human_lines[i - 1])
                detect_correct = re.split('错误术语：|修改为', human_lines[i+1])
                human_label.append([detect_correct[1], detect_correct[2]])
            elif human_lines[i-1].startswith('机器标注'):
                print(i)
                print(line)
                print(human_lines[i + 1])
                print('short line', line, human_lines[i + 1], human_lines[i - 1])
                detect_correct = re.split('机器标注：|修改为', human_lines[i-1])
                human_label.append([detect_correct[1], detect_correct[2]])
            else:
                print(i)
                print(line)
                print(human_lines[i + 1])
                print('short line',line, human_lines[i+1], human_lines[i-1])
        else:
            print(i)
            print(line)
            print(human_lines[i + 1])
            print('long line', line)


# print(human_label)
detection_label = []
correction_label = []
print(len(human_label))

symbols = r'[：。，、（）]|\b错误术语\b'
for label in human_label:
    det_label = re.sub(symbols, '', label[0]).replace(' ', '')
    cor_label = re.sub(symbols, '', label[1]).replace(' ', '')
    detection_label.append(det_label)
    correction_label.append(cor_label)

print(detection_label[:10])
print(correction_label[:10])
print(len(detection_label))
#
# np.save('detection_label', detection_label)
# np.save('correction_label', correction_label)























