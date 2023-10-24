import pickle
from docx import Document
import numpy as np

file_human = 'annotation_data/annotation_human.docx'
human_file = Document(file_human).paragraphs
human_lines = [para.text for para in human_file if para.text.strip()]

### 147
wrong_list = [6, 32, 97, 110, 147, 160, 163, 170, 216, 216, 216, 216, 233, 278, 325, 365, 402, 428, 430, 456, 473, 474, 477]


for i, line in enumerate(human_lines):
    # for wrong_l in wrong_list:
    if line.startswith('Case:'):
        index = line.split('Case: ')[1]




