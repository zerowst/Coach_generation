import jieba
import numpy as np
from Testing.annotation.metrics import normalize_zh_answer


a = """医生，关于预后的描述可能有误。根据提供的医学背景，None肝癌的预后一般较差，但并不是每个病人都只能存活6个月左右。您可以提供更加准确的信息，例如存活时间的范围。"""
a = normalize_zh_answer(a)
c = list(jieba.cut(a, cut_all=False))
print(c)






