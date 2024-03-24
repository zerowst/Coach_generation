import pickle
import csv
import re
import numpy as np



coach_input = []
with open('../training_data/coach_testing.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if row:
            coach_input.append(row[0])

coach_input = coach_input[1:]

with open('../annotation/human_detection_all', 'rb') as f:
    human_detection = pickle.load(f)

conversation_len = np.array([len(v) for v in human_detection.values()])
conv_len_dict = {k: len(v) for k, v in human_detection.items()}

def saving_file(input_, filename):
    with open('coach_data/' + str(filename) + '_coach' + '.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['input'])
        for c in input_:
            writer.writerow([c])


def get_disease(text):
    pattern = r"<medical context>:\s*(\S+)"

    match = re.search(pattern, text)
    disease = match.group(1)
    return disease

### head prompt + doctor + medical + Output: Your response (Generated response in Chinese)
def replace_prompt(prompt, text):
    start = text.find("""Provided medical context:""") + len("""Provided medical context:""")
    end = text.find('<dialogue history>:')
    start_ = text.find("""<Doctor's statement>:""")
    end_ = text.find("""Your response (Generated response in Chinese):""")

    new_text = prompt + text[start_: end_] + text[start:end] + """\nOutput: Your response (Generated response in Chinese):"""
    return new_text


### head prompt + doctor + medical
def slot_replace_prompt(prompt, text):
    start = text.find("""<medical context>:""") + len("""<medical context>:""")
    end = text.find('<dialogue history>:')
    start_ = text.find("""<Doctor's statement>:""") + len("""<Doctor's statement>:""")
    end_ = text.find("""Your response (Generated response in Chinese):""")
    prompt_tail = """\nShow me output directly.
    """
    new_text = prompt + """{Doctor's statement}:""" + text[start_:end_] +\
               """\n{Medical context}: """ + text[start:end]
    return new_text

def slot_filling_prompt(text):
    start = text.find("""<medical context>:""") + len("""<medical context>:""")
    end = text.find('<dialogue history>:')
    start_ = text.find("""<Doctor's statement>:""") + len("""<Doctor's statement>:""")
    end_ = text.find("""Your response (Generated response in Chinese):""")

    medical_context = text[start:end]
    doctor_sen = text[start_: end_]

    slot_filling = """Act as a linguistic and medical analyst. Assess the {doctor's statement} against the provided medical context: {medical context} and fill in the relevant slots based on your analysis. Follow these steps:
1. Identify <medical_words> in the {doctor's statement}.
2. If no <medical_words> are detected, mark it as detected: "no_medical_words_detected": true. Otherwise, mark it as false.
3. If "no_medical_words_detected" is false,  compare the <medical words> with the {medical context}.
4 If  <medical words> is aligned with the {medical context},  record the alignment status as true: "alignment": true. Otherwise, record it as false.
5 If they don't align: 
 a. Identify and list any symptoms that are mentioned incorrectly as "incorrect_symptoms", and note the correct symptoms from the medical context as "correct_symptoms".
b. Identify and list any diseases that are mentioned incorrectly as "incorrect_disease", and note the correct disease from the medical context as "correct_disease".
c. Identify and list any medications that are mentioned incorrectly as "incorrect_medications", and note the correct medication from the medical context as "correct_medications".

The output should follow these specific constraints:
"medical_words" encompasses any medical terminology found in the doctor's statement.
"no_medical_words_detected" is a boolean indicating the absence (true) or presence (false) of medical words.
"alignment" is a boolean indicating whether the medical words are consistent (true) or inconsistent (false) with the medical context.
"incorrect_symptoms", "incorrect_disease", and "incorrect_medications" are the terms from the  <medical_words> from {doctor's statement} that do not match the {medical context}.
"correct_symptoms", "correct_disease", and "correct_medications" are the terms from the {medical context} that should have been used if there were inaccuracies in the {doctor's statement}."

Input:
{doctor's statement}:""" + str(doctor_sen) + \
"""{medical's context}: """ + str(medical_context) + \
"""
The output should strictly follow the dictionary format as follows:
{ "medical_words": `<medical words>`, "no_medical_words_detected": true/false, "alignment": true/false, "incorrect_symptoms": `<incorrect symptoms>`, "incorrect_disease": `<incorrect disease>`, "incorrect_medications": `<incorrect medications>`, "correct_symptoms": `<correct symptoms>`, "correct_disease": `<correct disease>`, "correct_medications": `<correct medications>` } 

Show me output in above format directly"""

    return slot_filling


slot_prompt_second = """"Act as a linguistic coach for a physician. Assess the doctor's statement against a provided medical context and guide the physician if discrepancies arise. Your steps are:
1. Identify <medical words> in: {doctor's statement}.
2. If no <medical words> are detected, encourage the doctor.
3. Compare the <medical words> with the {medical context}.
4. If they align, provide positive feedback and provide further medical advice or knowledge.
5. If not, pinpoint inaccuracies like <incorrect symptoms>, <incorrect disease>, or <incorrect medications>. From the medical context, determine <correct symptoms>, <correct disease>, and <correct medication>.
6. Respond accordingly:
For symptoms: "Doctor, '' <incorrect symptoms>' is not a typical symptom for <correct disease>'. The symptoms of <correct disease>' usually include  '<correct symptoms>'."
For diagnosis: "Doctor, your diagnosis of <incorrect disease> is off. Given the symptoms, perhaps it's <correct disease>."
For medications: "Doctor, '<incorrect medication>' isn't suitable for <correct disease>. '<correct medication>' might be more apt."
Note:
Address all discrepancies in a single response.
placeholders (‘<>’) represent variables. After steps 1 and 5, ensure placeholders (‘<>’) are replaced with corresponding values.  
If any placeholders (‘<>’) are not replaced, please find the corresponding value from the dictionary: <dictionary> and go over step 1-6 again.
Provide responses in Chinese."


Provided medical context: 
<medical context>:
{medical_context}

<doctor's statement>:
{doctor_statement}

<dictionary>:
{dictionary}

Your response (Generated response in Chinese):
"""
def slot_filling_second(text):
    start = text.find("""<medical context>:""") + len("""<medical context>:""")
    end = text.find('<dialogue history>:')
    start_ = text.find("""<Doctor's statement>:""") + len("""<Doctor's statement>:""")
    end_ = text.find("""Your response (Generated response in Chinese):""")

    medical_context = text[start:end]
    doctor_sen = text[start_: end_]
    slot_prompt_second = """
Act as a linguistic coach for a physician. Assess the doctor's statement against a provided medical context and guide the physician if discrepancies arise. Your steps are:
1. Identify <medical words> in: {doctor's statement}.
2. If no <medical words> are detected, encourage the doctor.
3. Compare the <medical words> with the {medical context}.
4. If they align, provide positive feedback and provide further medical advice or knowledge.
5. If not, pinpoint inaccuracies like <incorrect symptoms>, <incorrect disease>, or <incorrect medications>. From the medical context, determine <correct symptoms>, <correct disease>, and <correct medication>.
6. Respond accordingly:
For symptoms: "Doctor, '' <incorrect symptoms>' is not a typical symptom for <correct disease>'. The symptoms of <correct disease>' usually include  '<correct symptoms>'."
For diagnosis: "Doctor, your diagnosis of <incorrect disease> is off. Given the symptoms, perhaps it's <correct disease>.
For medications: "Doctor, '<incorrect medication>' isn't suitable for <correct disease>. '<correct medication>' might be more apt.
Note:
Address all discrepancies in a single response.
placeholders (‘<>’) represent variables. After steps 1 and 5, ensure placeholders (‘<>’) are replaced with corresponding values.  
If any placeholders (‘<>’) are not replaced, please find the corresponding value from the dictionary: <dictionary> and go over step 1-6 again.
Provide responses in Chinese.

Provided medical context: 
{medical context}:
""" + str(medical_context) + \
                         """
                         {doctor's statement}:
                         """ + str(doctor_sen)
    # """
    #
    # <dictionary>:
    # """ + str(dictionary) + \
    # """
    #
    # Your response (Generated response in Chinese):
    # """
    return slot_prompt_second

# slot_coach_second = [slot_filling_second(t) for t in coach_input]
# with open('coach_data/slot_second_coach.csv', 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['input'])
#     for c in slot_coach_second:
#         writer.writerow([c])


# slot_coach_input = [slot_filling_prompt(t) for t in coach_input]
#
# with open('coach_data/slot_coach.csv', 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['input'])
#     for c in slot_coach_input:
#         writer.writerow([c])


# zeroshot_input = [replace_prompt(zeroshot_prompt, t) for t in coach_input]
# with open('coach_data/zero_coach.csv', 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['input'])
#     for c in zeroshot_input:
#         writer.writerow([c])


# cot_coach_input = [replace_prompt(cot_prompt, t) for t in coach_input]
#
#
# with open('coach_data/cot_coach.csv', 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['input'])
#     for c in cot_coach_input:
#         writer.writerow([c])


# slot_coach_input = [slot_replace_prompt(slot_prompt, t) for t in coach_input]
#
# with open('coach_data/slot_coach.csv', 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['input'])
#     for c in slot_coach_input:
#         writer.writerow([c])

# changed_coach_input = [replace_prompt(vannila_prompt, t) for t in coach_input]
#
# gcot_coach_input = [replace_prompt(gcot_prompt, t) for t in coach_input]

# disease_coach_input = [get_disease(t) for t in coach_input]
# disease_dict = {k: disease_coach_input[1+sum(conversation_len[0:i])] for i, k in enumerate(conv_len_dict.keys())}
# with open('disease_dic', 'wb') as f:
#     pickle.dump(disease_dict, f)


# with open('coach_data/vanilla_coach.csv', 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['input'])
#     for c in changed_coach_input:
#         writer.writerow([c])
#
#
# with open('coach_data/gcot_coach.csv', 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['input'])
#     for c in gcot_coach_input:
#         writer.writerow([c])


### vannila prompt


vannila_prompt = """As a linguistic coach for a physician, evaluate the doctor's statement against the given medical context. \
If there are discrepancies, guide the doctor. If not, provide positive feedback.
"""

### ICL
### ICL
# icl_prompt = """As a linguistic coach for a physician, evaluate the {doctor's statement} against the given {medical context}. \
# If there are discrepancies, guide the doctor. If not, provide positive feedback. Please answer in Chinese. \
# You can follow these examples.
#
# Examples:
# Input:
# {doctor's statement}:医生：感谢您的信任，病情资料我已详细阅读。根据您现有的资料，建议：可能是气管炎。 建议行气管镜检查，了解气管部情况。
#
# {medical context}: 咽喉炎   喉咙痛    发烧    咳嗽    头痛    呕吐    耳痛    鼻塞    皮疹    全身酸痛    吞咽困难    发冷   食欲不振   链球菌性咽炎 链球菌性扁桃体炎或链球菌性喉咙痛（俗称链球菌性咽喉炎）是由 A 组链球菌感染引起的一种咽炎。它会影响咽部，包括扁桃体，可能还会影响喉部。常见症状包括发烧 喉咙痛和淋巴结肿大。它是 37% 的儿童喉咙痛和 5-15% 的成人喉咙痛的原因。   流感病毒抗体检测   扁桃体切除术和/或腺样体切除术     阿莫西林   青霉素   头孢丙烯   青霉素 G 苄星青霉素（比西林）   苯酚外用   头孢羟氨苄   口服电解质替代溶液   异美汀粘酸盐
#
# Output: 教练：医生，根据病情资料和医学背景，您使用的术语气管炎不是很准确，正确的医学词汇应该是是咽喉炎。建议行电子咽喉镜检查，以了解咽喉部情况。
#
# Input:
# {doctor's statement}:医生：尿路高血压控制怎样。 应该肾性高血压，可以吃这些神经元，血压控制不好最好医院面诊调药。
#
# {medical context}:高血压   剧烈的胸痛 ， 头痛   高血压 (HTN) 或高血压，有时称为动脉高血压，是一种动脉血压升高的慢性疾病。这需要心脏比正常情况下更努力地工作以使血液在血管中循环。血压由收缩压和舒张压两种测量值概括，这取决于心脏肌肉在节拍之间是收缩（收缩）还是放松（舒张），分别等于最大压力和最小压力。正常静息血压在 100-140mmHg 收缩压（顶部读数）和 60-90mmHg 舒张压（底部读数）范围内。如果血压持续处于或高于 140/90 mmHg，则称存在高血压。   血液学测试（血液测试） ， 全血细胞计数（Cbc） ， 脂质面板 ， 葡萄糖测量（葡萄糖水平） ， 心电图 ， 血红蛋白 A1c 测量（血红蛋白 a1c 测试） ， 超声检查（超声波）     氢氯噻嗪   氨氯地平   奥美沙坦（贝尼卡）   贝那普利
#
# Output:教练：医生，您提到的病情似乎有一些误解。在高血压的背景下，使用'尿路高血压'这个词似乎不太合适。您可以考虑使用以下术语：'肾性高血压'。另外，在提到药物时，使用'神经元'似乎不是正确的术语。您可能需要考虑使用正确的药物名称。
#
# Input:
# {doctor's statement}:医生：左氧氟沙星眼水一天六次。
#
# {medical context}:麦粒肿   眼睛疼痛    眼睛肿胀    眼睛发红    眼睑肿胀    眼睛症状    眼睑肿块    眼睛发痒    眼睑病变或皮疹    异常出现皮肤   眼睛灼伤或刺痛   皮肤肿胀   流泪   外部麦粒肿或麦粒肿 /\xcb\x88sta\xc9\xaa/，也称为麦粒肿 /h\xc9\x94r\xcb\x88di\xcb\x90\xc9\x99l\xc9\x99m/，是皮脂腺的感染Zeis 在睫毛根部，或 Moll 的大汗腺感染。外部麦粒肿在眼睑外侧形成，可以看到是红色的小肿块。内部麦粒肿是眼睑内侧的睑板腺皮脂腺感染。它们还会在眼睑下方形成一个红色肿块，外部仅可见泛红和肿胀。麦粒肿与麦粒肿相似，但往往体积更小，疼痛更严重，而且通常不会产生持久的损伤。它们含有水和脓液，如果麦粒肿被用力弄破，细菌就会扩散。麦粒肿的特点是急性发作且持续时间通常较短（7\xe2\x80\x9310 天，未经治疗），而霰粒肿是慢性的，通常不经干预无法解决。麦粒肿通常由金黄色葡萄球菌引起。   物理治疗练习；操纵；和其他程序   切开和引流（I d）   眼科检查和评估（眼科检查）   非手术去除异物   培养伤口     红霉素 ， 红霉素眼药 ， 磺胺醋酸钠眼药 ， 庆大霉素眼药 ， 妥布霉素眼药 ， 妥布霉素（Tobi） ， 丁卡因（一触式） ， 地塞米松 - 妥布霉素眼药 ， 庆大霉素（庆大霉素）   四氢唑啉眼科   荧光素眼科
#
# Output:医生，您对于麦粒肿患者使用左氧氟沙星眼水一天六次的建议是恰当的。这显示了您对疾病的专业理解和治疗的精准把握。对于麦粒肿，确保患者严格遵守用药频率和疗程是非常重要的，以促进快速恢复。同时，提醒患者注意个人卫生，避免用手触摸眼部，这样可以减少感染的风险，并防止病情恶化。
#
# Now assess the doctor's statement: {doctor’s statement} against a provided medical context: {medical context} \
# and provide your response in Chinese based on above examples.
#
# """
# icl_coach_input = [replace_prompt(icl_prompt, t) for t in coach_input]
# with open('coach_data/icl_coach.csv', 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['input'])
#     for c in icl_coach_input:
#         writer.writerow([c])

### zeroshot
zeroshot_prompt = """As a linguistic coach for a physician, evaluate the doctor's statement against the given medical context. \
If there are discrepancies, guide the doctor. If not, provide positive feedback. Please answer in Chinese and think step by step.
"""

### COT
cot_prompt = """Act as a linguistic coach for a physician. Assess the doctor's statement: {a doctor’s statement sample}\
  against a provided medical context: {medical context sample}  and guide the physician if discrepancies arise. 

You thought: the medical words in the statement are: xxxx.  By comparing xxx with the medical context, \
I find that xxx is not align with the medical context. Since this is related to symptoms, it should be treated as incorrect symptoms,\
 then I should check the corresponding accurate symptoms is ***. 
So, you final response:  Doctor, 'xxx' isn't aligned with '<correct disease>'. Consider '<correct symptoms>

Now assess the doctor's statement: {doctor’s statement}  against a provided medical context: {medical context} and guide the physician if discrepancies arise. 

"""

### GCOT
gcot_prompt = """"Act as a linguistic coach for a physician. Assess the doctor's statement against a provided medical \
context and guide the physician if discrepancies arise. Your steps are:

1. Identify <medical words> in: {doctor's statement}.
2. If no <medical words> are detected, encourage the doctor.
3. Compare the <medical words> with the {medical context}.
4. If they align, provide positive feedback and provide further medical advice or knowledge.
5. If not, pinpoint inaccuracies like <incorrect symptoms>, <incorrect disease>, or <incorrect medications>. From the medical context, determine <correct symptoms>, <correct disease>, and <correct medication>.
6. Respond accordingly:
For symptoms: "Doctor, '<incorrect symptoms>' isn't aligned with '<correct disease>'. Consider '<correct symptoms>'."
For diagnosis: "Doctor, your diagnosis of <incorrect disease> is off. Given the symptoms, perhaps it's <correct disease>."
For medications: "Doctor, '<incorrect medication>' isn't suitable for <correct disease>. '<correct medication>' might be more apt."

Note:
Address all discrepancies in a single response.
After steps 1 and 5, ensure placeholders (‘<>’) are replaced.
Provide responses in Chinese."
"""

### vannila COT  hybrid.json
vcot_prompt = """Act as a linguistic coach for a physician. Assess the doctor's statement: {doctor’s statement}  against a provided medical context: {medical context}  and guide the physician if discrepancies arise. 
Your thought: the medical words in the statement are: xxxx.  By comparing xxx with the medical context,  if I find that xxx is not align with the {medical context}, since this is related to symptoms and it should be treated as incorrect symptoms, then I should check the corresponding accurate symptoms is ***. If I find that xxx is align with the {medical context}, then I should encourage the doctor and provide medical advice about xxx inside of the {doctor's statement}.
So, your final response:  
Doctor, 'xxx' isn't aligned with '<correct disease>'. Consider '<correct symptoms> 
Or something encouraging and medicl advice.
Note: <correct disease> and <correct symptoms> are extracted from {medical context}

Examples:
Input:
{doctor's statement}:医生：感谢您的信任，病情资料我已详细阅读。根据您现有的资料，建议：可能是气管炎。 建议行气管镜检查，了解气管部情况。

{medical context}: 咽喉炎   喉咙痛    发烧    咳嗽    头痛    呕吐    耳痛    鼻塞    皮疹    全身酸痛    吞咽困难    发冷   食欲不振   链球菌性咽炎 链球菌性扁桃体炎或链球菌性喉咙痛（俗称链球菌性咽喉炎）是由 A 组链球菌感染引起的一种咽炎。它会影响咽部，包括扁桃体，可能还会影响喉部。常见症状包括发烧 喉咙痛和淋巴结肿大。它是 37% 的儿童喉咙痛和 5-15% 的成人喉咙痛的原因。   流感病毒抗体检测   扁桃体切除术和/或腺样体切除术     阿莫西林   青霉素   头孢丙烯   青霉素 G 苄星青霉素（比西林）   苯酚外用   头孢羟氨苄   口服电解质替代溶液   异美汀粘酸盐  

Output: 教练：医生，根据病情资料和医学背景，您使用的术语气管炎不是很准确，正确的医学词汇应该是是咽喉炎。建议行电子咽喉镜检查，以了解咽喉部情况。

Input:
{doctor's statement}:医生：尿路高血压控制怎样。 应该肾性高血压，可以吃这些神经元，血压控制不好最好医院面诊调药。

{medical context}:高血压   剧烈的胸痛 ， 头痛   高血压 (HTN) 或高血压，有时称为动脉高血压，是一种动脉血压升高的慢性疾病。这需要心脏比正常情况下更努力地工作以使血液在血管中循环。血压由收缩压和舒张压两种测量值概括，这取决于心脏肌肉在节拍之间是收缩（收缩）还是放松（舒张），分别等于最大压力和最小压力。正常静息血压在 100-140mmHg 收缩压（顶部读数）和 60-90mmHg 舒张压（底部读数）范围内。如果血压持续处于或高于 140/90 mmHg，则称存在高血压。   血液学测试（血液测试） ， 全血细胞计数（Cbc） ， 脂质面板 ， 葡萄糖测量（葡萄糖水平） ， 心电图 ， 血红蛋白 A1c 测量（血红蛋白 a1c 测试） ， 超声检查（超声波）     氢氯噻嗪   氨氯地平   奥美沙坦（贝尼卡）   贝那普利  

Output:教练：医生，您提到的病情似乎有一些误解。在高血压的背景下，使用'尿路高血压'这个词似乎不太合适。您可以考虑使用以下术语：'肾性高血压'。另外，在提到药物时，使用'神经元'似乎不是正确的术语。您可能需要考虑使用正确的药物名称。

Input:
{doctor's statement}:医生：左氧氟沙星眼水一天六次。

{medical context}:麦粒肿   眼睛疼痛    眼睛肿胀    眼睛发红    眼睑肿胀    眼睛症状    眼睑肿块    眼睛发痒    眼睑病变或皮疹    异常出现皮肤   眼睛灼伤或刺痛   皮肤肿胀   流泪   外部麦粒肿或麦粒肿 /\xcb\x88sta\xc9\xaa/，也称为麦粒肿 /h\xc9\x94r\xcb\x88di\xcb\x90\xc9\x99l\xc9\x99m/，是皮脂腺的感染Zeis 在睫毛根部，或 Moll 的大汗腺感染。外部麦粒肿在眼睑外侧形成，可以看到是红色的小肿块。内部麦粒肿是眼睑内侧的睑板腺皮脂腺感染。它们还会在眼睑下方形成一个红色肿块，外部仅可见泛红和肿胀。麦粒肿与麦粒肿相似，但往往体积更小，疼痛更严重，而且通常不会产生持久的损伤。它们含有水和脓液，如果麦粒肿被用力弄破，细菌就会扩散。麦粒肿的特点是急性发作且持续时间通常较短（7\xe2\x80\x9310 天，未经治疗），而霰粒肿是慢性的，通常不经干预无法解决。麦粒肿通常由金黄色葡萄球菌引起。   物理治疗练习；操纵；和其他程序   切开和引流（I d）   眼科检查和评估（眼科检查）   非手术去除异物   培养伤口     红霉素 ， 红霉素眼药 ， 磺胺醋酸钠眼药 ， 庆大霉素眼药 ， 妥布霉素眼药 ， 妥布霉素（Tobi） ， 丁卡因（一触式） ， 地塞米松 - 妥布霉素眼药 ， 庆大霉素（庆大霉素）   四氢唑啉眼科   荧光素眼科  

Output:医生，您对于麦粒肿患者使用左氧氟沙星眼水一天六次的建议是恰当的。这显示了您对疾病的专业理解和治疗的精准把握。对于麦粒肿，确保患者严格遵守用药频率和疗程是非常重要的，以促进快速恢复。同时，提醒患者注意个人卫生，避免用手触摸眼部，这样可以减少感染的风险，并防止病情恶化。

Now assess the doctor's statement: {doctor’s statement}  against a provided medical context: {medical context}  and guide the physician if discrepancies arise. 
"""
# vcot_coach_input = [replace_prompt(vcot_prompt, t) for t in coach_input]
#
#
# with open('coach_data/hybrid_coach.csv', 'w', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['input'])
#     for c in vcot_coach_input:
#         writer.writerow([c])


### cot_slot

# FILE = 'cot_slot'
cot_slot_prompt = """
Instruction: 
Act as a linguistic and medical analyst. Assess the {Doctor's statement} against the provided {medical context} and infer each element of the dictionary based on your analysis. You should follow the samples about input, think steps and output.

Input: 

{doctor’s statement}: 医生：尿路高血压控制怎样。 应该肾性高血压，可以吃这些神经元，血压控制不好最好医院面诊调药。

{medical context}: 高血压   剧烈的胸痛 ， 头痛   高血压 (HTN) 或高血压，有时称为动脉高血压，是一种动脉血压升高的慢性疾病。这需要心脏比正常情况下更努力地工作以使血液在血管中循环。血压由收缩压和舒张压两种测量值概括，这取决于心脏肌肉在节拍之间是收缩（收缩）还是放松（舒张），分别等于最大压力和最小压力。正常静息血压在 100-140mmHg 收缩压（顶部读数）和 60-90mmHg 舒张压（底部读数）范围内。如果血压持续处于或高于 140/90 mmHg，则称存在高血压。   血液学测试（血液测试） ， 全血细胞计数（Cbc） ， 脂质面板 ， 葡萄糖测量（葡萄糖水平） ， 心电图 ， 血红蛋白 A1c 测量（血红蛋白 a1c 测试） ， 超声检查（超声波） 	氢氯噻嗪   氨氯地平   奥美沙坦（贝尼卡）   贝那普利  

Thinking steps: 

1.Identify <medical words> in the doctor's statement:
"尿路" (urinary tract)
"高血压" (hypertension)
"肾性高血压" (renal hypertension)
"神经元" (seems out of context)
"血压控制" (blood pressure control)
"医院面诊调药" (hospital face-to-face consultation and medication adjustment)

2.Check if medical words are detected:
Detected: true

3.Compare the <medical words> with the medical context:
Medical context includes "高血压", symptoms like "剧烈的胸痛", "头痛", and medications "氢氯噻嗪", "氨氯地平", "奥美沙坦（贝尼卡）", "贝那普利".

4.Record the alignment status:
"神经元" seems out of context, and no specific medications from the context are mentioned. Therefore, alignment should be false.

5.List discrepancies if any:
Incorrect symptoms: Not specifically mentioned in the doctor's statement.
Incorrect disease: Not applicable.
Incorrect medications: "神经元" (if interpreted as medication)
Correct symptoms: "剧烈的胸痛", "头痛"
Correct disease: "高血压"
Correct medications: "氢氯噻嗪", "氨氯地平", "奥美沙坦（贝尼卡）", "贝那普利"

Output:(Follow the dictionary format with value in Chinese): 
{
  "medical_words": "尿路, 高血压, 肾性高血压, 神经元, 血压控制, 医院面诊调药",
  "no_medical_words_detected": false,
  "alignment": false,
  "incorrect_symptoms": "",
  "incorrect_disease": "",
  "incorrect_medications": "神经元",
  "correct_symptoms": "剧烈的胸痛, 头痛",
  "correct_disease": "高血压",
  "correct_medications": "氢氯噻嗪, 氨氯地平, 奥美沙坦（贝尼卡）, 贝那普利"
}

Input:
{doctor’s statement}: 医生：感谢您的信任，病情资料我已详细阅读。根据您现有的资料，建议：考虑咽炎。 咽痛，咽部异物感有吗。

{medical context}: 咽炎   咳嗽   发烧   喉咙痛   气喘   吞咽困难   声音嘶哑   鼻炎   呼吸困难   腹痛   恶心   耳痛   鼻塞   咽炎 意思是喉咙，后缀 -itis 意思是炎症。这是喉咙发炎。在大多数情况下，这是非常痛苦的，并且是喉咙痛的最常见原因。   乳房检查	抑郁症筛查（抑郁症筛查）	雾化器治疗	切除（切除）	直肠检查	角膜移植	脾脏手术（脾脏手术） 	愈创甘油醚 (Mucinex)   Robitussin Dm   Pregabalin (Lyrica)   Prednisolone   Ipratropium   Doxycycline   Mecamylamine   Grepafloxacin (Raxar)   Malathion Topical   Gemtuzumab (Mylotarg) 

Thinking steps:
1.Identify <medical words> in the doctor's statement:
"咽炎" (pharyngitis)
"咽痛" (throat pain)
"咽部异物感" (foreign body sensation in the throat)

2.Check if medical words are detected:
Detected: true

3.Compare the <medical words> with the medical context:
Medical context for "咽炎" includes symptoms like "咳嗽" (cough), "发烧" (fever), "喉咙痛" (sore throat), "气喘" (asthma), "吞咽困难" (difficulty swallowing), "声音嘶哑" (hoarseness), "鼻炎" (rhinitis), "呼吸困难" (difficulty breathing), "腹痛" (abdominal pain), "恶心" (nausea), "耳痛" (ear pain), "鼻塞" (nasal congestion). Medications include "愈创甘油醚" (Mucinex), "Robitussin Dm", "Pregabalin (Lyrica)", "Prednisolone", "Ipratropium", "Doxycycline", "Mecamylamine", "Grepafloxacin (Raxar)", "Malathion Topical", "Gemtuzumab (Mylotarg)".

4.Record the alignment status:
"咽痛" (throat pain), "咽部异物感"(foreign body sensation in the throat) are not precise compared with {medical context}for ‘咽炎’. Besides, the doctor's statement doesn't include all the symptoms or any specific medications from the medical context. Therefore alignment is false.

5.List discrepancies if any:
Incorrect symptoms: "咽痛" (throat pain), "咽部异物感"(foreign body sensation in the throat)
Incorrect disease: Not applicable.
Incorrect medications: Not mentioned in the doctor's statement.
Correct symptoms: ‘喉咙痛， 吞咽困难’
Correct disease: Not mentioned
Correct medications: Not mentioned in the doctor's statement but include those listed in the medical context.

Output(Follow the dictionary format with value in Chinese):
{
  "medical_words": "咽炎, 咽痛, 咽部异物感",
  "no_medical_words_detected": false,
  "alignment": false,
  "incorrect_symptoms": "咽痛, 咽部异物感",
  "incorrect_disease": "",
  "incorrect_medications": "",
  "correct_symptoms": "喉咙痛, 吞咽困难",
  "correct_disease": "",
  "correct_medications": ""
}

Input:
{doctor's statement}:医生：感谢您的信任，病情资料我已详细阅读。根据您现有的资料，建议：甲硝唑有影响，你这个是阳性球菌感染，可以口服罗红霉素。然后清洗外阴。硝呋太尔片阴道上药可以。
{medical context}:阴道炎   白带    阴道瘙痒    腹痛    尿痛    阴道痛    下腹痛    盆腔痛    阴道发红    耻骨上痛    疼痛怀孕期间   外阴刺激   性交疼痛   阴道炎是阴道的炎症。它会导致分泌物 瘙痒和疼痛，并且通常与外阴刺激或感染有关。这通常是由于感染。阴道炎的三种主要类型是细菌性阴道病 (BV) 阴道念珠菌病和滴虫病。一名女性可能同时感染多种阴道感染。出现的症状因感染而异，尽管所有阴道炎感染都有一般症状，而且受感染的女性也可能没有症状。阴道感染检测不是常规骨盆检查的一部分；因此，妇女不应假设她们的医疗保健提供者会知道感染，也不应假设她们会在没有她们的意见的情况下提供适当的治疗。   骨盆检查   尿液分析   标准妊娠试验   衣原体检查   乳房检查   显微镜检查（细菌涂片；培养；毒理学）   子宫颈抹片检查     甲硝唑   氟康唑（大扶康）   甲硝唑外用产品   特康唑外用   咪康唑外用产品   倍他米松-克霉唑外用   克林霉素外用产品   替硝唑   制霉菌素-曲安西龙外用   头孢克肟 (Suprax)   戊聚糖多硫酸钠 (Elmiron)   外用硼酸

Thinking steps:
1.Identify <medical words> in the doctor's statement:
"甲硝唑" (Metronidazole)
"阳性球菌感染" (Gram-positive cocci infection)
"口服罗红霉素" (oral Erythromycin)
"清洗外阴" (cleaning the vulva)
"硝呋太尔片" (Nitrofurantoin tablets)
"阴道上药" (vaginal medication)

2.Check if medical words are detected:
Detected: true

3.Compare the <medical words> with the medical context:
Medical context for "阴道炎" (vaginitis) includes symptoms like "白带" (vaginal discharge), "阴道瘙痒" (vaginal itching), "腹痛" (abdominal pain), and medications like "甲硝唑" (Metronidazole), "氟康唑" (Fluconazole), "甲硝唑外用产品" (Topical Metronidazole products), "特康唑外用" (Topical Tioconazole), and others.

4.Record the alignment status:
There is partial alignment. The doctor's statement mentions Metronidazole, which is relevant to vaginitis. However, Gram-positive cocci infection, Erythromycin and Nitrofurantoin are not directly mentioned in the provided medical context for vaginitis, but they could be relevant depending on the specific type of infection.

5.List discrepancies if any:
Incorrect symptoms: "阳性球菌感染" (Gram-positive cocci infection) is not mentioned in medical context.
Incorrect disease: Not applicable.
Incorrect medications: "口服罗红霉素" (Erythromycin) and "硝呋太尔片" (Nitrofurantoin) are not mentioned in the medical context, but they may still be relevant for treatment.
Correct symptoms: Not mentioned in the doctor's statement.
Correct disease: Not applicable
Correct medications: "甲硝唑" (Metronidazole)

Output(Follow the dictionary format with value in Chinese):
{
  "medical_words": "甲硝唑, 阳性球菌感染, 口服罗红霉素, 清洗外阴, 硝呋太尔片, 阴道上药",
  "no_medical_words_detected": false,
  "alignment": true,
  "incorrect_symptoms": "阳性球菌感染",
  "incorrect_disease": "",
  "incorrect_medications": "口服罗红霉素, 硝呋太尔片",
  "correct_symptoms": "白带增多、阴道瘙痒和疼痛",
  "correct_disease": "",
  "correct_medications": "甲硝唑"
}


Input:
{doctor's statement}: 医生：别担心，积极治疗就好了。
{medical context}: 咽喉炎   喉咙痛    发烧    咳嗽    头痛    呕吐    耳痛    鼻塞    皮疹    全身酸痛    吞咽困难    发冷   食欲不振   链球菌性咽炎 链球菌性扁桃体炎或链球菌性喉咙痛（俗称链球菌性咽喉炎）是由 A 组链球菌感染引起的一种咽炎。它会影响咽部，包括扁桃体，可能还会影响喉部。常见症状包括发烧 喉咙痛和淋巴结肿大。它是 37% 的儿童喉咙痛和 5-15% 的成人喉咙痛的原因。   流感病毒抗体检测   扁桃体切除术和/或腺样体切除术     阿莫西林   青霉素   头孢丙烯   青霉素 G 苄星青霉素（比西林）   苯酚外用   头孢羟氨苄   口服电解质替代溶液   异美汀粘酸盐

Thinking steps:
1.Identify <medical words> in the doctor's statement:
The doctor's statement: "别担心，积极治疗就好了" translates to "Don't worry, active treatment will be fine."
There are no specific medical words or terms in this statement.

2.Check if medical words are detected:
Detected: false

3.Compare the <medical words> with the medical context:
Since there are no medical words detected in the doctor's statement, this step is not applicable.

4.Record the alignment status:
Not applicable as there are no medical words to compare.

5.List discrepancies if any:
Not applicable as there are no medical words in the doctor's statement.

Output(Follow the dictionary format with value in Chinese):
{
  "medical_words": "",
  "no_medical_words_detected": true,
  "alignment": false,
  "incorrect_symptoms": "",
  "incorrect_disease": "",
  "incorrect_medications": "",
  "correct_symptoms": "",
  "correct_disease": "",
  "correct_medications": ""
}

Now based on the above samples, you will be given {doctor’s statement} and {medical context}, show me thinking steps and output in dictionary format.

Input:
"""
def cot_slot_replace(prompt, text):
    start = text.find("""<medical context>:""") + len("""<medical context>:""")
    end = text.find('<dialogue history>:')
    start_ = text.find("""<Doctor's statement>:""") + len("""<Doctor's statement>:""")
    end_ = text.find("""Your response (Generated response in Chinese):""")
    doctor_statement = text[start_: end_]
    medical_context = text[start:end]

    new_text = prompt + """{doctor's statement}: """ + doctor_statement + """{medical context}: """ + medical_context
    return new_text


bcot_prompt = """Given the doctor's statement and the medical context provided:

Assess the Probability of Incorrect Terminology (P(Z₁)):

Analyze the terms used in the doctor's statement.
Estimate the probability that any given term is used incorrectly based on the medical context. Note that the medical context gives the information about the disease that the patient has been diagnosed.
List the terms along with their corresponding probability of being incorrect.
Identify Specific Errors (P(Z₂|Z₁)):

For terms with a high probability of being incorrect, identify the specific term(s) that are used inappropriately.
Provide a brief explanation for each identified error, referencing the medical context.
Determine Correction Requirement (P(Z₃|Z₂, Z₁)):

Based on the errors identified, decide if a correction is needed for each term.
For each term that requires correction, state the appropriate medical term that should be used.
Evaluate Contextual Misalignment (P(Z₄|Z₃, Z₂, Z₁)):

Judge how the incorrect use of terms affects the alignment of the doctor's statement with the medical context.
Assess the impact of these errors on the overall understanding of the medical situation.
Judge Diagnostic Accuracy (P(Z₅|Z₄, Z₃, Z₂, Z₁)):

Consider the implications of terminology errors on the accuracy of the diagnosis presented by the doctor.
Assess Treatment Suggestion Validity (P(Z₆|Z₅, Z₄, Z₃, Z₂, Z₁)):

Evaluate whether the treatment suggested by the doctor is still valid despite the terminology errors.
Determine Communication Clarity (P(Z₇|Z₆, Z₅, Z₄, Z₃, Z₂, Z₁)):

Analyze how the clarity of communication is affected by the incorrect terminology.
Estimate Potential for Misunderstanding (P(Z₈|Z₇, Z₆, Z₅, Z₄, Z₃, Z₂, Z₁)):

Estimate the likelihood of misunderstanding or miscommunication due to terminology errors.

For each step, provide your reasoning and the associated probabilities, if applicable, to mimic the process of Bayesian inference. 
Conclude by generating the coach feedback that assesses the doctor's statement against a provided medical context and \
guides the physician if discrepancies arise, if no mistakes occur, then encourage the doctor and provide further medical advice.  

Your provided response should be in Chinese and in the end you should provide coach response in the format:
教练:{coach response}

"""

bcot_prompt_ori = """Given the doctor's statement and the medical context provided:

Assess the Probability of Incorrect Terminology (P(Z₁)):

Analyze the terms used in the doctor's statement.
Estimate the probability that any given term is used incorrectly based on the medical context. Note that the medical context gives the information about the disease that the patient has been diagnosed.
List the terms along with their corresponding probability of being incorrect.
Identify Specific Errors (P(Z₂|Z₁)):

For terms with a high probability of being incorrect, identify the specific term(s) that are used inappropriately.
Provide a brief explanation for each identified error, referencing the medical context.
Determine Correction Requirement (P(Z₃|Z₂, Z₁)):

Based on the errors identified, decide if a correction is needed for each term.
For each term that requires correction, state the appropriate medical term that should be used.
Evaluate Contextual Misalignment (P(Z₄|Z₃, Z₂, Z₁)):

Judge how the incorrect use of terms affects the alignment of the doctor's statement with the medical context.
Assess the impact of these errors on the overall understanding of the medical situation.
Judge Diagnostic Accuracy (P(Z₅|Z₄, Z₃, Z₂, Z₁)):

Consider the implications of terminology errors on the accuracy of the diagnosis presented by the doctor.
Assess Treatment Suggestion Validity (P(Z₆|Z₅, Z₄, Z₃, Z₂, Z₁)):

Evaluate whether the treatment suggested by the doctor is still valid despite the terminology errors.
Determine Communication Clarity (P(Z₇|Z₆, Z₅, Z₄, Z₃, Z₂, Z₁)):

Analyze how the clarity of communication is affected by the incorrect terminology.
Estimate Potential for Misunderstanding (P(Z₈|Z₇, Z₆, Z₅, Z₄, Z₃, Z₂, Z₁)):

Estimate the likelihood of misunderstanding or miscommunication due to terminology errors.
For each step, provide your reasoning and the associated probabilities, if applicable, to mimic the process of Bayesian \
inference. Conclude by generating the coach feedback that assesses the doctor's statement against a provided medical context \
and guides the physician if discrepancies arise, if no mistakes occur, then encourage the doctor and provide further medical advice.
Coach feedback should be in Chinese. 
"""
file = 'bcot_ori'
# bcot_coach = [slot_replace_prompt(bcot_prompt_ori, t) for t in coach_input]
# saving_file(bcot_coach, file)


### chain of thought with four examples containing thinking steps
file = 'cot2'
prompt = """Act as a linguistic coach for a physician. Assess the doctor's statement: {doctor’s statement}  against a provided\
 medical context: {medical context}. You should provide your response based on the following examples of input, \
 thinking steps and output.

Examples:

{Input}:
{doctor’s statement}: 医生：尿路高血压控制怎样。 应该肾性高血压，可以吃这些神经元，血压控制不好最好医院面诊调药。
{medical context}: 高血压   剧烈的胸痛 ， 头痛   高血压 (HTN) 或高血压，有时称为动脉高血压，是一种动脉血压升高的慢性疾病。这需要心脏比正常情况下更努力地工作以使血液在血管中循环。血压由收缩压和舒张压两种测量值概括，这取决于心脏肌肉在节拍之间是收缩（收缩）还是放松（舒张），分别等于最大压力和最小压力。正常静息血压在 100-140mmHg 收缩压（顶部读数）和 60-90mmHg 舒张压（底部读数）范围内。如果血压持续处于或高于 140/90 mmHg，则称存在高血压。   血液学测试（血液测试） ， 全血细胞计数（Cbc） ， 脂质面板 ， 葡萄糖测量（葡萄糖水平） ， 心电图 ， 血红蛋白 A1c 测量（血红蛋白 a1c 测试） ， 超声检查（超声波） 	氢氯噻嗪   氨氯地平   奥美沙坦（贝尼卡）   贝那普利  

{thinking steps}:
作为一名针对医生的医疗指导教练，我会根据提供的医疗背景和医生的陈述来进行指导。以下是我的分步回应和指导：

核对医疗背景与医生陈述：

医疗背景：高血压（HTN）或称为动脉高血压，是一种动脉血压升高的慢性疾病。正常静息血压在100-140mmHg收缩压和60-90mmHg舒张压范围内。持续处于或高于140/90mmHg的血压被认为是高血压。治疗方法包括药物治疗（如氢氯噻嗪、氨氯地平、奥美沙坦或贝那普利）和生活方式的改变。重要的检测项目包括血液学测试、全血细胞计数、脂质面板、葡萄糖测量、心电图和超声检查等。

医生的陈述：医生提到“尿路高血压控制怎样。应该肾性高血压，可以吃这些神经元，血压控制不好最好医院面诊调药。”

分析：

医生提到的“尿路高血压”可能是指肾性高血压，这是高血压的一种特殊类型，与肾脏健康状况有关。
他提议的“吃这些神经元”表述不够明确，可能是指药物治疗，但需要具体化。
提到“血压控制不好最好医院面诊调药”是合理的，表示对病情的动态管理和调整治疗方案的重要性。
回应：

针对医生的陈述，建议如下：

明确“肾性高血压”的诊断需要通过具体的检测，如肾功能测试和肾脏超声波检查。
在提及药物治疗时，应详细说明具体的药物名称和剂量，以便于更准确的治疗和沟通。
您提到的“面诊调药”是非常重要的，持续监测和根据病情调整治疗方案是高血压管理的关键。

{Output}: 
医生，您提到的病情似乎有一些误解。在高血压的背景下，使用'尿路高血压'这个词似乎不太合适。您可以考虑使用以下术语：'肾性高血压'。另外，在提到药物时，使用'神经元'似乎不是正确的术语。您可能需要考虑使用正确的药物名称。


{Input}:
{doctor's statement}: 医生：别担心，积极治疗就好了。
{medical context}: 咽喉炎   喉咙痛    发烧    咳嗽    头痛    呕吐    耳痛    鼻塞    皮疹    全身酸痛    吞咽困难    发冷   食欲不振   链球菌性咽炎 链球菌性扁桃体炎或链球菌性喉咙痛（俗称链球菌性咽喉炎）是由 A 组链球菌感染引起的一种咽炎。它会影响咽部，包括扁桃体，可能还会影响喉部。常见症状包括发烧 喉咙痛和淋巴结肿大。它是 37% 的儿童喉咙痛和 5-15% 的成人喉咙痛的原因。   流感病毒抗体检测   扁桃体切除术和/或腺样体切除术     阿莫西林   青霉素   头孢丙烯   青霉素 G 苄星青霉素（比西林）   苯酚外用   头孢羟氨苄   口服电解质替代溶液   异美汀粘酸盐

{thinking steps}:
医疗情境分析及医生陈述：

医疗情境: 患者出现咽喉炎症状，包括喉咙痛、发烧、咳嗽、头痛、呕吐、耳痛、鼻塞、皮疹、全身酸痛、吞咽困难、发冷、食欲不振。这些症状表明患者可能患有链球菌性咽炎或链球菌性扁桃体炎，通常与A组链球菌感染相关。治疗方法可能包括流感病毒抗体检测、扁桃体切除术、阿莫西林、青霉素、头孢丙烯等药物治疗。

医生陈述: “别担心，积极治疗就好了。”

分析:
在这个情境中，医生的陈述基本符合医疗情境。医生鼓励患者不必过度担心，并强调了积极治疗的重要性。考虑到患者的症状和可能的链球菌感染，积极的治疗方法确实是关键。

无矛盾性：提供支持和相关医疗建议

鉴于医生的陈述与医疗情境相符，我建议您继续鼓励患者保持积极态度。同时，可以进一步解释治疗方法的重要性和效果，比如使用抗生素（如阿莫西林或青霉素）来对抗链球菌感染。此外，建议监测患者的症状变化，并在必要时调整治疗方案。若症状持续或加重，可能需要考虑进一步的检查或治疗，如扁桃体切除术。

{Output}:
医生，您的指导与当前医疗情境相符，您鼓励患者不必过度担心，同时强调了积极治疗的重要性。继续关注患者的反应和恢复情况，根据需要调整治疗方案，是保证患者健康的关键。您的专业态度和关怀对患者的恢复至关重要。

{Input}:
{doctor's statement}:医生：感谢您的信任，病情资料我已详细阅读。根据您现有的资料，建议：可能是气管炎。 建议行气管镜检查，了解气管部情况。

{medical context}: 咽喉炎   喉咙痛    发烧    咳嗽    头痛    呕吐    耳痛    鼻塞    皮疹    全身酸痛    吞咽困难    发冷   食欲不振   链球菌性咽炎 链球菌性扁桃体炎或链球菌性喉咙痛（俗称链球菌性咽喉炎）是由 A 组链球菌感染引起的一种咽炎。它会影响咽部，包括扁桃体，可能还会影响喉部。常见症状包括发烧 喉咙痛和淋巴结肿大。它是 37% 的儿童喉咙痛和 5-15% 的成人喉咙痛的原因。   流感病毒抗体检测   扁桃体切除术和/或腺样体切除术     阿莫西林   青霉素   头孢丙烯   青霉素 G 苄星青霉素（比西林）   苯酚外用   头孢羟氨苄   口服电解质替代溶液   异美汀粘酸盐  

{thinking steps}:
在这种情况下，您作为医疗教练需要帮助医生更准确地理解和处理病情。下面是对医生陈述和医疗背景的分析，以及相应的回应：

医疗背景和医生陈述：

医疗背景：患者表现出咽喉炎的症状，如喉咙痛、发烧、咳嗽、头痛、呕吐、耳痛、鼻塞、皮疹、全身酸痛、吞咽困难、发冷、食欲不振等。病因可能是链球菌性咽炎或链球菌性扁桃体炎，治疗包括抗生素（如青霉素）和对症治疗。
医生陈述：医生基于提供的病情资料，认为可能是气管炎，并建议进行气管镜检查。
分析：

存在的差异：医生的初步诊断是气管炎，这与病人展现的主要症状及潜在的链球菌性咽炎诊断不一致。气管炎通常表现为持续性咳嗽、呼吸困难等症状，而患者的主要症状是喉咙痛、发烧等，更符合咽喉炎或链球菌感染的特征。
回应：

指导医生重新考虑诊断：建议医生考虑链球菌性咽炎或链球菌性扁桃体炎的可能性，特别是考虑到患者的主要症状。建议进行相关的检测，如喉部拭子培养和抗链球菌抗体测试，以进一步确认诊断。
建议治疗方案的调整：鉴于当前的症状和可能的病因，建议采用适当的抗生素治疗（如青霉素或头孢类药物），并对症治疗以缓解患者的不适。

{Output}: 
医生，根据病情资料和医学背景，您使用的术语气管炎不是很准确，正确的医学词汇应该是是咽喉炎。建议行电子咽喉镜检查，以了解咽喉部情况。


{Input}:
{doctor's statement}:医生：左氧氟沙星眼水一天六次。
{medical context}:麦粒肿   眼睛疼痛    眼睛肿胀    眼睛发红    眼睑肿胀    眼睛症状    眼睑肿块    眼睛发痒    眼睑病变或皮疹    异常出现皮肤   眼睛灼伤或刺痛   皮肤肿胀   流泪   外部麦粒肿或麦粒肿，也称为麦粒肿 ，是皮脂腺的感染Zeis 在睫毛根部，或 Moll 的大汗腺感染。外部麦粒肿在眼睑外侧形成，可以看到是红色的小肿块。内部麦粒肿是眼睑内侧的睑板腺皮脂腺感染。它们还会在眼睑下方形成一个红色肿块，外部仅可见泛红和肿胀。麦粒肿与麦粒肿相似，但往往体积更小，疼痛更严重，而且通常不会产生持久的损伤。它们含有水和脓液，如果麦粒肿被用力弄破，细菌就会扩散。麦粒肿的特点是急性发作且持续时间通常较短（7-10 天，未经治疗），而霰粒肿是慢性的，通常不经干预无法解决。麦粒肿通常由金黄色葡萄球菌引起。   物理治疗练习；操纵；和其他程序   切开和引流（I d）   眼科检查和评估（眼科检查）   非手术去除异物   培养伤口     红霉素 ， 红霉素眼药 ， 磺胺醋酸钠眼药 ， 庆大霉素眼药 ， 妥布霉素眼药 ， 妥布霉素（Tobi） ， 丁卡因（一触式） ， 地塞米松 - 妥布霉素眼药 ， 庆大霉素（庆大霉素）   四氢唑啉眼科   荧光素眼科  

{thinking steps}:
根据所提供的医疗背景和医生的陈述，我们可以进行以下步骤的分析和指导：

对照医疗背景与医生陈述：
医疗背景：病人患有麦粒肿，表现为眼睛疼痛、肿胀、发红、眼睑肿胀等症状。麦粒肿通常由金黄色葡萄球菌引起，治疗方法包括物理治疗、药物治疗（例如红霉素、庆大霉素等眼药水）以及在必要时进行切开和引流。
医生陈述：医生建议使用左氧氟沙星眼水一天六次。

评估是否存在差异：
在提到的治疗方法中，虽然没有特别提到左氧氟沙星眼水，但它是一种广谱抗生素，通常用于治疗由细菌引起的眼部感染，包括麦粒肿。
因此，医生建议的治疗方案似乎是合理的，没有明显的差异。

提供指导和鼓励：
指导：确认医生对于病情的评估是否全面，例如是否有进行眼科检查和评估，以确保治疗方案的适宜性。同时，提醒注意观察患者对左氧氟沙星眼水的反应，如有任何异常反应应立即调整治疗方案。
鼓励：您选择的治疗方案是恰当的，左氧氟沙星眼水能有效针对麦粒肿的病原体。继续观察病人的反应和恢复情况，并根据需要调整治疗频率或药物种类。

{Output}:
医生，您对于麦粒肿患者使用左氧氟沙星眼水一天六次的建议是恰当的。这显示了您对疾病的专业理解和治疗的精准把握。对于麦粒肿，确保患者严格遵守用药频率和疗程是非常重要的，以促进快速恢复。同时，提醒患者注意个人卫生，避免用手触摸眼部，这样可以减少感染的风险，并防止病情恶化。

You will be given {Input}. Follow the above examples to provide your {thinking steps} and {Output} in Chinese.
{Input}:
"""



### bcot_end_note new one
prompt = """Given the doctor's statement and the medical context provided:

Assess the Probability of Incorrect Terminology (P(Z₁)):

Analyze the terms used in the doctor's statement.
Estimate the probability that any given term is used incorrectly based on the medical context.
List the terms along with their corresponding probability of being incorrect.
Identify Specific Errors (P(Z₂|Z₁)):

For terms with a high probability of being incorrect, identify the specific term(s) that are used inappropriately.
Provide a brief explanation for each identified error, referencing the medical context.
Determine Correction Requirement (P(Z₃|Z₂, Z₁)):

Based on the errors identified, decide if a correction is needed for each term.
For each term that requires correction, state the appropriate medical term that should be used.
Evaluate Contextual Misalignment (P(Z₄|Z₃, Z₂, Z₁)):

Judge how the incorrect use of terms affects the alignment of the doctor's statement with the medical context.
Assess the impact of these errors on the overall understanding of the medical situation.
Judge Diagnostic Accuracy (P(Z₅|Z₄, Z₃, Z₂, Z₁)):

Consider the implications of terminology errors on the accuracy of the diagnosis presented by the doctor.
Assess Treatment Suggestion Validity (P(Z₆|Z₅, Z₄, Z₃, Z₂, Z₁)):

Evaluate whether the treatment suggested by the doctor is still valid despite the terminology errors.
Determine Communication Clarity (P(Z₇|Z₆, Z₅, Z₄, Z₃, Z₂, Z₁)):

Analyze how the clarity of communication is affected by the incorrect terminology.
Estimate Potential for Misunderstanding (P(Z₈|Z₇, Z₆, Z₅, Z₄, Z₃, Z₂, Z₁)):

Estimate the likelihood of misunderstanding or miscommunication due to the terminology errors.
                                                                                                                                                                                                       
For each step, provide your reasoning and the associated probabilities (give real numbers ranging from 0 to 1)  , if applicable, to mimic the process of Bayesian inference.                                        
Conclude by generating the coach feedback (in Chinese) that assesses the doctor's statement against a provided medical \
context and guides the physician by pointing out the particular medical terminology errors and providing the \
corresponding corrections if discrepancies arise, if no mistakes occurred, then encouraging the doctor and provide \
further medical advice.
"""
file = 'bcot_end'


### bcot recap new one
prompt = """Given the {doctor's statement} and the {medical context} provided:

Assess the Probability of Incorrect Terminology (P(Z₁)):

Analyze the medical terms used in the {doctor's statement}.
Estimate the probability that any given medical term is used incorrectly based on the {medical context} which gives the information about the disease that the patient has been diagnosed with.
List the medical terms along with their corresponding numerical probability of being incorrect.
Identify Specific Errors (P(Z₂|Z₁)):

For medical terms with a high probability of being incorrect, identify the specific term(s) that are used inappropriately.
Provide a brief explanation for each identified error, referencing the {medical context}.
List the incorrect medical terms along with their identified errors and numerical probability.
Determine Correction Requirement (P(Z₃|Z₂, Z₁)):

Based on the errors identified, decide if a correction is needed for each term.
For each term that requires correction, state the appropriate medical term extracted from {medical context} that should be used.
Evaluate Contextual Misalignment (P(Z₄|Z₃, Z₂, Z₁)):

Judge how the incorrect use of medical terms affects the alignment of the doctor's statement with the {medical context}.
List the incorrect medical terms along with their numerical probability of affecting alignment of {doctor’s statement}.
Assess the impact of these errors on the overall understanding of the medical situation.
Judge Diagnostic Accuracy (P(Z₅|Z₄, Z₃, Z₂, Z₁)):

Consider the implications of terminology errors on the accuracy of the diagnosis presented by the doctor.
Assess Treatment Suggestion Validity (P(Z₆|Z₅, Z₄, Z₃, Z₂, Z₁)):

Evaluate whether the treatment suggested by the doctor is still valid despite the terminology errors.
Determine Communication Clarity (P(Z₇|Z₆, Z₅, Z₄, Z₃, Z₂, Z₁)):

Analyze how the clarity of communication is affected by the incorrect terminology.
Estimate Potential for Misunderstanding (P(Z₈|Z₇, Z₆, Z₅, Z₄, Z₃, Z₂, Z₁)):

Estimate the likelihood of misunderstanding or miscommunication due to the terminology errors.
                                                                                                                                                                                                       
For each step, provide your reasoning and the associated probabilities (give real numbers ranging from 0 to 1) , if applicable, to mimic the process of Bayesian inference.                                        
Conclude by generating the coach feedback (in Chinese) that assesses the {doctor's statement} against a provided {medical context} following all above reasoning steps.
If no mistakes occurred, then encouraging the doctor and provide further medical advice.
If discrepancies arise, guides the physician by pointing out the particular medical terminology errors and providing the corresponding corrections. Make your coach feedback natural in conversation.

"""
file = 'bcot_recap_new'

### bcot short delete irrelevant reasoning step, only detection and correction parts are left
prompt = """Given the {doctor's statement} and the {medical context} provided:

Assess the Probability of Incorrect Terminology (P(Z₁)):

Analyze the medical terms used in the {doctor's statement}.
Estimate the probability that any given medical term is used incorrectly based on the {medical context}. If medical term is irrelevant to {medical context} then it was considered incorrect.
List the medical terms along with their corresponding numerical probability of being incorrect.
Identify Specific Errors (P(Z₂|Z₁)):

For medical terms with a high probability of being incorrect, identify the specific term(s) that are used inappropriately.
Provide a brief explanation for each identified error, referencing the {medical context}.
Determine Correction Requirement (P(Z₃|Z₂, Z₁)):

Based on the errors identified, decide if a correction is needed for each term.
For each term that requires correction, state the appropriate medical term extracted from {medical context} that should be used.

For each step, provide your reasoning and the associated probabilities (give real numbers ranging from 0 to 1)  , if applicable, to mimic the process of Bayesian inference.       
                                 
Conclude by generating the coach feedback (in Chinese) that assesses the doctor's statement against a provided medical \
context and guides the physician by pointing out the particular medical terminology errors and providing the corresponding \
corrections if discrepancies arise, if no mistakes occurred, then encouraging the doctor and provide further medical advice.  \

"""
file = 'bcot_short'


coach = [slot_replace_prompt(prompt, t) for t in coach_input]
saving_file(coach, file)
exit()

















