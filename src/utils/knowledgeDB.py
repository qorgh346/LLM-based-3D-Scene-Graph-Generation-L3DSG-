import pickle
import os
import json
# 파일 경로 지정
file_path = '/home/baebro/hojun_ws/LMM based 3D Scene Graph Generation(L3DSG)/L3DSG_New/knowledge'

tb = '/home/baebro/hojun_ws/LMM based 3D Scene Graph Generation(L3DSG)/L3DSG_New/LMM Prompting'

a = 'PriorObjectKnowledge_DB'
b = 'PriorRelationKnowledge_DB'
# 파일 읽기

# with open(os.path.join(tb,"rel_knowledge.json"), 'r') as file:
#     data = json.load(file)



temp_dict = {"scene1" : {"A-B-C" : [1,2,3,4] }}





with open(os.path.join(tb,"storage room rel_knowledge.json"), 'r') as file:
    data = json.load(file)

# 키 값을 사용하여 원하는 값을 찾기
key = 'your_desired_key'
value = data.get(key, 'Key not found in the database')

print(f"The value for the key '{key}' is: {value}")