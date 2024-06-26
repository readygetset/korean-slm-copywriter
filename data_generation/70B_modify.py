import json
import numpy as np
from tqdm import tqdm

with open('/home/aikusrv04/aiku/1.3B/HG/70B_dataset_evaluated.json', 'r') as f:
    score_dataset = json.load(f)

with open("/home/aikusrv04/aiku/1.3B/HG/generated_data/70B_dataset.json", 'r') as f:
    original_dataset = json.load(f)

print(len(original_dataset))

# # 제거할 index 
# index_list = []

# for i in tqdm(range(len(score_dataset))):
#     score_str = score_dataset[i]['점수']
#     try:
#         score = float(score_str)
#     except ValueError:
#         pass
#     else:
#         if score >= 0.5:
#             index_list.append(i)

# print(f"-- 총 데이터셋 개수 : {len(index_list)}")

# modified_list = []
# modified_path = '/home/aikusrv04/aiku/1.3B/HG/70B_dataset_modified.json'

# for index in tqdm(index_list):
#     modified_list.append(original_dataset[index])

# with open(modified_path, 'w', encoding='utf-8') as f:
#     json.dump(modified_list, f, indent=4, ensure_ascii=False)

