import json
import numpy as np
from tqdm import tqdm

with open('/home/aikusrv04/aiku/1.3B/HG/70B_dataset_evaluated.json', 'r') as f:
    dataset = json.load(f)

def eval_stat(dataset, target_score):
    score_list = []
    target_list = []
    for i in tqdm(range(len(dataset))):
        score_str = dataset[i]['점수']
        try:
            score = float(score_str)
        except ValueError:
            pass
        else:
            score_list.append(score)
            if score == target_score:
                target_list.append(dataset[i])
    
    return score_list, target_list


score_list, target_list = eval_stat(dataset, target_score=0.5)


score_count = np.unique(score_list, return_counts = True)
print(score_count)

for item in target_list:
    print("{")
    for key, value in item.items():
        print(f"\t'{key}': {value},")
    print("},")

