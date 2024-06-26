from openai import OpenAI
import json
from tqdm import tqdm

client = OpenAI(
    base_url = "http://sionic.chat:8001/v1",
    api_key = "934c4bbc-c384-4bea-af82-1450d7f8128d"
)

system_prompt = """너는 주어진 상품, 핵심가치에 대한 광고문구를 보고 이를 0.00~1.00점 사이의 실수값으로 평가하는 광고전문가야.
    만약 광고문구가 상품, 핵심가치를 창의적인 방식으로 잘 반영하고 있다면 1점에 가깝게 점수를 줘.
    그렇지 않고 상품, 핵심가치를 광고문구에 그대로 넣어서 광고문구를 작성했다면 0점에 가깝게 점수를 줘.
    아래는 몇 가지 예시야.

    Example 1
    ```````````
    User input: 
		"상품": "통신사",
		"담고자 하는 가치": "혁신적인",
		"카피라이팅": "연결의 힘을 믿습니다"
	Assistant Response:
        1.00
    ```````````

    Example 2
    ```````````
    User input: 
        "상품": "통신사",
		"담고자 하는 가치": "혁신적인",
		"카피라이팅": "혁신적인 통신사를 만듭니다"
    Assistant Response:
       0.00
    ```````````
"""

with open('/home/aikusrv04/aiku/1.3B/HG/generated_data/70B_dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)



output_list = []
for i in tqdm(range(len(dataset))):

    # make user_input with generated dataset
    user_input = f"""
        "상품": {dataset[i]['상품']},
        "담고자 하는 가치": {dataset[i]['담고자 하는 가치']},
        "카피라이팅": {dataset[i]['카피라이팅']}
    """

    response = client.chat.completions.create(
        model="xionic-ko-llama-3-70b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}]
    )

    score = response.choices[0].message.content
    
    output = {
                "상품": dataset[i]['상품'],
                "담고자 하는 가치": dataset[i]['담고자 하는 가치'],
                "카피라이팅": dataset[i]['카피라이팅'],
                "점수" : score
            }

    output_list.append(output)

# JSON 파일에 저장
output_file_path = "/home/aikusrv04/aiku/1.3B/HG/70B_dataset_evaluated.json"
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(output_list, f, indent=4, ensure_ascii=False)