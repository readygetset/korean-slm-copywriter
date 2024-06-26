![image](https://github.com/joon999/aiku-24-1-DoNotClickUnder1.3B/assets/133974077/01299d7c-6dfd-4882-ac03-957d402fef59)# 1.3B 이하면 클릭하지 마세요

📢 20##년 1/여름/2/겨울학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다
🎉 20##년 1/여름/2/겨울학기 AIKU Conference 열심히상 수상!

## 소개

LLM보다 SLM이 점차 주목받고 있기 때문에,
고품질 데이터셋 구축을 SLM 성능 개선의 핵심이라 가정하고 직접 데이터셋을 생성하여 고성능의 SLM을 구현하는 프로젝트를 진행했습니다.
데이터셋을 기존의 LLM으로 생성할 것이므로,
  1. 기존의 LLM이 잘 하는 task
  2. 생성 토큰 수가 많지 않은 짧은 길이의 생성 task
  3. 창의성 등 기존의 SLM에 남아있는 도전적인 과제를 해결할만한 task

위 3가지 조건을 만족하는 카피라이팅 생성 모델을 구현하기로 하였습니다.


## 방법론

퀄리티 높은 데이터셋을 구축하기 위해서 크게 3가지를 고려했습니다.
 1. 데이터의 양
 2. 데이터의 질
 3. 데이터의 다양성

데이터의 질을 높이기 위해서 gpt4 API를 사용하면 가장 좋겠지만 비용적인 측면에서 한계가 있어 xionic사에서 제공하는 라마3 기반의 xionic-ko-llama-3-70b API를 사용하여 데이터를 생성했습니다.
또한 퀄리티 좋은 실제 카피라이팅 사례를 직접 모아 프롬프트에 예시로서 사용하는 것으로 생성되는 카피라이팅의 질을 월등히 높일 수 있었습니다.
데이터의 다양성 문제를 해결하기 위해 카피라이팅 생성 시 제공되는 상품과 가치 pair의 다양성과 연관성을 높이기 위해 노력했으며, 프롬프트에 사용되는 실제 카피라이팅 사례도 수집된 목록에서 무작위로 선정되도록 했습니다.

이후 구축된 데이터셋으로 언어모델을 파인튜닝하였으며, 비교를 위해 SLM 외에도 bert기반 모델에도 학습을 진행했습니다.

<img src="./asset/models_result.png">

Evaluation을 위해서는 아래 3가지 메트릭을 기반으로 gpt4o API로 confident-ai의 deepeval을 이용하여 평가를 진행했습니다. 

 PV_metric : 상품(P)과 핵심가치(V)가 광고문구에 적절히 반영 되었는가?
 
 Naturalness_metric: 광고문구가 문법적, 의미론적으로 자연스러운가?
 
 Creativity_metric : 광고문구가 창의적인가?

<img src="./asset/models_evaluation.png">


## 환경 설정

(Requirements, Anaconda, Docker 등 프로젝트를 사용하는데에 필요한 요구 사항을 나열해주세요)

## 사용 방법

(프로젝트 실행 방법 (명령어 등)을 적어주세요.)

## 예시 결과

(사용 방법을 실행했을 때 나타나는 결과나 시각화 이미지를 보여주세요)

## 팀원

- [박준하](https://github.com/joon999): 전체 파이프라인 구성, 모델 학습
- [강현규](): evaluation 메트릭 구성, 카피라이팅 생성, 모델 학습
- [서연우](): 상품-가치 pair 구축, 카피라이팅 예시 목록 구성, 모델 학습
- [이민재](): 상품-가치 pair 구축, test data 구성, 모델 학습
