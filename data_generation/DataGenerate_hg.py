##
import numpy as np
import os
import torch
import re

## Hugging face 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

## langchain
from langchain.prompts import PromptTemplate



# LLM model for generation
# used_model = "kyujinpy/Ko-PlatYi-6B"

class DataGenerate():

    def __init__(self, model_name = "kyujinpy/Ko-PlatYi-6B"):
    # qunatization_config
        self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
                )


        self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype='auto',
                    quantization_config=self.quantization_config,
                    low_cpu_mem_usage=True,
                    ignore_mismatched_sizes=True
                    )

        self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True
                )

        # self.template = given_template
        
    def _create_prompt_template(self, template):
        if template is None:
            raise ValueError("Template is not provided.")
        

        prompt_template = PromptTemplate(template=template, input_variables=["good", "value"])
        return prompt_template
    
    def _get_model_input(self, template, query):
        # 상품과 핵심가치가 주어진 경우에만 실행
        query_good = query.split('*')[0].split(':')[1]
        query_value = query.split('*')[1].split(':')[1]
        # prompt_template 객체 생성
        prompt = self._create_prompt_template(template)
        # prompt_template에 딕셔너리 값을 적용하여 형식화
        prompt = prompt.format(good = query_good, value = query_value)
        conversation = [
            {"role": "system", "content": "너는 상품과 담고자 하는 가치가 주어지면 이를 반영하는 카피라이팅을 1~2줄로 작성하는 마케팅 전문가야"},
            {'role': 'user', 'content': prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        return prompt
    
    def text_generate(self, template, query):
        prompt = self._get_model_input(template, query)

        max_new_tokens = len(self.tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids[0]) + 128

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            )
        
        output_text = self.tokenizer.decode(outputs[0])
        return output_text
    
    def _output_postprocess(self, output_text):

        # [/INST] 문자열의 인덱스 찾기
        inst_index = output_text.index('[/INST]')
        
        result_text = output_text[inst_index + len('[/INST] '):].split('\n')[0]

        if "<<SYS>>," in result_text:
            result_text = result_text.replace("<<SYS>>,", "")
        
        if len(result_text) > 100:
            result_text = result_text[:100]
        

        return result_text