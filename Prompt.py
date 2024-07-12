import json
import random
import pandas as pd
from utils import load_knowledge_base

domain_file_path = 'CrossDomain.json'
with open(domain_file_path) as json_file:
    domain_file = json.load(json_file)
    
SDDT_file_path = 'SDDT.json'
with open(SDDT_file_path) as json_file:
    SDDT_file = json.load(json_file)


class Prompt_maker:
    def __init__(self, question, choices, domain, selected_file):
        self.question = question
        self.choices = choices
        self.domain = domain
        self.selected_file = selected_file
            
            
    def No_ICL_Prompt(self):
        prompt = ""
        prompt += f"Question: {self.question}\n"
        prompt += f"Options:\n"
        prompt += f"A) {self.choices[0]}\n"
        prompt += f"B) {self.choices[1]}\n"
        prompt += f"C) {self.choices[2]}\n"
        prompt += f"D) {self.choices[3]}\n"
        prompt += "Answer:"
        return prompt
    
    def Cross_domain_ICL_Prompt(self, example_num):
        other_domains = [d for d in domain_file.keys() if d != self.domain]
        selected_domains = random.sample(other_domains, 2)
        
        examples = []
        for domain in selected_domains:
            domain_dfs = []
            for file in domain_file[domain]:
                df = load_knowledge_base(file)
                domain_dfs.append(df)
            domain_data = pd.concat(domain_dfs, ignore_index=True)
            examples.append(domain_data)
        
        combined_examples = pd.concat(examples, ignore_index=True)
        
        selected_examples = combined_examples.sample(n=example_num)
        
        icl_examples = ""
        for index, row in selected_examples.iterrows():
            icl_examples += f"Question: {row['question']}\n"
            icl_examples += f"Options:\n"
            icl_examples += f"A) {row['choice_1']}\n"
            icl_examples += f"B) {row['choice_2']}\n"
            icl_examples += f"C) {row['choice_3']}\n"
            icl_examples += f"D) {row['choice_4']}\n"
            icl_examples += f"Answer: {row['answer']}\n\n"
            
        icl_prompt = icl_examples
        icl_prompt += f"Question: {self.question}\n"
        icl_prompt += f"Options:\n"
        icl_prompt += f"A) {self.choices[0]}\n"
        icl_prompt += f"B) {self.choices[1]}\n"
        icl_prompt += f"C) {self.choices[2]}\n"
        icl_prompt += f"D) {self.choices[3]}\n"
        icl_prompt += "Answer:"
        
        return icl_prompt
    
    def Same_domain_ICL_Prompt(self, example_num):
        
        df = load_knowledge_base(self.selected_file)
        df_filtered = df[df['question'] != self.question]
        selected_examples = df_filtered.sample(n=example_num)
        icl_examples = ""
        for index, row in selected_examples.iterrows():
            icl_examples += f"Question: {row['question']}\n"
            icl_examples += f"Options:\n"
            icl_examples += f"A) {row['choice_1']}\n"
            icl_examples += f"B) {row['choice_2']}\n"
            icl_examples += f"C) {row['choice_3']}\n"
            icl_examples += f"D) {row['choice_4']}\n"
            icl_examples += f"Answer: {row['answer']}\n\n"
            
        icl_prompt = icl_examples
        icl_prompt += f"Question: {self.question}\n"
        icl_prompt += f"Options:\n"
        icl_prompt += f"A) {self.choices[0]}\n"
        icl_prompt += f"B) {self.choices[1]}\n"
        icl_prompt += f"C) {self.choices[2]}\n"
        icl_prompt += f"D) {self.choices[3]}\n"
        icl_prompt += "Answer:"
        
        return icl_prompt
    
    def SDDT_ICL_Prompt(self, example_num):
        sddt_files = [file for file in SDDT_file[self.domain] if file != self.selected_file]
        
        examples = []
        for file in sddt_files:
            df = load_knowledge_base(file)
            examples.append(df)
        
        combined_examples = pd.concat(examples, ignore_index=True)
        
        selected_examples = combined_examples.sample(n=example_num)
        
        icl_examples = ""
        for index, row in selected_examples.iterrows():
            icl_examples += f"Question: {row['question']}\n"
            icl_examples += f"Options:\n"
            icl_examples += f"A) {row['choice_1']}\n"
            icl_examples += f"B) {row['choice_2']}\n"
            icl_examples += f"C) {row['choice_3']}\n"
            icl_examples += f"D) {row['choice_4']}\n"
            icl_examples += f"Answer: {row['answer']}\n\n"
            
        icl_prompt = icl_examples
        icl_prompt += f"Question: {self.question}\n"
        icl_prompt += f"Options:\n"
        icl_prompt += f"A) {self.choices[0]}\n"
        icl_prompt += f"B) {self.choices[1]}\n"
        icl_prompt += f"C) {self.choices[2]}\n"
        icl_prompt += f"D) {self.choices[3]}\n"
        icl_prompt += "Answer:"
        
        return icl_prompt
        