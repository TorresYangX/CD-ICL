from Prompt import Prompt_maker
from models.GPT_Neo import GPT_Neo
from utils import load_knowledge_base
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import json
from tqdm import tqdm

domain_file_path = 'CrossDomain.json'
with open(domain_file_path) as json_file:
    domain_file = json.load(json_file)
    
SDDT_file_path = 'SDDT.json'
with open(SDDT_file_path) as json_file:
    SDDT_file = json.load(json_file)
    
    
def random_select_question(IsSDDT=False):
    if IsSDDT:
        selected_domain = random.choice(list(SDDT_file.keys()))
        selected_file = random.choice(SDDT_file[selected_domain])
        selected_df = load_knowledge_base(selected_file)
        selected_Q = selected_df.sample(n=1).iloc[0]
        question = selected_Q['question']
        choices = [selected_Q['choice_1'], selected_Q['choice_2'], selected_Q['choice_3'], selected_Q['choice_4']]
        answer = selected_Q['answer']
    else:
        selected_domain = random.choice(list(domain_file.keys()))
        selected_file = random.choice(domain_file[selected_domain])
        selected_df = load_knowledge_base(selected_file)
        selected_Q = selected_df.sample(n=1).iloc[0]
        question = selected_Q['question']
        choices = [selected_Q['choice_1'], selected_Q['choice_2'], selected_Q['choice_3'], selected_Q['choice_4']]
        answer = selected_Q['answer']
    return selected_domain, question, choices, answer, selected_file


class Evaluater:
    def __init__(self, model):
        self.model = model
        
    def Cross_domain_ICL(self, example_num):
        selected_domain, question, choices, correct_answer, selected_file = random_select_question(IsSDDT=False)
        promptMaker = Prompt_maker(question, choices, selected_domain, selected_file)
        CD_Icl_prompt = promptMaker.Cross_domain_ICL_Prompt(example_num)
        model_answer = self.model.generate_text(CD_Icl_prompt)
        return model_answer, correct_answer
    
    def No_ICL(self):
        selected_domain, question, choices, correct_answer, selected_file = random_select_question(IsSDDT=False)
        promptMaker = Prompt_maker(question, choices, selected_domain, selected_file)
        No_ICL_prompt = promptMaker.No_ICL_Prompt()
        model_answer = self.model.generate_text(No_ICL_prompt)
        return model_answer, correct_answer
    
    def Same_domain_ICL(self, example_num):
        selected_domain, question, choices, correct_answer, selected_file = random_select_question(IsSDDT=False)
        promptMaker = Prompt_maker(question, choices, selected_domain, selected_file)
        Same_domain_ICL_prompt = promptMaker.Same_domain_ICL_Prompt(example_num)
        model_answer = self.model.generate_text(Same_domain_ICL_prompt)
        return model_answer, correct_answer
    
    def SDDT_ICL(self, example_num):
        selected_domain, question, choices, correct_answer, selected_file = random_select_question(IsSDDT=True)
        promptMaker = Prompt_maker(question, choices, selected_domain, selected_file)
        SDDT_ICL_prompt = promptMaker.SDDT_ICL_Prompt(example_num)
        model_answer = self.model.generate_text(SDDT_ICL_prompt)
        return model_answer, correct_answer
    
    
    def evaluate(self, num_samples, icl_type='no_icl', example_num=0):
        correct = 0
        total = num_samples
        y_true = []
        y_pred = []
        invalid_answers = 0
        
        for _ in tqdm(range(num_samples)):
            if icl_type == 'cross_domain_icl':
                model_answer, correct_answer = self.Cross_domain_ICL(example_num)
            elif icl_type == 'no_icl':
                model_answer, correct_answer = self.No_ICL()
            elif icl_type == 'same_domain_icl':
                model_answer, correct_answer = self.Same_domain_ICL(example_num)
            elif icl_type == 'sddt_icl':
                model_answer, correct_answer = self.SDDT_ICL(example_num)
            else:
                raise ValueError("icl_type should be one of 'cross_domain_icl', 'no_icl', 'same_domain_icl'")

            y_true.append(correct_answer)
            if model_answer:
                y_pred.append(model_answer)
            else:
                y_pred.append("E") #Error
                invalid_answers += 1

            if model_answer == correct_answer:
                correct += 1
                
                
        accuracy = correct / total

        print(f"Accuracy: {accuracy}, Invalid Answers: {invalid_answers}")
        return accuracy, invalid_answers
        
        
    