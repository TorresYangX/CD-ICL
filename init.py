from models.GPT_Neo import GPT_Neo
from models.GPT2_Small import GPT2_Small
from models.GPT2_XL import GPT2_XL

from Evaluater import Evaluater

if __name__ == '__main__':
    evaluater = Evaluater(GPT2_XL())
    ICL_TYPE = 'sddt_icl'
    EXAMPLE_NUM = 3
    print(f"Evaluating {ICL_TYPE} with {EXAMPLE_NUM} examples")
    evaluater.evaluate(100, icl_type=ICL_TYPE, example_num=EXAMPLE_NUM)
    
    
