import pandas as pd

def load_knowledge_base(path):
    
    csv_root = 'data/val/'
    
    df = pd.read_csv(csv_root+path, header=None)
    df.columns = ['question', 'choice_1', 'choice_2','choice_3','choice_4', 'answer']
    return df