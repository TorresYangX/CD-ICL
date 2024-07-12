import os
import json
from collections import defaultdict

if __name__ == '__main__':

    folder_path = 'data/val/'
    file_names = os.listdir(folder_path)

    categories = {
        'Math': ['algebra', 'mathematics', 'statistics', 'econometrics'],
        'Physics': ['physics', 'astronomy', 'conceptual_physics'],
        'Biology': ['biology', 'anatomy', 'human_aging', 'human_sexuality', 'medical_genetics'],
        'Chemistry': ['chemistry'],
        'Computer Science': ['computer_science', 'machine_learning', 'computer_security'],
        'Engineering': ['electrical_engineering'],
        'Medicine': ['clinical_knowledge', 'college_medicine', 'professional_medicine'],
        'Business': ['business_ethics', 'management', 'marketing', 'accounting'],
        'Law': ['professional_law', 'jurisprudence', 'international_law'],
        'Psychology': ['psychology', 'professional_psychology'],
        'History': ['european_history', 'us_history', 'world_history', 'prehistory'],
        'Geography': ['geography'],
        'Philosophy': ['philosophy', 'moral_disputes', 'moral_scenarios'],
        'Sociology': ['sociology'],
        'Others': ['miscellaneous', 'global_facts', 'security_studies', 'public_relations', 'us_foreign_policy', 'virology', 'world_religions']
    }

    categorized_files = defaultdict(list)

    for file_name in file_names:
        found_category = False
        for category, keywords in categories.items():
            if any(keyword in file_name for keyword in keywords):
                categorized_files[category].append(file_name)
                found_category = True
                break
        if not found_category:
            categorized_files['Others'].append(file_name)

    categorized_files = dict(categorized_files)

    domain_path = 'CrossDomain.json'
    with open(domain_path, 'w') as json_file:
        json.dump(categorized_files, json_file, indent=4)
