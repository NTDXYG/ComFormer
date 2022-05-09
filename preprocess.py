import pandas as pd
from tqdm import tqdm
from utils import transformer_pre

def load_dataset(dataset_path) -> list:
    """
    load the dataset from given path
    :param dataset_path: path of dataset
    :return: lines from the dataset
    """
    lines = []
    with open(dataset_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            words = line.strip()
            lines.append(words)
    return lines

for type in ['train', 'valid', 'test']:
    code_list = load_dataset(type+'/'+type+'.token.code')
    nl_list = load_dataset(type+'/'+type+'.token.nl')
    data_list = []
    for i in tqdm(range(len(code_list))):
        try:
            code_seq, sbt = transformer_pre(code_list[i])
            input_text = ' '.join(code_seq.split()[:256]) + '<code>' + ' '.join(sbt.split()[:256])
            data_list.append([input_text, nl_list[i]])
        except:
            pass
    df = pd.DataFrame(data_list, columns=['input_text', 'target_text'])
    df.to_csv(type+'.csv', index=False)