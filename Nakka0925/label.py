from ast import Num
import pandas as pd

def get_label(path):
    """path : csvファイルのパス 

    csvファイルからclassとaccessionを取り出す
    clr : classとラベルの対応関係(dict)

    return:
        label = {accession : class}

    """
    clr = {'Insecta' : 0, 'Mammalia' : 1, 'Actinopteri' : 2}
    label = {}

    
    df = pd.read_csv(path, encoding='shift-jis')
    df2 = df['class']

    accession = df['accession']
    
    num = len(df2)
    
    for i in range(num):
        if df2[i] not in clr:
            continue

        label[accession[i]] = clr[df2[i]]
    
    return label