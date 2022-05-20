import pandas as pd
import numpy as np

df = pd.read_csv('./data/metadata.csv', dtype=str, encoding='utf-8')

for i in range(len(df)):
    filename = './data/' + str(i)
    f = open(filename, 'w', encoding='utf-8')
    title = df.iloc[i, 3]
    if title != title:
        title = ''
    abstract = df.iloc[i, 8]
    if abstract != abstract:
        abstract = ''
    f.write(title + '\n' + abstract)
    f.close()
    print(str(i) + '/' + str(len(df)))
