import pandas as pd 
import os


labels_list = range(0, 90)
train_csv = pd.read_csv('../data/train.csv', index_col=0)
for i in labels_list:
    if not os.path.isdir('../data/data/train/'+str(i)):
        print(f'Creating Dir ../data/data/train/{i}')
        os.makedirs('../data/data/train/'+str(i))

    for file_path in train_csv.loc[train_csv['speaker'] == i].file_path:
        file_name = file_path.split(os.path.sep)[-1].rsplit('/', 1)[1]
        origin_path = '../data/data/train/' + file_name
        destiny_path = '../data/data/train/'+ str(i) + '/' + file_name
        os.rename(origin_path, destiny_path)
